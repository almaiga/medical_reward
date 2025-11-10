# Quick Start Guide: Stratified Data Generation

This guide walks you through generating the remaining stratified training data for your medical error detection system.

## Prerequisites

- ✅ Existing educational data: `data/sft_training/20251017_161801_sft_merged.jsonl` (1,592 examples)
- ✅ Existing adaptation data: `data/adaptation/game_format_adaptation.jsonl` (500 examples)
- ✅ MEDEC training set: `data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv`
- ✅ OpenAI API key set: `export OPENAI_API_KEY=your_key`

## Step-by-Step Execution

### Step 1: Stratified Data Splitting (2 minutes)

Split remaining MEDEC notes into 75% educational / 25% adaptation:

```bash
python3 script/split_medec_stratified.py \
    --medec_path data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv \
    --educational_data data/sft_training/20251017_161801_sft_merged.jsonl \
    --adaptation_data data/adaptation/game_format_adaptation.jsonl \
    --output_dir data/splits \
    --edu_ratio 0.75 \
    --random_seed 42
```

**Output:**
- `data/splits/educational_remaining.json` - 516 note IDs for educational
- `data/splits/adaptation_remaining.json` - 179 note IDs for adaptation
- `data/splits/split_summary.json` - Statistics

**Verify:**
- Check that stratification deviation is <0.5%
- Confirm no overlap between splits
- Review error type distribution

---

### Step 2: Generate Educational SFT Data (2-3 hours)

Generate educational examples using GPT-5:

```bash
python3 script/generate_sft_data.py \
    --medec_path data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv \
    --api_provider openai \
    --model gpt-5 \
    --note_ids_file data/splits/educational_remaining.json \
    --output_dir data/sft_training
```

**Progress:**
- 516 notes × 4 examples = 2,064 API calls
- ~7-10 seconds per call
- Total: ~2-3 hours

**Output:**
- `data/sft_training/TIMESTAMP_openai_gpt-5_raw.jsonl`
- `data/sft_training/TIMESTAMP_openai_gpt-5_sft.jsonl` ← Use this one

**Cost:** ~$40-50

**Monitor:**
- Watch for API errors (script has retry logic)
- Check success rate (should be >90%)
- Can resume with `--start_from ms-train-XXX` if interrupted

---

### Step 3: Generate Adaptation Data (5-10 minutes)

Generate adaptation examples using templates (fast, free):

```bash
python3 script/generate_game_format_adaptation.py \
    --medec_path data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv \
    --note_ids_file data/splits/adaptation_remaining.json \
    --output_path data/adaptation/game_adaptation_expansion.jsonl
```

**Progress:**
- 179 notes × 4 examples = 716 examples
- Template-based (no API calls)
- Total: ~5-10 minutes

**Output:**
- `data/adaptation/game_adaptation_expansion.jsonl`

**Cost:** $0 (template-based)

**Optional:** Add `--use_gpt` for higher quality reasoning (~$0.10, 1-2 hours)

---

### Step 4: Merge and Validate (1 minute)

Merge old + new data and validate quality:

```bash
python3 script/merge_and_validate.py \
    --existing_educational data/sft_training/20251017_161801_sft_merged.jsonl \
    --new_educational data/sft_training/TIMESTAMP_openai_gpt-5_sft.jsonl \
    --existing_adaptation data/adaptation/game_format_adaptation.jsonl \
    --new_adaptation data/adaptation/game_adaptation_expansion.jsonl \
    --output_dir data
```

**Replace `TIMESTAMP`** with the actual timestamp from Step 2.

**Output:**
- `data/sft_training/educational_sft_complete.jsonl` (3,656 examples)
- `data/adaptation/game_adaptation_complete.jsonl` (1,216 examples)
- `data/validation_report.json`

**Validation Checks:**
- ✅ Format valid: >95%
- ✅ No duplicates (4 examples per note)
- ✅ Stratification maintained
- ✅ Error type distribution matches MEDEC

---

## Final Verification

Check the validation report:

```bash
cat data/validation_report.json | python3 -m json.tool
```

**Expected Results:**
- Educational: 3,656 examples (914 notes)
- Adaptation: 1,216 examples (304 notes)
- Total: 4,872 examples (1,218 notes)
- Format valid: >95%
- Split ratio: 75% / 25%

---

## Next Steps: Training Pipeline

Once data generation is complete:

### 1. Educational SFT Training

```bash
python3 script/train_qwen3_trl.py \
    --model_id Qwen/Qwen2.5-4B-Instruct \
    --data_path data/sft_training/educational_sft_complete.jsonl \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --output_dir trainer_output/qwen3_educational_full
```

**Time:** ~3-4 hours

### 2. Game Adaptation Training

```bash
python3 script/train_qwen3_trl.py \
    --model_id trainer_output/qwen3_educational_full \
    --data_path data/adaptation/game_adaptation_complete.jsonl \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --output_dir trainer_output/qwen3_game_adapted
```

**Time:** ~1-2 hours

### 3. GRPO Self-Play Training

```bash
python3 script/train_selfplay_advanced.py \
    --model_id trainer_output/qwen3_game_adapted \
    --judge_model_id google/medgemma-4b-it \
    --num_samples 128 \
    --num_generations 4 \
    --rounds 5 \
    --learning_rate 1e-5
```

**Time:** ~4-6 hours

---

## Troubleshooting

### API Rate Limits

If you hit rate limits:
```bash
# Resume from last processed ID
python3 script/generate_sft_data.py \
    --note_ids_file data/splits/educational_remaining.json \
    --start_from ms-train-XXX \
    ...
```

### Low Success Rate

If success rate <90%:
- Check API key is valid
- Check internet connection
- Increase retry delay in script

### Format Validation Fails

If format validation <95%:
- Check GPT-5 API responses
- Review invalid examples in validation report
- May need to regenerate problematic examples

### Stratification Issues

If deviation >0.5%:
- Re-run split script with different random seed
- Check that note IDs are correctly filtered

---

## Summary

**Total Time:** ~3-4 hours (mostly Step 2)
**Total Cost:** ~$40-50 (GPT-5 API calls)
**Total Output:** 4,872 examples (1,218 notes)

**Key Benefits:**
- ✅ Stratified by error type (maintains MEDEC distribution)
- ✅ 75/25 split (educational/adaptation)
- ✅ Reuses existing 2,092 examples
- ✅ Only generates 2,780 new examples
- ✅ Ready for 3-stage training pipeline
