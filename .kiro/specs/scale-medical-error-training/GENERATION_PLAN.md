# Stratified Data Generation Plan (75/25 Split)

## Overview
Generate remaining training data maintaining 75% Educational / 25% Adaptation split with stratification by error type.

## Current State ✅

**Educational SFT:**
- Existing: 398 notes (1,592 examples)
- File: `data/sft_training/20251017_161801_sft_merged.jsonl`

**Game Adaptation:**
- Existing: 125 notes (500 examples)
- File: `data/adaptation/game_format_adaptation.jsonl`

**Total Existing:** 523 notes (2,092 examples)

## Target State 🎯

**Educational SFT (75%):**
- Target: 914 notes (3,656 examples)
- Need: 516 more notes (2,064 examples)
- Stratified by error type:
  - causalOrganism: 47 notes (188 examples)
  - diagnosis: 248 notes (992 examples)
  - management: 441 notes (1,764 examples)
  - pharmacotherapy: 90 notes (360 examples)
  - treatment: 87 notes (348 examples)

**Game Adaptation (25%):**
- Target: 304 notes (1,216 examples)
- Need: 179 more notes (716 examples)
- Stratified by error type:
  - causalOrganism: 15 notes (60 examples)
  - diagnosis: 82 notes (328 examples)
  - management: 147 notes (588 examples)
  - pharmacotherapy: 30 notes (120 examples)
  - treatment: 29 notes (116 examples)

**Total Target:** 1,218 notes (4,872 examples)
**Total Remaining:** 695 notes (2,780 examples)

## Execution Plan 📋

### Step 1: Create Data Splitting Script
**File:** `script/split_medec_stratified.py`

**Purpose:** 
- Load MEDEC training set
- Identify already processed note IDs
- Stratify remaining notes by error type
- Split into 75% educational / 25% adaptation
- Save split assignments to JSON

**Output:**
- `data/splits/educational_remaining.json` - 516 note IDs stratified by error type
- `data/splits/adaptation_remaining.json` - 179 note IDs stratified by error type
- `data/splits/split_summary.json` - Verification statistics

### Step 2: Generate Educational SFT Data
**Command:**
```bash
python3 script/generate_sft_data.py \
    --medec_path data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv \
    --api_provider openai \
    --model gpt-5 \
    --note_ids_file data/splits/educational_remaining.json \
    --output_dir data/sft_training
```

**Expected Output:**
- `data/sft_training/TIMESTAMP_openai_gpt-5_raw.jsonl` (516 notes)
- `data/sft_training/TIMESTAMP_openai_gpt-5_sft.jsonl` (2,064 examples)

**Time Estimate:** ~2-3 hours (4 API calls per note × 516 notes)

### Step 3: Generate Game Adaptation Data
**Command:**
```bash
python3 script/generate_game_format_adaptation.py \
    --medec_path data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv \
    --note_ids_file data/splits/adaptation_remaining.json \
    --output_path data/adaptation/game_adaptation_expansion.jsonl
```

**Expected Output:**
- `data/adaptation/game_adaptation_expansion.jsonl` (716 examples)

**Time Estimate:** ~5-10 minutes (template-based, no API calls)
**Note:** Add `--use_gpt` flag for higher quality reasoning (~$0.10, 1-2 hours)

### Step 4: Merge and Validate
**Command:**
```bash
python3 script/merge_and_validate.py \
    --existing_educational data/sft_training/20251017_161801_sft_merged.jsonl \
    --new_educational data/sft_training/TIMESTAMP_openai_gpt-5_sft.jsonl \
    --existing_adaptation data/adaptation/game_format_adaptation.jsonl \
    --new_adaptation data/adaptation/game_adaptation_expansion.jsonl \
    --output_dir data
```

**Actions:**
1. Merge existing + new educational data
2. Merge existing + new adaptation data
3. Validate stratification maintained
4. Validate format consistency
5. Check for duplicates
6. Generate final statistics

**Output:**
- `data/sft_training/educational_sft_complete.jsonl` (3,656 examples)
- `data/adaptation/game_adaptation_complete.jsonl` (1,216 examples)
- `data/validation_report.json` (quality metrics)

### Step 5: Final Verification
**Checks:**
- ✅ Total examples: 4,872
- ✅ Educational: 3,656 (75%)
- ✅ Adaptation: 1,216 (25%)
- ✅ Error type distribution matches MEDEC (±0.5%)
- ✅ All examples have valid CoT format
- ✅ No duplicate note IDs between splits
- ✅ 4 examples per note (2 attacker + 2 assessor)

## Stratification Verification 🔍

**MEDEC Ground Truth:**
- management: 48.32%
- diagnosis: 27.15%
- pharmacotherapy: 9.84%
- treatment: 9.52%
- causalOrganism: 5.17%

**Educational Target (75%):**
- management: 48.25% (Δ 0.07%)
- diagnosis: 27.13% (Δ 0.02%)
- pharmacotherapy: 9.85% (Δ 0.01%)
- treatment: 9.52% (Δ 0.00%)
- causalOrganism: 5.14% (Δ 0.03%)

**Adaptation Target (25%):**
- management: 48.36% (Δ 0.04%)
- diagnosis: 26.97% (Δ 0.18%)
- pharmacotherapy: 9.87% (Δ 0.03%)
- treatment: 9.54% (Δ 0.02%)
- causalOrganism: 4.93% (Δ 0.24%)

**Maximum Deviation:** 0.24% ✅ Excellent stratification!

## Cost Estimate 💰

**API Calls:**
- Educational: 516 notes × 4 calls = 2,064 calls
- Adaptation: 179 notes × 4 calls = 716 calls
- Total: 2,780 API calls

**GPT-5 Pricing (estimated):**
- Average tokens per call: ~1,500 input + 400 output
- Cost per call: ~$0.02
- Total estimated cost: ~$55-60

## Risk Mitigation 🛡️

1. **API Failures:** Script includes retry logic with exponential backoff
2. **Rate Limits:** Built-in rate limiting (0.05s between calls)
3. **Resume Capability:** Can resume from last processed note ID
4. **Incremental Validation:** Validate after each batch of 100 notes
5. **Backup Strategy:** Save intermediate results every 50 notes

## Success Criteria ✅

- [ ] All 695 remaining notes processed
- [ ] Stratification maintained (max deviation <0.5%)
- [ ] Format validation passes (>95% valid CoT)
- [ ] No duplicate note IDs
- [ ] Educational + Adaptation = 4,872 examples
- [ ] Ready for 3-stage training pipeline

## Next Steps After Generation

1. **Educational SFT Training:** Use `educational_sft_complete.jsonl`
2. **Game Adaptation Training:** Use `game_adaptation_complete.jsonl`
3. **GRPO Self-Play:** Use validation set as seeds
4. **Evaluation:** Test on MEDEC test set

---

**Ready to proceed?** Review this plan and confirm before I start creating the scripts.
