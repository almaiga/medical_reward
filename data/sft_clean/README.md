# Clean Stratified Training Data

This directory contains properly stratified educational and adaptation data for medical error detection training.

## Files

- `educational_stratified.jsonl` - Educational SFT data (913 notes, 75% of MEDEC)
- `adaptation_stratified.jsonl` - Game format adaptation data (306 notes → 1,224 examples, 25% of MEDEC)
- `SUMMARY.json` - Detailed statistics and validation

## Statistics

### Educational Data
- **Total examples**: 2083
- **Unique notes**: 913 (75% of MEDEC)
- **Error types**: All 5 types included
  - causalOrganism: 47
  - diagnosis: 248
  - management: 519
  - pharmacotherapy: 132
  - treatment: 133

### Adaptation Data
- **Total examples**: 1224
- **Unique notes**: 306 (25% of MEDEC)
- **Format**: Game format with attacker/assessor roles
- **Style split**: 75% clean AI-style / 25% messy human-style (for safe examples)
- **Error types**: All 5 types included
  - causalOrganism: 32
  - diagnosis: 166
  - management: 296
  - pharmacotherapy: 60
  - treatment: 58

## Validation

- ✓ Proper 75/25 stratification
- ✓ All 5 error types in both datasets
- ✓ No overlap between educational and adaptation
- ✓ 100% MEDEC coverage (1219/1219 notes)

## Usage

### Educational SFT Training
```bash
python3 script/train_qwen3_sft.py \
  --data_path data/sft_clean/educational_stratified.jsonl \
  --epochs 3 \
  --batch_size 4 \
  --output_dir trainer_output/qwen3_educational
```

### Adaptation Training
```bash
python3 script/train_qwen3_sft.py \
  --model_id trainer_output/qwen3_educational \
  --data_path data/sft_clean/adaptation_stratified.jsonl \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --output_dir trainer_output/qwen3_game_adapted
```

### GRPO Self-Play Training
```bash
python3 script/train_selfplay_advanced.py \
  --model_id trainer_output/qwen3_game_adapted \
  --num_samples 16 \
  --rounds 3
```

## Generation Details

- **Educational data**: Merged from 3 sources with proper stratification
  - 20251107_044104_openai_gpt-5_sft.jsonl
  - 20251017_161801_sft_merged.jsonl
  - 20251107_142710_openai_gpt-5_sft.jsonl (supplemental)

- **Adaptation data**: Generated with GPT-5 in parallel
  - Strategic reasoning for attacker and assessor roles
  - Pre-fill CoT format
  - 75/25 AI-style/human-style split for robustness

Generated: 1762947515.1320171
