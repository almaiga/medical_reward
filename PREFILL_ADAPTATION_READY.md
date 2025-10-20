# Pre-Fill Adaptation Training - Ready to Go!

## What I Created

### 1. Data Generation Script
**File**: `script/generate_prefill_adaptation_data.py`

Generates adaptation data in **PRE-FILL format** to match your educational SFT:
- Format: `<think>reasoning</think><output>response</output>`
- Uses ms-train-733 to ms-train-932 (200 rows)
- Creates 800 examples (4 per row)
- Template-based reasoning (no LLM calls needed)

### 2. Training Script
**File**: `run_sft_prefill_adaptation.sh`

Runs adaptation training with:
- 1 epoch (prevent overfitting)
- Learning rate 1e-5 (half of original SFT)
- Batch size 4, gradient accumulation 4
- Auto-generates data if not present

## Why Pre-Fill Format?

**Your model has**:
- ✅ 1592 educational SFT examples (pre-fill format)
- ✅ 400 post-fill adaptation examples (post-fill format)

**Result**: Model is **4x more familiar** with pre-fill format!

**Pre-fill format**:
```
<think>I'm changing dosage from 100mg to 200mg</think>
<output>Patient prescribed 200mg aspirin</output>
```

This matches your educational SFT, so the model won't be confused.

## How to Use

### Step 1: Generate Adaptation Data
```bash
python3 script/generate_prefill_adaptation_data.py \
    --start_id 733 \
    --num_rows 200 \
    --output_path data/adaptation/prefill_cot_adaptation.jsonl
```

**Output**: 800 examples (200 rows × 4 examples per row)

### Step 2: Run Adaptation Training
```bash
bash run_sft_prefill_adaptation.sh
```

Or specify a different base model:
```bash
bash run_sft_prefill_adaptation.sh trainer_output/qwen3_trl_20251020_142117
```

**Time**: ~10-15 minutes

### Step 3: Run GRPO
```bash
bash run_selfplay_training.sh trainer_output/qwen3_adapted_prefill_TIMESTAMP
```

## Data Format Examples

### Attacker Harmful
```json
{
  "messages": [
    {"role": "system", "content": "You are testing medical safety..."},
    {"role": "user", "content": "Create an error variation..."},
    {"role": "assistant", "content": "<think>Changed medication from aspirin to ibuprofen</think>\n<output>Patient prescribed ibuprofen 100mg daily</output>"}
  ]
}
```

### Assessor Harmful
```json
{
  "messages": [
    {"role": "system", "content": "You classify medical notes..."},
    {"role": "user", "content": "Classify this medical note..."},
    {"role": "assistant", "content": "<think>Medication error detected</think>\n<output>Harmful</output>"}
  ]
}
```

## Expected Results

After adaptation training + GRPO:

✅ **Attacker outputs**: `<think>reasoning</think><output>modified note</output>`
✅ **Assessor outputs**: `<think>analysis</think><output>Safe/Concerning/Harmful</output>`
✅ **Parser extracts correctly**: Both thinking and output
✅ **Faithfulness**: 60-80% pass rate
✅ **Reward variance**: Non-zero (enables learning)
✅ **No mode collapse**: Model stays on task

## Advantages Over Post-Fill

1. **Consistency**: Matches 1592 educational SFT examples
2. **No confusion**: Model knows one format well
3. **Proven**: Your educational SFT already works with this format
4. **Simpler**: No need to retrain on new format

## Files Created

- `script/generate_prefill_adaptation_data.py` - Data generator
- `run_sft_prefill_adaptation.sh` - Training script
- `PREFILL_ADAPTATION_READY.md` - This document

## GRPO Already Updated

The GRPO prompts in `script/train_selfplay_advanced.py` are already updated to use pre-fill format, so everything will work together!

Ready to run! 🚀
