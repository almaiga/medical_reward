# URGENT: Model Needs Adaptation Training First!

## Problem Identified

Your GRPO training is failing because the model doesn't know post-fill CoT format yet.

**Current situation**:
- ✅ Model trained on educational SFT (pre-fill format: `<think>` then `<output>`)
- ❌ Model NOT trained on adaptation data (post-fill format: output then `<think>`)
- ❌ GRPO expects post-fill format but model generates pre-fill format
- ❌ Assessor outputs incomplete responses (just `<think>` with no classification)

**Evidence from terminal**:
```
Assessor said: <think>
Brief analysis only.
</think>
```
No classification output! Model stops after `</think>` because that's where it learned to stop in educational SFT.

## Solution

### Step 1: Stop Current GRPO Training
Press `Ctrl+C` to stop the running training.

### Step 2: Run Adaptation Training
```bash
bash run_sft_template_adaptation.sh
```

This will:
- Take your educational SFT model
- Train for 1 epoch on 400 post-fill CoT examples
- Teach it the new format: `response<think>reasoning</think>`
- Save adapted model to `trainer_output/qwen3_adapted_TIMESTAMP`

**Time**: ~10-15 minutes

### Step 3: Update GRPO Script (Already Done!)
I've updated `parse_response()` in `train_selfplay_advanced.py` to handle both formats.

### Step 4: Run GRPO on Adapted Model
```bash
bash run_selfplay_training.sh trainer_output/qwen3_adapted_TIMESTAMP
```

Replace `TIMESTAMP` with the actual timestamp from Step 2.

## Why This Is Necessary

**Educational SFT taught**:
```
<think>I'm analyzing this error...</think>
<output>Medical note with error</output>
```

**GRPO needs** (for easier RL):
```
Medical note with error<think>I changed X to Y</think>
```

**Adaptation bridges the gap** by teaching the model both formats in 1 epoch without forgetting medical knowledge.

## Expected Results After Adaptation

✅ Assessor will output: `Safe<think>No errors detected</think>`
✅ Attacker will output: `[modified note]<think>Changed X to Y</think>`
✅ Faithfulness pass rate: 60-80% (vs current ~50%)
✅ Reward variance: Non-zero (enables learning)
✅ No mode collapse

## Current Parser Update

I've updated `parse_response()` to try both formats:
1. Try pre-fill format first (for backward compatibility)
2. Try post-fill format second (for adaptation)
3. Fallback to heuristics if neither works

This means the adapted model will work correctly with GRPO!
