# Training Collapse Analysis & Solution

## Problem Summary

Your model generates garbage ("I'm here. I'm not scared...") instead of medical notes because:

**SFT Format ≠ GRPO Format** (Distribution Shift)

## SFT Training (What Model Learned)

**Attacker Examples (796)**:
- Prompt: "I'm developing a medical error detection system... [shows original + error]"
- Response: Long educational analysis + reproduce error
- Style: Educational, verbose, shows both versions

**Assessor Examples (796)**:
- Prompt: "I need to train an AI to assess... Expected classification: Harmful"
- Response: Detailed systematic analysis + classification
- Style: Educational, comprehensive medical reasoning

## GRPO Expectations (What Model Sees)

**Attacker Phase**:
- Prompt: "Create a variation of the seed note with ONE substitution error"
- Expected: Brief thinking + modified note
- Style: Direct, task-focused, NO educational framing

**Assessor Phase**:
- Prompt: "Classify this medical note for errors"
- Expected: Brief analysis + Safe/Concerning/Harmful
- Style: Direct, concise classification

## Why Model Fails

1. **No "I'm developing..." framing** → Model doesn't recognize task
2. **No original+error shown** → Model has no reference
3. **Different prompt structure** → Out of distribution
4. **Shorter, direct style** → Model trained on verbose educational text

Result: Model generates safe repetitive text instead of attempting unfamiliar task

## Solution: Adaptation Phase

Create 500-800 examples matching GRPO's EXACT format, train for 1 epoch.

### Data Sources
- **Validation set**: ms-val-0 to ms-val-573 (574 cases, 319 with errors)
- **Available**: Clean notes (Corrected Text) + Error notes (Text) + Error types
- **No overlap** with training data (ms-train-0 to ms-train-732)

### What to Generate

**Type A: Attacker Format (300-400 examples)**
Match `build_attacker_prompts()` exactly:
- System: "You are testing medical safety. Create a variation..."
- User: "Create an error variation of this seed note: [note]"
- Assistant: "<think>Changing X to Y</think><output>[modified note]</output>"

**Type B: Assessor Format (300-400 examples)**
Match `make_assessor_prompts()` exactly:
- System: "You classify medical notes for safety errors..."
- User: "Classify this medical note for errors: [note]"
- Assistant: "<think>brief analysis</think><output>Safe/Concerning/Harmful</output>"

### Key Requirements

1. **Identical system prompts** to GRPO
2. **Short, concise thinking** (not educational essays)
3. **Both game types**: harmful (create error) + safe (keep clean)
4. **Correct format 100%** of the time
5. **Use validation set** (ms-val-*) to avoid data leakage

### Training Settings

```bash
python3 script/train_adaptation.py \
    --base_model trainer_output/qwen3_trl_20251020_142117 \
    --adaptation_data data/adaptation/grpo_format.jsonl \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --output_dir trainer_output/qwen3_adapted
```

- **1 epoch only** (prevent forgetting SFT knowledge)
- **Lower LR** (1e-5 vs 2e-5) to preserve base capabilities
- **Quick training** (~10-15 minutes for 600 examples)

### Expected Results

After adaptation:
- Model recognizes GRPO prompt format
- Generates valid `<think>` and `<output>` tags
- 60-80% faithfulness pass rate (vs current 0%)
- Non-zero reward variance (enables GRPO learning)
- No mode collapse

## Next Steps

1. Create adaptation data generation script
2. Generate 600 examples from validation set
3. Run 1-epoch adaptation training
4. Test with GRPO (should see immediate improvement)
