# Attack Plausibility Evaluation

Tools to evaluate whether attacks generated during self-play training are medically plausible and appropriately challenging.

## Problem

During self-play training, if attacks are:
- **Too implausible**: The assessor learns to detect obvious errors, not realistic ones
- **Too easy to detect**: The assessor doesn't learn anything meaningful
- **Not medically coherent**: The training data becomes noisy

## Solution

Use medgemma-4b as a medical expert judge to evaluate:
1. **Medical Plausibility**: Is the error realistic?
2. **Detection Difficulty**: How hard is it to spot the error?
3. **Clinical Impact**: What harm could result?

## Quick Start

### 1. Quick Visual Inspection (Recommended First Step)

View a few examples side-by-side to get a sense of attack quality:

```bash
# View 10 random examples
python quick_attack_comparison.py \
    --input "data/Trainer Output Interactions.jsonl" \
    --num-examples 10 \
    --random

# View only successful attacks (fooled the assessor)
python quick_attack_comparison.py \
    --input "data/Trainer Output Interactions.jsonl" \
    --num-examples 10 \
    --filter-successful

# View only failed attacks (caught by assessor)
python quick_attack_comparison.py \
    --input "data/Trainer Output Interactions.jsonl" \
    --num-examples 10 \
    --filter-failed
```

### 2. Full Automated Evaluation

Use medgemma-4b to evaluate attacks with Chain-of-Thought reasoning:

```bash
# Evaluate 50 samples (default)
python evaluate_attack_plausibility.py \
    --input "data/Trainer Output Interactions.jsonl" \
    --max-samples 50

# Evaluate more samples
python evaluate_attack_plausibility.py \
    --input "data/Trainer Output Interactions.jsonl" \
    --max-samples 100

# Use a different judge model
python evaluate_attack_plausibility.py \
    --input "data/Trainer Output Interactions.jsonl" \
    --model "google/medgemma-4b-it"
```

### 3. Using the Shell Script

```bash
# Quick view
./run_attack_evaluation.sh --quick-view

# Full evaluation
./run_attack_evaluation.sh --max-samples 50

# Custom input file
./run_attack_evaluation.sh \
    --input "results/20250930_115652_Qwen_Qwen3-4B_grpo_assessor.jsonl" \
    --max-samples 100
```

## Output

### Quick Comparison Output
- Side-by-side display of original vs attacked notes
- Highlighted differences
- Assessor performance (if available)
- Rewards (if available)

### Full Evaluation Output
CSV file with columns:
- `original_note`: Original clinical note
- `attacked_note`: Modified note with error
- `plausibility`: plausible/implausible
- `difficulty`: easy/medium/hard
- `impact`: minor/moderate/severe
- `reasoning`: Judge's detailed reasoning
- `assessor_label`: What the assessor predicted
- `assessor_correct`: Whether assessor was right
- `total_reward`: Reward received

Plus summary statistics:
- Distribution of plausibility
- Distribution of difficulty
- Distribution of impact
- Assessor performance by difficulty level

## Interpreting Results

### Good Training Data
- **Plausibility**: Mostly "plausible"
- **Difficulty**: Mix of medium/hard (not too many "easy")
- **Assessor Performance**: Should struggle with hard attacks

### Bad Training Data (Needs Improvement)
- **Too many implausible attacks**: Assessor learns wrong patterns
- **Too many easy attacks**: No learning signal
- **High assessor accuracy on implausible attacks**: Wasting compute

## Next Steps Based on Results

### If attacks are too implausible:
- Adjust attacker model temperature (lower = more conservative)
- Add medical plausibility constraints to reward function
- Use few-shot examples of good attacks

### If attacks are too easy:
- Increase attack subtlety in prompt
- Reward harder-to-detect errors more
- Filter out obvious attacks during training

### If attacks are too hard:
- The assessor might need more training data
- Consider if the task is realistic
- Check if errors are actually detectable without original note

## Example Workflow

```bash
# 1. Run evaluation (50 samples takes ~10-15 min on GPU)
python evaluate_attack_plausibility.py \
    --input "data/Trainer Output Interactions.jsonl" \
    --max-samples 50

# 2. If parsing fails, re-parse the results
python reparse_evaluation_results.py \
    --input "results/attack_plausibility_eval_TIMESTAMP.csv"

# 3. Get comprehensive summary with recommendations
python summarize_attack_quality.py \
    --input "results/attack_plausibility_eval_TIMESTAMP.csv"

# 4. Manually inspect interesting cases
python quick_attack_comparison.py \
    --input "data/Trainer Output Interactions.jsonl" \
    --filter-successful \
    --num-examples 10
```

## Files

- `evaluate_attack_plausibility.py`: Main evaluation script using medgemma-4b
- `reparse_evaluation_results.py`: Fix JSON parsing issues in existing results
- `summarize_attack_quality.py`: Comprehensive analysis with recommendations
- `quick_attack_comparison.py`: Quick visual inspection tool
- `analyze_attack_difficulty.py`: Statistical analysis of attack characteristics
- `run_attack_evaluation.sh`: Convenience wrapper script
- `ATTACK_EVALUATION_README.md`: This file

## Requirements

```bash
pip install transformers torch pandas tqdm
```

Make sure you have enough GPU memory for medgemma-4b (requires ~8GB VRAM).
