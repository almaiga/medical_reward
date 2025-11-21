# Attack Plausibility Evaluation - Summary

## Problem Statement

Your self-play training showed no improvement. You suspected that:
1. Attacks might not be medically plausible
2. Attacks might be too easy to detect
3. The assessor isn't learning anything meaningful

## Solution

Created evaluation tools using medgemma-4b as a medical expert judge to assess:
- **Medical Plausibility**: Are errors realistic?
- **Detection Difficulty**: How hard to spot?
- **Clinical Impact**: Potential harm level?

## Dataset Prepared

**File**: `results/selfplay/adversarial_all.jsonl`

- **150 adversarial interactions** (both harmful and benign)
- **64 adversarial_harmful**: Make harmful notes MORE harmful
- **86 adversarial_benign**: Make harmful notes LOOK safe
- **Assessor baseline**: 26.7% accuracy (very challenging!)

## Tools Created

1. **`evaluate_attack_plausibility.py`** - Main evaluation with medgemma-4b + CoT
2. **`reparse_evaluation_results.py`** - Fix JSON parsing issues
3. **`summarize_attack_quality.py`** - Comprehensive analysis + recommendations
4. **`quick_attack_comparison.py`** - Visual side-by-side inspection
5. **`analyze_attack_difficulty.py`** - Statistical analysis
6. **`filter_game_type.py`** - Filter by game category

## Quick Start

```bash
# Run evaluation on 50 adversarial examples
./run_attack_evaluation.sh

# After completion, analyze results
python summarize_attack_quality.py \
    --input "results/attack_plausibility_eval_TIMESTAMP.csv"
```

## What You'll Learn

### From Previous Evaluation (vanilla attacks)
- 77% assessor accuracy on plausible attacks → **too easy**
- 28% implausible attacks → **noisy training data**
- No hard attacks → **not challenging enough**
- Reward function incentivizing implausible attacks

### From This Evaluation (adversarial attacks)
Will reveal:
- Are adversarial attacks more medically plausible?
- Why is assessor accuracy only 27%? (too hard vs implausible)
- Do adversarial_harmful and adversarial_benign differ in quality?
- Should you focus training on one game type?

## Expected Outcomes

### If adversarial attacks are better:
- More medically plausible than vanilla attacks
- Appropriately challenging (not too easy/hard)
- Good training signal for the assessor

### If adversarial attacks have issues:
- Still implausible → Need better attacker model
- Too hard → Task might be unrealistic
- Too easy → Need harder attack strategies

## Next Steps Based on Results

1. **Filter training data** - Keep only plausible, medium/hard attacks
2. **Adjust rewards** - Incentivize plausibility + difficulty
3. **Use medgemma as filter** - Real-time plausibility checking during training
4. **Pre-train on MEDEC** - Learn realistic error patterns
5. **Curriculum learning** - Start easy, increase difficulty

## Files Reference

- `RUN_ADVERSARIAL_EVALUATION.md` - Detailed guide for adversarial evaluation
- `ATTACK_EVALUATION_README.md` - General evaluation documentation
- `results/selfplay/adversarial_all.jsonl` - Filtered adversarial dataset
- `results/Attack Plausibility Evaluation Nov 21 2025.csv` - Previous vanilla results

## Key Insight

The 26.7% assessor accuracy on adversarial attacks (vs 66% on vanilla) suggests:
- Adversarial attacks are MUCH harder
- This could be good (challenging training) or bad (implausible/impossible)
- The evaluation will tell you which!
