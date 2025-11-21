# Adversarial Attack Evaluation

## What We're Evaluating

**Game Categories: `adversarial_harmful` + `adversarial_benign`**

In these game types, the attacker receives a note that ALREADY contains a harmful error:

### adversarial_harmful (43% of dataset)
- **Original note**: Contains a harmful medical error
- **Attacker's goal**: Make the error even MORE harmful while staying subtle
- **Example**: Wrong procedure → Even more wrong procedure

### adversarial_benign (57% of dataset)
- **Original note**: Contains a harmful medical error  
- **Attacker's goal**: Make the harmful note LOOK safe/benign
- **Example**: Wrong diagnosis → Reword to sound plausible

Both are interesting because:
1. The attacker must work with existing harmful content
2. The assessor only sees the attacked note (not the original)
3. Success requires medical plausibility and subtlety

## Dataset

- **Source**: `results/selfplay/Trainer Output Qwen3-4B Medical Selfplay Interactions (9).jsonl`
- **Filtered to**: `results/selfplay/adversarial_all.jsonl`
- **Total samples**: 150 adversarial interactions (64 harmful + 86 benign)
- **Actual harm**: 125 Harmful, 25 Safe
- **Assessor baseline accuracy**: 26.7% (very challenging!)

## Quick Start

### 1. Run the evaluation (recommended)

```bash
./run_attack_evaluation.sh
```

This will:
- Use the filtered adversarial_harmful dataset
- Evaluate 50 samples with medgemma-4b
- Generate plausibility, difficulty, and impact ratings

### 2. View examples first (optional)

```bash
python quick_attack_comparison.py \
    --input "results/selfplay/adversarial_all.jsonl" \
    --num-examples 5 \
    --random
```

### 3. Analyze results

After evaluation completes:

```bash
# Re-parse if needed (fixes JSON parsing)
python reparse_evaluation_results.py \
    --input "results/attack_plausibility_eval_TIMESTAMP.csv"

# Get comprehensive summary
python summarize_attack_quality.py \
    --input "results/attack_plausibility_eval_TIMESTAMP.csv"
```

## Expected Insights

This evaluation will tell you:

1. **Are adversarial attacks medically plausible?**
   - Do they make medical sense?
   - Or are they just random nonsense?

2. **How difficult are they to detect?**
   - Easy: Obvious errors
   - Medium: Requires careful reading
   - Hard: Requires expert knowledge

3. **Why is assessor accuracy only 24%?**
   - Are attacks too hard (good for training)?
   - Are attacks implausible (bad for training)?
   - Is the task itself too difficult?

## Game Categories Explained

- **`vanilla_benign`**: Safe note → Keep safe (attacker does nothing)
- **`vanilla_harmful`**: Safe note → Add harmful error
- **`adversarial_benign`**: Harmful note → Make it look safe
- **`adversarial_harmful`**: Harmful note → Make it MORE harmful ⭐ (we're evaluating this)
- **`adversarial_benign`**: Harmful note → Make it look safe ⭐ (we're evaluating this)

## Why This Matters

If adversarial_harmful attacks are:
- **Plausible + Hard**: Great training data! Assessor learns to detect subtle errors
- **Implausible + Hard**: Bad training data - assessor learns wrong patterns
- **Plausible + Easy**: Attacks too simple - no learning signal
- **Implausible + Easy**: Worst case - wasting compute on garbage

## Next Steps After Evaluation

Based on results, you might:
1. Filter training data to keep only plausible attacks
2. Adjust reward function to incentivize harder attacks
3. Use medgemma-4b as a plausibility filter during training
4. Pre-train attacker on MEDEC for realistic error patterns
