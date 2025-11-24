# Adversarial Attack Evaluation Results

## Executive Summary

**The adversarial attacks have SERIOUS quality issues that explain why self-play training isn't working.**

## Key Findings

### Adversarial Attacks (Current Evaluation)
- **38% implausible** - HIGH rate of unrealistic medical errors
- **Assessor accuracy: 40%** on all attacks, **42% on plausible attacks**
- **All rated "medium" difficulty** - no variation in challenge level
- **No severe impact attacks** - all moderate

### Comparison to Vanilla Attacks (Previous Evaluation)
| Metric | Vanilla Attacks | Adversarial Attacks | Change |
|--------|----------------|---------------------|---------|
| **Implausible rate** | 28% | 38% | ⬆️ +10% WORSE |
| **Assessor accuracy (plausible)** | 77% | 42% | ⬇️ -35% |
| **Hard attacks** | 0% | 0% | No change |
| **Plausible attacks** | 70% | 62% | ⬇️ -8% WORSE |

## Critical Insights

### 1. Why Self-Play Isn't Working

**The 42% assessor accuracy on plausible adversarial attacks reveals:**

❌ **NOT because attacks are appropriately challenging**
- If attacks were good but hard, we'd see high plausibility (>80%) with low accuracy
- Instead: 62% plausible + 42% accuracy = many attacks are just bad

✅ **The real problem: 38% of attacks are medically implausible**
- Training on nonsense teaches the assessor wrong patterns
- The assessor is learning to detect garbage, not subtle medical errors
- This explains why performance doesn't improve

### 2. Adversarial vs Vanilla: Both Have Issues

**Adversarial attacks are WORSE than vanilla:**
- More implausible (38% vs 28%)
- Still no hard attacks
- No variation in difficulty

**Why adversarial is harder for the attacker:**
- Must modify already-harmful notes
- Harder to make coherent changes
- Results in more nonsensical outputs

### 3. The Low Assessor Accuracy (40%) Explained

The 40% accuracy is NOT a sign of good training data. It's because:
- 38% of attacks are implausible → assessor confused by nonsense
- Even plausible attacks only fool assessor 58% of the time
- Task might be too hard without seeing the original note

Compare to vanilla:
- 77% accuracy on plausible attacks = attacks too easy
- 42% accuracy on plausible attacks = attacks implausible or task too hard

## Detailed Breakdown

### Medical Plausibility
```
Adversarial Attacks:
├── Plausible: 31 (62%)
│   └── Assessor accuracy: 42% (13/31 correct)
└── Implausible: 19 (38%)
    └── Assessor accuracy: 37% (7/19 correct)
```

**Problem**: 38% implausible is HIGH
- These are wasted training examples
- Teach assessor wrong patterns
- Explain lack of improvement

### Detection Difficulty
```
All 50 attacks rated: MEDIUM
├── Easy: 0
├── Medium: 50
└── Hard: 0
```

**Problem**: No variation
- No easy attacks to build confidence
- No hard attacks to push boundaries
- All "medium" suggests judge can't differentiate

### Clinical Impact
```
All 50 attacks rated: MODERATE
├── Minor: 0
├── Moderate: 50
└── Severe: 0
```

**Problem**: No high-stakes errors
- Missing the most important cases
- Training doesn't prioritize critical errors

## Root Cause Analysis

### Why Are Adversarial Attacks More Implausible?

**Adversarial game is harder:**
1. Start with already-harmful note
2. Must make it MORE harmful (adversarial_harmful) or LOOK safe (adversarial_benign)
3. Harder to maintain medical coherence
4. Attacker model struggles → generates nonsense

**Evidence:**
- Vanilla: 28% implausible
- Adversarial: 38% implausible
- +10% increase in garbage

### Why Is Assessor Accuracy So Low?

**Three possible reasons:**

1. **Implausible attacks confuse the assessor** (most likely)
   - 38% nonsense → assessor can't learn patterns
   - Even on plausible attacks, only 42% accuracy

2. **Task is too hard without original note**
   - Assessor only sees attacked note
   - Can't compare to original
   - Might be unrealistic to detect subtle changes

3. **Attacks are actually good but very subtle**
   - Unlikely given 38% implausibility
   - Would expect >80% plausible if this were true

## Recommendations

### Immediate Actions

1. **Filter Training Data**
   ```bash
   # Use medgemma-4b to filter out implausible attacks
   # Keep only plausible + medium/hard attacks
   # Target: <15% implausible rate
   ```

2. **Fix Reward Function**
   - Add plausibility penalty
   - Current reward incentivizes fooling assessor, not medical realism
   - Need: `reward = harm_evasion * plausibility_score`

3. **Reduce Adversarial Game Difficulty**
   - Focus on vanilla_harmful (safe → harmful)
   - Easier for attacker to maintain coherence
   - Build up to adversarial games later

### Medium-Term Improvements

4. **Pre-train Attacker on MEDEC**
   - Learn realistic error patterns
   - Examples of actual medical mistakes
   - Transfer to self-play

5. **Use Medgemma as Real-Time Filter**
   - During training, check plausibility
   - Reject implausible attacks immediately
   - Only train on good examples

6. **Curriculum Learning**
   - Start with vanilla_harmful (easier)
   - Progress to adversarial_benign
   - Finally adversarial_harmful (hardest)

### Long-Term Strategy

7. **Rethink Game Design**
   - Maybe assessor SHOULD see original note
   - Or provide "diff" highlighting changes
   - Current task might be too hard

8. **Diversity Bonus**
   - Reward novel attack strategies
   - Prevent repetitive patterns
   - Encourage exploration

9. **Human-in-the-Loop**
   - Sample and label some attacks
   - Use as few-shot examples
   - Guide attacker toward realism

## Comparison Table

| Aspect | Vanilla Attacks | Adversarial Attacks | Verdict |
|--------|----------------|---------------------|---------|
| **Plausibility** | 70% plausible | 62% plausible | Both need improvement |
| **Implausibility** | 28% | 38% | Adversarial WORSE |
| **Assessor Accuracy** | 77% (too easy) | 42% (too hard/bad) | Both problematic |
| **Difficulty Variation** | None (all medium) | None (all medium) | Both lack variation |
| **Training Value** | Low (too easy) | Very Low (implausible) | Adversarial WORSE |

## Conclusion

**Your self-play training isn't improving because:**

1. ❌ **38% of adversarial attacks are medically implausible**
   - Training on garbage
   - Assessor learns wrong patterns
   - No improvement possible

2. ❌ **Even plausible attacks only fool assessor 58% of the time**
   - Not challenging enough
   - Or task is unrealistic

3. ❌ **No variation in difficulty**
   - All "medium"
   - No curriculum
   - No progression

**Next Steps:**
1. Filter out implausible attacks (use medgemma-4b)
2. Fix reward function to incentivize plausibility
3. Consider starting with vanilla_harmful instead of adversarial
4. Add plausibility checking during training

**The good news:** You now know exactly what's wrong and how to fix it!
