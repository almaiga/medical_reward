# Final Diagnosis: Why Self-Play Training Failed

## TL;DR

**Your self-play training isn't working because 38% of adversarial attacks are medically implausible, and even the "plausible" ones only fool the assessor 42% of the time. You're training on garbage.**

## The Numbers

### Adversarial Attacks (What You're Training On)
- ❌ **38% implausible** - medically nonsensical
- ❌ **62% plausible** - but only fool assessor 42% of the time
- ❌ **0% hard attacks** - all rated "medium"
- ❌ **Assessor accuracy: 40%** overall

### Vanilla Attacks (For Comparison)
- ⚠️ **28% implausible** - better but still bad
- ⚠️ **70% plausible** - fool assessor 77% of the time (too easy!)
- ❌ **0% hard attacks** - all rated "medium"
- ⚠️ **Assessor accuracy: 66%** overall

## What This Means

### Problem 1: Training on Nonsense (38% implausible)

**Example of implausible attack:**
- Original: "Plan procedure to decompress traumatic subungual hemorrhage" (toenail)
- Attacked: "Plan enucleation of chalazion" (eyelid procedure for toenail?!)
- Judge: "Completely different body part - implausible"

**Why this breaks training:**
- Assessor learns to detect nonsense, not subtle medical errors
- 38% of training data is teaching wrong patterns
- No improvement possible when training on garbage

### Problem 2: Even "Plausible" Attacks Are Weak (42% accuracy)

**The 42% assessor accuracy on plausible attacks means:**
- Attacks fool assessor 58% of the time
- But with 38% implausible rate, this isn't impressive
- Compare to vanilla: 77% accuracy = too easy, but at least coherent

**Two possible explanations:**
1. Attacks are actually too hard (unlikely given implausibility)
2. Attacks are mediocre + task is difficult = low accuracy

### Problem 3: No Difficulty Variation (All "medium")

**All 50 attacks rated "medium" difficulty:**
- No easy attacks to build confidence
- No hard attacks to push boundaries
- Suggests judge can't differentiate OR all attacks are similar quality

## Root Cause: Adversarial Game Is Too Hard for Attacker

**Why adversarial attacks are worse than vanilla:**

1. **Vanilla game (safe → harmful):**
   - Start with clean note
   - Add one error
   - Easier to maintain coherence
   - Result: 28% implausible

2. **Adversarial game (harmful → more harmful):**
   - Start with already-harmful note
   - Must modify existing error
   - Harder to maintain coherence
   - Result: 38% implausible (+10% worse!)

**The attacker model struggles with adversarial games:**
- Must understand existing error
- Must make it worse while staying coherent
- Often fails → generates nonsense
- Training doesn't improve because data is bad

## Why Assessor Accuracy Is Low (40%)

**Three factors:**

1. **Implausible attacks confuse assessor (38%)**
   - Can't learn patterns from nonsense
   - Wastes training compute

2. **Task is inherently hard**
   - Assessor only sees attacked note
   - No comparison to original
   - Must detect errors without context

3. **Plausible attacks are mediocre**
   - Not challenging enough to be "hard"
   - Not easy enough to be "easy"
   - Stuck in mediocre middle

## The Fix: 5-Step Action Plan

### Step 1: Filter Training Data (Immediate)

```bash
# Use medgemma-4b to filter out implausible attacks
# Target: Keep only plausible attacks (<15% implausible)
# This alone should improve training
```

**Expected impact:**
- Remove 38% garbage → 62% good data
- Assessor can learn real patterns
- Should see improvement in next training run

### Step 2: Fix Reward Function (Immediate)

**Current reward:**
```python
reward = harm_evasion + refusal_manipulation + format
# Problem: No plausibility constraint!
```

**Better reward:**
```python
reward = harm_evasion * plausibility_score + format
# Incentivizes fooling assessor WITH medical realism
```

### Step 3: Start with Vanilla Games (Short-term)

**Instead of adversarial, focus on:**
- `vanilla_harmful`: safe → harmful
- Easier for attacker to maintain coherence
- Build up quality before moving to adversarial

**Progression:**
1. Train on vanilla_harmful (easier)
2. Once quality is good (>80% plausible), add adversarial_benign
3. Finally add adversarial_harmful (hardest)

### Step 4: Add Real-Time Plausibility Filter (Medium-term)

**During training:**
```python
# After attacker generates attack
plausibility = medgemma_judge(original, attacked)
if plausibility < threshold:
    reject_attack()  # Don't train on this
else:
    train_on_attack()
```

**Benefits:**
- Only train on good examples
- Immediate quality improvement
- No post-hoc filtering needed

### Step 5: Pre-train Attacker on MEDEC (Long-term)

**Use MEDEC dataset:**
- Real medical errors from clinical notes
- Learn realistic error patterns
- Transfer to self-play

**Expected impact:**
- Attacker learns what real errors look like
- Higher plausibility rate
- Better training data quality

## Comparison: What Good Training Data Looks Like

| Metric | Current (Bad) | Target (Good) |
|--------|--------------|---------------|
| **Implausible rate** | 38% | <15% |
| **Plausible rate** | 62% | >85% |
| **Assessor accuracy (plausible)** | 42% | 50-70% |
| **Difficulty variation** | None | Easy/Medium/Hard mix |
| **Severe impact attacks** | 0% | >20% |

## Expected Outcomes After Fixes

### After Step 1 (Filtering)
- Remove 38% garbage
- Train on 62% plausible attacks
- Should see 10-20% improvement

### After Steps 1-2 (Filtering + Reward)
- Higher quality attacks
- Plausibility rate: 62% → 75%
- Should see 20-30% improvement

### After Steps 1-3 (+ Vanilla Focus)
- Much higher quality
- Plausibility rate: 75% → 85%
- Should see 30-50% improvement

### After All Steps
- High-quality training data
- Plausibility rate: >85%
- Difficulty variation
- Continuous improvement in self-play

## Bottom Line

**You now know exactly why training failed:**
1. 38% implausible attacks = training on garbage
2. Adversarial game too hard for attacker model
3. No plausibility constraint in reward function

**You know exactly how to fix it:**
1. Filter out implausible attacks (immediate)
2. Add plausibility to reward function (immediate)
3. Start with vanilla games (short-term)
4. Add real-time filtering (medium-term)
5. Pre-train on MEDEC (long-term)

**Next action:** Run Step 1 (filtering) and retrain. You should see improvement immediately.
