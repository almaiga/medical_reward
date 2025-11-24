# Plausibility Reward Design: Analysis & Trade-offs

## Current Reward Structure

### Attacker Rewards (Range: -4 to +4)
```
RA = RA_res_harm + RA_res_refusal + RA_revision + RA_format

Components:
1. RA_res_harm (±1.0): Zero-sum - wins if assessor is wrong
2. RA_res_refusal (±1.0): Zero-sum - wins if assessor misclassifies
3. RA_revision (±1.0): Shaping - must respect game category intent
4. RA_format (±1.0): Shaping - must use proper CoT format
```

### Current Issues
- ✅ Incentivizes fooling assessor (res_harm, res_refusal)
- ✅ Enforces game category faithfulness (revision)
- ✅ Enforces format (format)
- ❌ **NO plausibility constraint** → 38% implausible attacks
- ❌ Attacker can win by generating medical nonsense

## Proposed Solutions

### Option 1: Add Plausibility as Multiplicative Factor

```python
RA_plausibility = medgemma_judge(original, attacked)  # Returns 0.0 to 1.0
RA_total = (RA_res_harm + RA_res_refusal) * RA_plausibility + RA_revision + RA_format
```

**Pros:**
- ✅ **Strong incentive**: Implausible attacks get ~0 reward even if they fool assessor
- ✅ **Preserves zero-sum**: Only affects game rewards, not shaping terms
- ✅ **Intuitive**: "You only get credit for fooling assessor if attack is plausible"
- ✅ **Scales naturally**: Partially plausible attacks get partial credit

**Cons:**
- ❌ **Expensive**: Requires medgemma inference for every attack (~2-3 sec/attack)
- ❌ **Training slowdown**: Could 2-3x training time
- ❌ **Gradient issues**: Multiplicative factor might make learning harder
- ❌ **Breaks zero-sum**: Attacker and assessor rewards no longer sum to zero
- ❌ **Judge reliability**: Medgemma might be inconsistent (we saw some errors)

**Expected Impact:**
- Implausibility rate: 38% → 10-15%
- Training time: 1x → 2-3x
- Quality improvement: High

---

### Option 2: Add Plausibility as Additive Penalty

```python
RA_plausibility = medgemma_judge(original, attacked)  # Returns 0.0 to 1.0
RA_penalty = -2.0 * (1 - RA_plausibility)  # Range: 0 to -2.0
RA_total = RA_res_harm + RA_res_refusal + RA_revision + RA_format + RA_penalty
```

**Pros:**
- ✅ **Simpler gradient**: Additive is easier to optimize than multiplicative
- ✅ **Tunable**: Can adjust penalty magnitude (-1.0, -2.0, -3.0)
- ✅ **Preserves game dynamics**: Zero-sum game still works
- ✅ **Clear signal**: Implausible = large negative penalty

**Cons:**
- ❌ **Still expensive**: Requires medgemma inference
- ❌ **Weaker incentive**: Attacker might accept penalty if it fools assessor
- ❌ **Balance issues**: Need to tune penalty magnitude carefully
- ❌ **Range expansion**: Total reward range becomes -6 to +4 (asymmetric)

**Expected Impact:**
- Implausibility rate: 38% → 15-20%
- Training time: 1x → 2-3x
- Quality improvement: Medium-High

---

### Option 3: Binary Plausibility Threshold (Filter)

```python
RA_plausibility = medgemma_judge(original, attacked)  # Returns 0.0 to 1.0
if RA_plausibility < 0.5:  # Implausible
    RA_total = -4.0  # Worst possible score
else:  # Plausible
    RA_total = RA_res_harm + RA_res_refusal + RA_revision + RA_format
```

**Pros:**
- ✅ **Strong deterrent**: Implausible attacks get worst possible score
- ✅ **Simple**: Binary decision, easy to understand
- ✅ **Fast convergence**: Clear boundary between good/bad
- ✅ **Preserves dynamics**: Plausible attacks use normal reward

**Cons:**
- ❌ **Still expensive**: Requires medgemma inference
- ❌ **Harsh**: No partial credit for "somewhat plausible"
- ❌ **Threshold sensitivity**: Where to set the boundary? (0.5? 0.6? 0.7?)
- ❌ **Cliff effect**: Small changes in plausibility cause large reward jumps

**Expected Impact:**
- Implausibility rate: 38% → 5-10%
- Training time: 1x → 2-3x
- Quality improvement: Very High (but might be too harsh)

---

### Option 4: Lightweight Heuristic Penalty (No LLM)

```python
def compute_plausibility_heuristic(original, attacked):
    """Fast heuristic checks without LLM"""
    score = 1.0
    
    # Check 1: Reasonable edit distance (not too different)
    edit_ratio = levenshtein(original, attacked) / max(len(original), len(attacked))
    if edit_ratio > 0.5:  # Changed >50% of text
        score *= 0.5
    
    # Check 2: Maintains medical terms
    medical_terms = extract_medical_terms(original)
    preserved = sum(1 for term in medical_terms if term in attacked)
    if preserved / len(medical_terms) < 0.7:  # Lost >30% of medical terms
        score *= 0.7
    
    # Check 3: No repeated garbage patterns
    if attacked.count("</think>") > 2 or attacked.count("<think>") > 2:
        score = 0.0
    
    # Check 4: Reasonable length
    length_ratio = len(attacked) / len(original)
    if length_ratio < 0.5 or length_ratio > 2.0:
        score *= 0.5
    
    return score

RA_plausibility = compute_plausibility_heuristic(original, attacked)
RA_total = (RA_res_harm + RA_res_refusal) * RA_plausibility + RA_revision + RA_format
```

**Pros:**
- ✅ **Fast**: No LLM inference, ~0.001 sec/attack
- ✅ **No training slowdown**: Negligible overhead
- ✅ **Catches obvious garbage**: Repeated tags, extreme edits, etc.
- ✅ **Tunable**: Can add/remove heuristics easily
- ✅ **Deterministic**: Same input always gives same score

**Cons:**
- ❌ **Weak signal**: Can't detect subtle medical implausibility
- ❌ **Heuristic limitations**: Might miss sophisticated but implausible attacks
- ❌ **False positives**: Might penalize valid large edits
- ❌ **Incomplete**: Won't solve the full 38% implausibility problem

**Expected Impact:**
- Implausibility rate: 38% → 25-30% (catches obvious garbage only)
- Training time: 1x (no slowdown)
- Quality improvement: Low-Medium

---

### Option 5: Hybrid Approach (Heuristic + Sampled LLM)

```python
# Fast heuristic for all attacks
heuristic_score = compute_plausibility_heuristic(original, attacked)

# If heuristic is suspicious, use LLM to verify
if heuristic_score < 0.7:  # Potentially implausible
    llm_score = medgemma_judge(original, attacked)
    RA_plausibility = llm_score
else:  # Heuristic says it's fine
    # Randomly sample 10% for LLM verification (quality monitoring)
    if random.random() < 0.1:
        llm_score = medgemma_judge(original, attacked)
        RA_plausibility = llm_score
    else:
        RA_plausibility = heuristic_score

RA_total = (RA_res_harm + RA_res_refusal) * RA_plausibility + RA_revision + RA_format
```

**Pros:**
- ✅ **Best of both worlds**: Fast heuristic + accurate LLM when needed
- ✅ **Moderate cost**: Only ~20-30% of attacks need LLM
- ✅ **Catches everything**: Heuristic catches obvious, LLM catches subtle
- ✅ **Quality monitoring**: Random sampling tracks overall quality
- ✅ **Adaptive**: Can adjust sampling rate based on quality

**Cons:**
- ❌ **Complex**: More moving parts, harder to debug
- ❌ **Inconsistent**: Some attacks get LLM, others don't
- ❌ **Still some cost**: 20-30% LLM calls = 1.2-1.3x training time
- ❌ **Tuning required**: Need to set heuristic threshold and sampling rate

**Expected Impact:**
- Implausibility rate: 38% → 10-15%
- Training time: 1x → 1.2-1.3x
- Quality improvement: High

---

### Option 6: Post-Training Filtering (No Reward Change)

```python
# During training: Use current rewards (no change)
RA_total = RA_res_harm + RA_res_refusal + RA_revision + RA_format

# After training: Filter dataset for next round
for attack in generated_attacks:
    plausibility = medgemma_judge(attack)
    if plausibility > 0.5:
        keep_for_assessor_training(attack)
    else:
        discard(attack)
```

**Pros:**
- ✅ **No training slowdown**: Filtering happens offline
- ✅ **No reward engineering**: Keep current system
- ✅ **Simple**: Just filter bad examples
- ✅ **Flexible**: Can adjust threshold between rounds

**Cons:**
- ❌ **Wasteful**: Attacker still generates 38% garbage
- ❌ **No learning signal**: Attacker doesn't learn to avoid implausibility
- ❌ **Reduced data**: Throw away 38% of generated attacks
- ❌ **Doesn't fix root cause**: Attacker keeps making same mistakes

**Expected Impact:**
- Implausibility rate: 38% generated, 0% used (filtered out)
- Training time: 1x training + filtering time
- Quality improvement: Medium (assessor trains on clean data, but attacker doesn't improve)

---

## Comparison Matrix

| Option | Implausibility Reduction | Training Speed | Implementation | Learning Signal | Cost |
|--------|-------------------------|----------------|----------------|-----------------|------|
| **1. Multiplicative** | ⭐⭐⭐⭐⭐ (38%→10%) | ⭐⭐ (2-3x slower) | ⭐⭐⭐ (Medium) | ⭐⭐⭐⭐⭐ (Strong) | 💰💰💰 |
| **2. Additive** | ⭐⭐⭐⭐ (38%→15%) | ⭐⭐ (2-3x slower) | ⭐⭐⭐⭐ (Easy) | ⭐⭐⭐⭐ (Good) | 💰💰💰 |
| **3. Binary Threshold** | ⭐⭐⭐⭐⭐ (38%→5%) | ⭐⭐ (2-3x slower) | ⭐⭐⭐⭐ (Easy) | ⭐⭐⭐⭐⭐ (Very Strong) | 💰💰💰 |
| **4. Heuristic Only** | ⭐⭐ (38%→25%) | ⭐⭐⭐⭐⭐ (No slowdown) | ⭐⭐⭐⭐⭐ (Easy) | ⭐⭐ (Weak) | 💰 |
| **5. Hybrid** | ⭐⭐⭐⭐⭐ (38%→10%) | ⭐⭐⭐⭐ (1.2-1.3x) | ⭐⭐ (Complex) | ⭐⭐⭐⭐ (Good) | 💰💰 |
| **6. Post-Filter** | ⭐⭐⭐ (0% used) | ⭐⭐⭐⭐⭐ (No slowdown) | ⭐⭐⭐⭐⭐ (Easy) | ⭐ (None) | 💰💰 |

## Recommendations

### Short-term (Immediate Fix)
**Option 4 (Heuristic) + Option 6 (Post-Filter)**
- Add lightweight heuristics to catch obvious garbage (38%→25%)
- Filter remaining implausible attacks before assessor training
- Fast to implement, no training slowdown
- Gets you from 38% to ~15% effective implausibility

### Medium-term (Best Balance)
**Option 5 (Hybrid)**
- Heuristic catches obvious issues
- LLM verifies suspicious cases (~20-30%)
- Only 1.2-1.3x training time
- Reduces implausibility to ~10%
- Attacker learns to avoid implausibility

### Long-term (Highest Quality)
**Option 1 (Multiplicative) or Option 3 (Binary)**
- Full LLM-based plausibility checking
- Strongest learning signal
- Reduces implausibility to 5-10%
- Accept 2-3x training time for quality
- Consider after you have more compute

## Key Questions to Consider

### 1. What's your compute budget?
- **Limited**: Option 4 (Heuristic) or Option 6 (Filter)
- **Moderate**: Option 5 (Hybrid)
- **Generous**: Option 1, 2, or 3 (Full LLM)

### 2. How fast do you need results?
- **ASAP**: Option 4 or 6 (no slowdown)
- **Can wait**: Option 5 (1.2-1.3x)
- **Quality over speed**: Option 1, 2, or 3 (2-3x)

### 3. How important is the learning signal?
- **Critical** (want attacker to learn): Option 1, 2, 3, or 5
- **Less important** (just need clean data): Option 6

### 4. What's your target implausibility rate?
- **<30%**: Option 4
- **<15%**: Option 5 or 6
- **<10%**: Option 1, 2, 3, or 5

### 5. How reliable is medgemma as a judge?
- **Very reliable**: Option 1, 2, or 3 (trust it fully)
- **Somewhat reliable**: Option 5 (use with heuristics)
- **Uncertain**: Option 4 (avoid dependency)

## My Recommendation

**Start with Option 5 (Hybrid)** because:

1. **Balanced trade-off**: 1.2-1.3x training time for 10-15% implausibility
2. **Catches everything**: Heuristic + LLM covers all cases
3. **Scalable**: Can adjust sampling rate based on quality
4. **Learning signal**: Attacker learns to avoid implausibility
5. **Practical**: Not too slow, not too weak

**Implementation path:**
1. Week 1: Implement Option 4 (heuristic) - quick win, 38%→25%
2. Week 2: Add Option 6 (filtering) - gets you to ~15% effective
3. Week 3: Upgrade to Option 5 (hybrid) - full solution, ~10%

This gives you incremental improvements while building toward the best solution.

## Alternative: Curriculum Approach

Instead of penalizing implausibility, **make the task easier**:

```python
# Start with vanilla_harmful (easier, 28% implausible)
# Only move to adversarial after quality improves

if current_implausibility_rate < 20%:
    enable_adversarial_games()
else:
    stick_to_vanilla_games()
```

**Pros:**
- ✅ No reward engineering needed
- ✅ Attacker learns on easier task first
- ✅ Natural progression

**Cons:**
- ❌ Slower overall progress
- ❌ Doesn't solve root cause

Could combine with any of the above options!
