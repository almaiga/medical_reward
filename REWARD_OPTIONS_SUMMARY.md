# Plausibility Reward Options: Quick Reference

## The Problem
- **38% of adversarial attacks are medically implausible**
- Current reward has no plausibility constraint
- Attacker learns to generate nonsense that fools assessor

## The Goal
- Reduce implausibility from 38% to <15%
- Maintain training efficiency
- Give attacker clear learning signal

---

## Option Comparison

### 🏆 RECOMMENDED: Hybrid Approach (Option 5)

**How it works:**
```
Fast heuristic checks all attacks (0.001 sec)
  ↓
If suspicious → LLM verification (2 sec)
If looks good → Random 10% LLM sampling
  ↓
Multiply game rewards by plausibility score
```

**Why it's best:**
- ✅ Only 1.2-1.3x training time (not 2-3x)
- ✅ Catches both obvious and subtle implausibility
- ✅ Strong learning signal for attacker
- ✅ Reduces implausibility to ~10%

**Trade-offs:**
- More complex to implement
- Requires tuning heuristic thresholds
- Some inconsistency (not all attacks get LLM)

---

### 🚀 FASTEST: Heuristic Only (Option 4)

**How it works:**
```
Check edit distance, medical terms, garbage patterns
  ↓
Compute plausibility score (0.0 to 1.0)
  ↓
Multiply game rewards by score
```

**Why consider it:**
- ✅ Zero training slowdown
- ✅ Catches obvious garbage (repeated tags, extreme edits)
- ✅ Easy to implement and tune
- ✅ Deterministic and fast

**Trade-offs:**
- Only reduces implausibility to ~25%
- Misses subtle medical implausibility
- Might need to combine with filtering

---

### 💪 STRONGEST: Binary Threshold (Option 3)

**How it works:**
```
LLM judges plausibility (2 sec)
  ↓
If plausibility < 0.5 → Worst score (-4.0)
If plausibility ≥ 0.5 → Normal rewards
```

**Why consider it:**
- ✅ Strongest deterrent (implausible = worst score)
- ✅ Reduces implausibility to ~5%
- ✅ Simple binary decision
- ✅ Clear learning signal

**Trade-offs:**
- 2-3x training slowdown
- Harsh (no partial credit)
- Sensitive to threshold choice

---

### 🎯 BALANCED: Multiplicative (Option 1)

**How it works:**
```
LLM judges plausibility (2 sec)
  ↓
Game rewards × plausibility score
Shaping rewards unchanged
```

**Why consider it:**
- ✅ Intuitive ("only get credit if plausible")
- ✅ Scales naturally (partial credit for partial plausibility)
- ✅ Reduces implausibility to ~10%
- ✅ Strong learning signal

**Trade-offs:**
- 2-3x training slowdown
- Breaks zero-sum property
- Might make gradients harder

---

### 🔧 SIMPLEST: Post-Filter (Option 6)

**How it works:**
```
Train normally (no reward changes)
  ↓
After training, filter out implausible attacks
  ↓
Only use plausible attacks for assessor training
```

**Why consider it:**
- ✅ No training slowdown
- ✅ No reward engineering
- ✅ Easy to implement
- ✅ Flexible (adjust threshold between rounds)

**Trade-offs:**
- Wasteful (attacker generates 38% garbage)
- No learning signal (attacker doesn't improve)
- Reduced training data (throw away 38%)

---

## Decision Tree

```
Do you have 2-3x compute budget?
├─ YES → Option 1 (Multiplicative) or Option 3 (Binary)
│         Best quality, strongest signal
│
└─ NO → Do you need results ASAP?
    ├─ YES → Option 4 (Heuristic) + Option 6 (Filter)
    │         Quick win, no slowdown
    │
    └─ NO → Option 5 (Hybrid) ⭐ RECOMMENDED
              Best balance of speed and quality
```

---

## Implementation Difficulty

**Easy** (1-2 hours):
- Option 4: Heuristic
- Option 6: Post-filter

**Medium** (3-5 hours):
- Option 2: Additive penalty
- Option 3: Binary threshold

**Complex** (1-2 days):
- Option 1: Multiplicative
- Option 5: Hybrid

---

## Expected Outcomes

| Metric | Current | Option 4 | Option 5 | Option 1/3 |
|--------|---------|----------|----------|------------|
| **Implausibility** | 38% | 25% | 10% | 5-10% |
| **Training Time** | 1x | 1x | 1.2-1.3x | 2-3x |
| **Assessor Accuracy** | 40% | 45-50% | 55-65% | 60-70% |
| **Implementation** | - | Easy | Complex | Medium |

---

## My Specific Recommendation

### Phase 1 (This Week): Quick Win
**Implement Option 4 (Heuristic)**
```python
# Add to rewards.py
def compute_plausibility_heuristic(original, attacked):
    score = 1.0
    
    # Penalize extreme edits
    edit_ratio = levenshtein_distance(original, attacked) / max(len(original), len(attacked))
    if edit_ratio > 0.5:
        score *= 0.3
    
    # Penalize garbage patterns
    if attacked.count("</think>") > 2:
        score = 0.0
    
    # Penalize extreme length changes
    length_ratio = len(attacked) / len(original)
    if length_ratio < 0.5 or length_ratio > 2.0:
        score *= 0.5
    
    return score

# In attacker_reward_fn:
heuristic_score = compute_plausibility_heuristic(original, attacked_note)
total_reward = (RA_res_harm + RA_res_refusal) * heuristic_score + RA_revision + RA_format
```

**Expected result:** 38% → 25% implausibility, no slowdown

### Phase 2 (Next Week): Add Filtering
**Implement Option 6 (Post-filter)**
- After each round, filter out implausible attacks
- Only use plausible attacks for assessor training

**Expected result:** 25% → 15% effective implausibility

### Phase 3 (Week 3): Full Solution
**Upgrade to Option 5 (Hybrid)**
- Keep heuristic from Phase 1
- Add LLM verification for suspicious cases
- Random sampling for quality monitoring

**Expected result:** 15% → 10% implausibility, 1.2-1.3x training time

---

## Key Insights from Your Data

From the evaluation results:
1. **38% implausible** - Need strong intervention
2. **Assessor 40% accurate** - Low accuracy partly due to implausibility
3. **All "medium" difficulty** - No variation, suggests quality issues
4. **Adversarial worse than vanilla** - Task is too hard for attacker

**This suggests:**
- Start with heuristic (catches obvious garbage)
- Add filtering (ensures clean assessor training)
- Consider curriculum (start with vanilla games)
- Eventually upgrade to hybrid (full solution)

---

## Questions to Discuss

1. **Compute budget**: Can you afford 1.2-1.3x training time? 2-3x?

2. **Timeline**: Need results this week? Or can wait for best solution?

3. **Medgemma reliability**: From your evaluation, how often did medgemma make mistakes?

4. **Priority**: Is it more important to:
   - Get quick improvement (→ Option 4)
   - Get best quality (→ Option 1/3)
   - Balance both (→ Option 5)

5. **Curriculum**: Should we also make the task easier (vanilla before adversarial)?
