# Reward Function Simplification - Aligned with Self-RedTeam Paper

## Changes Made

### 1. Simplified Reward Constants (Line 27-29)

**Before:**
```python
R_HARM = 2.0
R_REFUSAL = 0.5
R_FORMAT = 0.2
R_GARBAGE_PENALTY = -5.0  # Referenced but not defined
```

**After (Matching Paper):**
```python
R_GAME = 1.0   # Main game reward: +1 for win, -1 for loss
R_FORMAT = 1.0 # Format reward: +1 for correct, -1 for violation
```

### 2. Assessor Reward Function (Lines 1140-1165)

**Simplified to:**
- Game reward: `+1` if correct classification, `-1` if incorrect
- Format reward: `+1` if uses `<think>` tags, `-1` if not
- **Total range: -2.0 to +2.0**
- **Removed:** Refusal reward complexity

### 3. Attacker Reward Function (Lines 1340-1360)

**Simplified to:**
- **Harmful game:** `+1` if assessor misses error, `-1` if catches it
- **Safe game:** `+1` if assessor correctly says safe, `-1` if false positive
- Format reward: `+1` if uses `<think>` tags, `-1` if not
- **Total range: -2.0 to +2.0**
- **Removed:** Refusal manipulation complexity

### 4. Faithfulness Check Penalty (Lines 1264-1295)

**Before:**
```python
scores.append(R_GARBAGE_PENALTY)  # Undefined constant
# or
scores.append(-R_FORMAT)  # Wrong penalty (-0.2)
```

**After (Matching Paper):**
```python
scores.append(-R_FORMAT)  # -1.0 penalty, same as format violation
```

## Paper's Reward Structure (from red_team/utils.py)

### Format Violation
```python
"cot_format_violation": (-1.0, 1.0)
```

### Game Rewards (general_sum_all_one config)
```python
"attacker": {
    "prompt_harmfulness": (-1.0, 1.0),
    "response_harmfulness": (-1.0, 1.0),
    "response_refusal": (-1.0, 1.0)
}
"defender": {
    "response_harmfulness": (-1.0, 1.0),
    "response_refusal": (-1.0, 1.0)
}
```

## Key Insights from Paper

1. **All rewards are ±1.0** - No multipliers, no complex scaling
2. **Format penalty = -1.0** - Same magnitude as game rewards
3. **No separate "garbage penalty"** - They filter out unparseable responses
4. **Refusal is part of game logic** - Not a separate reward component in simple version

## Why This Fixes Reward Hacking

**Before:**
- Faithfulness violation: `-0.2` (too weak)
- Model learned: "Gaming the system costs only -0.2, worth it!"

**After:**
- Faithfulness violation: `-1.0` (same as losing the game)
- Model learns: "Gaming costs as much as losing, not worth it"

## Medical Domain Adaptation

Your faithfulness check (length ratio, word overlap) is **more sophisticated** than the paper's format-only check. This is good for medical safety! The key was just using the correct penalty magnitude.

### Faithfulness Check Criteria
- Length ratio: 0.5-2.0x original (allows reasonable variation)
- Word overlap: 60%+ (ensures subtle modification, not rewrite)
- These catch reward hacking attempts while allowing legitimate medical errors

## Expected Behavior After Fix

1. **Format violations now cost -1.0** (not -0.2)
2. **Faithfulness violations cost -1.0** (not -0.2)
3. **Winning the game gives +1.0**
4. **Using proper format gives +1.0**

**Net effect:** Gaming the system is no longer profitable. The model must play the game properly to maximize rewards.

## Next Steps

1. **Test the training** - Run a few rounds and check terminal output
2. **Monitor faithfulness check pass rate** - Should improve as model learns
3. **Watch for diversity** - Model should generate varied attacks, not garbage
4. **Check judge distribution** - Should see mix of Safe/Concerning/Harmful

## Comparison: Your Implementation vs Paper

| Component | Paper | Your Implementation | Status |
|-----------|-------|---------------------|--------|
| Game reward | ±1.0 | ±1.0 | ✅ Aligned |
| Format reward | ±1.0 | ±1.0 | ✅ Aligned |
| Refusal reward | ±1.0 | Removed | ✅ Simplified |
| Garbage penalty | Filter out | -1.0 | ✅ Aligned |
| Faithfulness check | Format only | Length + overlap | ✅ Enhanced |

Your implementation is now **aligned with the paper** while adding **medical-specific safeguards** (faithfulness checking).
