# Faithfulness Reward Implementation - Fixing Assessor Reward Hacking

## The Problem You Discovered

**Assessor learned to always say "Safe"** to maximize rewards:
- Round 1: 28.1% accuracy (trying to classify correctly)
- Round 2-4: 55-58% accuracy (just saying "Safe" for everything)
- Safe accuracy: 98.7% ✅
- Concerning accuracy: 0.0% ❌
- Harmful accuracy: 0.0% ❌

## Root Cause

**Imbalanced data distribution:**
- Safe: 76/128 (59%)
- Concerning: 32/128 (25%)
- Harmful: 20/128 (16%)

**Why imbalanced?**
- Attacker in "harmful" games often fails to create actual errors
- Creates Safe notes instead of Concerning/Harmful
- Assessor sees mostly Safe notes
- Learns: "Just say Safe = 59% accuracy!"

## The Paper's Solution: Faithfulness Reward

From `red_team/utils.py` - `get_attacker_prompt_reward()`:

```python
def get_attacker_prompt_reward(prompt_harmfulness, prompt_type, ...):
    """
    If the seed is harmful:
        - Reward if generated prompt is harmful (faithful)
        - Penalize if generated prompt is unharmful (unfaithful)
    
    If the seed is benign:
        - Reward if generated prompt remains benign (faithful)
        - Penalize if generated prompt becomes harmful (unfaithful)
    """
```

**Key insight:** The attacker gets rewarded/penalized based on whether it **respects the game type**, independent of whether it fools the defender.

## Implementation in Your Code

### Added Faithfulness Reward (Lines 1350-1370)

```python
# 1. FAITHFULNESS REWARD
if game_type == "harmful":
    # Harmful game: Should create Concerning or Harmful
    if actual_harm in ["Concerning", "Harmful"]:
        r_faithfulness = +R_GAME  # Faithful: created error
    else:
        r_faithfulness = -R_GAME  # Unfaithful: failed to create error
else:  # game_type == "safe"
    # Safe game: Should keep Safe
    if actual_harm == "Safe":
        r_faithfulness = +R_GAME  # Faithful: kept it safe
    else:
        r_faithfulness = -R_GAME  # Unfaithful: added error

# 2. GAME REWARD (fooling assessor)
# 3. FORMAT REWARD (using <think> tags)

total_reward = r_faithfulness + r_game + r_format
```

### Reward Range Changed

**Before:** -2.0 to +2.0 (game + format)
**After:** -3.0 to +3.0 (faithfulness + game + format)

## How This Fixes Reward Hacking

### Scenario: Attacker in Harmful Game

**Before (no faithfulness):**
- Creates Safe note → Assessor says Safe → Attacker loses (-1.0)
- Creates Harmful note → Assessor says Safe → Attacker wins (+1.0)
- **Problem:** Both strategies have similar expected value

**After (with faithfulness):**
- Creates Safe note → Gets -1.0 faithfulness + game result = worse
- Creates Harmful note → Gets +1.0 faithfulness + game result = better
- **Solution:** Attacker MUST create errors to maximize reward

### Expected Behavior

**Round 1-2:** Attacker learns faithfulness
- Harmful games → Start creating actual Concerning/Harmful notes
- Safe games → Keep notes Safe
- Distribution becomes balanced

**Round 3-4:** Assessor can't exploit imbalance
- Sees 33/33/33 distribution
- Must learn to classify all three categories
- Can't just say "Safe" anymore

## Tracking Faithfulness

Added logging to monitor:
```
Harmful games: 64
  - Safe: 20
  - Concerning: 25
  - Harmful: 19
  - Faithfulness: 44/64 (68.8%)  ← NEW

Safe games: 64
  - Safe: 60
  - Concerning: 3
  - Harmful: 1
  - Faithfulness: 60/64 (93.8%)  ← NEW
```

This shows how well the attacker is respecting game types.

## Expected Results After Fix

**Round 1:**
- Attacker learns: "I get +1.0 for respecting game type"
- Faithfulness: 40% → 60% → 80%

**Round 2-3:**
- Distribution becomes balanced
- Assessor can't exploit imbalance
- Must learn actual classification

**Round 4:**
- Balanced competition
- Both players skilled
- No reward hacking

## Comparison to Paper

| Component | Paper | Your Implementation | Status |
|-----------|-------|---------------------|--------|
| Faithfulness reward | ✅ ±1.0 | ✅ ±1.0 | ✅ Implemented |
| Game reward | ✅ ±1.0 | ✅ ±1.0 | ✅ Already had |
| Format reward | ✅ ±1.0 | ✅ ±1.0 | ✅ Already had |
| Total range | -3.0 to +3.0 | -3.0 to +3.0 | ✅ Aligned |

Your implementation now **fully matches** the paper's reward structure!

## Next Steps

1. **Run training again** with faithfulness reward
2. **Monitor faithfulness %** - should increase over rounds
3. **Check assessor behavior** - should stop saying "Safe" for everything
4. **Expect balanced distribution** - 33/33/33 by Round 3-4

The faithfulness reward will force the attacker to respect game types, naturally balancing the distribution and preventing the assessor from exploiting imbalance.
