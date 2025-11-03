# Self-RedTeam Reward Structure - Quick Reference

## Your Implementation (Simplified for Medical Domain)

### Attacker (Medical Note Modifier)
```python
r_faithfulness = ±1.0  # Respects game type (harmful vs safe)
r_game         = ±1.0  # Fools the assessor
r_format       = ±1.0  # Uses <think> tags correctly
# Total: -3.0 to +3.0
```

### Defender/Assessor (Medical Note Classifier)
```python
r_game   = ±1.0  # Classifies correctly
r_format = ±1.0  # Uses <think> tags correctly
# Total: -2.0 to +2.0
```

## Paper's Full Structure (from red_team/utils.py)

### Attacker
```python
prompt_harmfulness   = ±1.0  # Revision/faithfulness
response_refusal     = ±1.0  # Did defender refuse appropriately?
response_harmfulness = ±1.0  # Is response harmful?
cot_format          = ±1.0  # CoT formatting
# Total: -4.0 to +4.0
```

### Defender
```python
response_refusal     = ±1.0  # Refuse harmful, comply with benign
response_harmfulness = ±1.0  # Generate safe responses
cot_format          = ±1.0  # CoT formatting
# Total: -3.0 to +3.0
```

## Key Differences

1. **You combine** `response_refusal` + `response_harmfulness` → `r_game`
2. **Paper separates** them into two distinct rewards
3. **Both approaches are valid** - yours is simpler and appropriate for your domain

## Why Faithfulness Reward Matters

Without faithfulness reward:
- Attacker in "harmful" game might just output safe notes (easier)
- This creates imbalanced distribution (too many safe notes)
- Defender learns to always say "Safe" (reward hacking)

With faithfulness reward:
- Attacker MUST respect game type to get positive reward
- Distribution stays balanced (50% harmful, 50% safe)
- Defender must learn to classify all categories correctly

## Implementation Status

✅ **COMPLETE** - Your code correctly implements the paper's reward structure with appropriate simplifications for the medical domain.
