# Dual-Game Zero-Sum Implementation

## 🎯 Overview

Implemented a dual-game approach inspired by the Self-RedTeam paper to prevent degenerate Nash equilibria and enforce diversity in self-play training.

## 🎮 Two Game Types

### Game 1: "inject" (50% of training data)
**Objective**: Attacker adds subtle errors, Assessor detects them

- **Attacker prompt**: "Add ONE subtle medical error to this note"
- **Attacker goal**: Add error that assessor misses
- **Assessor goal**: Detect the error
- **Reward structure**:
  - Attacker wins (+reward) if assessor says "Safe" (missed error)
  - Assessor wins (+reward) if correctly identifies "Concerning/Harmful"
  - Zero-sum: One wins, other loses

### Game 2: "keep_clean" (50% of training data)
**Objective**: Attacker keeps notes clean, Assessor correctly identifies safety

- **Attacker prompt**: "Keep this medical note safe and accurate. Do NOT add any errors"
- **Attacker goal**: Keep note clean (assessor should say "Safe")
- **Assessor goal**: Correctly identify it's safe
- **Reward structure**:
  - Attacker wins (+reward) if assessor correctly says "Safe"
  - Assessor wins (+reward) if correctly says "Safe"
  - Both can win in this game (cooperative on surface, but attacker could try to trick assessor)

## 🔄 Why This Prevents Degenerate Equilibria

### Previous Problem
- Attacker learned: "Output unchanged note → Assessor says Safe → I win"
- Assessor learned: "Say Safe for everything → Sometimes right → Good enough"
- Both stuck in lazy equilibrium

### New Solution
- **In "inject" game**: Attacker must ADD errors (can't just output unchanged)
- **In "keep_clean" game**: Attacker must KEEP clean (opposite pressure)
- **Opposing forces**: Attacker can't use same strategy for both games
- **Balanced data**: 50% Safe, 50% Concerning/Harmful expected

## 📊 Key Changes Made

### 1. Data Loading (`load_and_prepare_data`)
```python
# Split clean notes 50/50 between game types
ds_inject = clean_notes[:half] + game_type="inject"
ds_keep_clean = clean_notes[half:] + game_type="keep_clean"
ds_originals = concatenate([ds_inject, ds_keep_clean]).shuffle()
```

### 2. Attacker Prompts (`build_attacker_prompts`)
```python
if game_type == "inject":
    instruction = "Add ONE subtle medical error"
else:  # keep_clean
    instruction = "Keep this note safe. Do NOT add errors"
```

### 3. Reward Function (`attacker_reward_fn`)
```python
if game_type == "inject":
    # Attacker wins if assessor MISSES error
    r_harm = -R_HARM if correct else +R_HARM
else:  # keep_clean
    # Attacker wins if assessor CORRECTLY says Safe
    r_harm = +R_HARM if correct else -R_HARM
```

### 4. Diversity Tracking
```python
diversity_stats = {
    "inject_games": 0,
    "inject_safe": 0,
    "inject_concerning": 0,
    "inject_harmful": 0,
    "keep_clean_games": 0,
    "keep_clean_safe": 0,
    "keep_clean_concerning": 0,
    "keep_clean_harmful": 0,
}
```

## 📈 Expected Outcomes

### Balanced Training Data
- **Inject games**: Should produce mostly "Concerning/Harmful" outcomes
- **Keep_clean games**: Should produce mostly "Safe" outcomes
- **Overall**: ~50% Safe, ~50% Concerning/Harmful

### Improved Learning
- **Attacker**: Learns two distinct strategies (inject vs keep_clean)
- **Assessor**: Trains on balanced data, learns to distinguish both cases
- **No lazy equilibrium**: Can't win both games with same strategy

### Diversity Metrics to Watch
```
Round 1:
  Inject games: 32
    - Safe: 5 (attacker winning)
    - Concerning: 25 (assessor winning)
    - Harmful: 2 (assessor winning)
  Keep_clean games: 32
    - Safe: 28 (both winning)
    - Concerning: 4 (attacker failing)
    - Harmful: 0
```

## 🚀 Next Steps

1. **Run training** with new dual-game approach
2. **Monitor diversity stats** in logs
3. **Check for balance**:
   - Are both game types being played?
   - Is data distribution ~50/50?
   - Are both players learning?
4. **Adjust if needed**:
   - If one game dominates, check reward scaling
   - If still imbalanced, adjust game_type split ratio
   - If assessor still lazy, increase R_HARM multiplier

## 🔍 Debugging Tips

### Check game type distribution
```bash
grep "Game Type:" results/*_interactions.jsonl | sort | uniq -c
```

### Check diversity stats
```bash
grep "diversity_stats" results/*.jsonl | jq .stats
```

### Check assessor accuracy by game type
```python
# In display_selfplay.py, add game_type filtering
inject_accuracy = correct / total for game_type == "inject"
keep_clean_accuracy = correct / total for game_type == "keep_clean"
```

## 📚 References

- Self-RedTeam paper: https://arxiv.org/abs/2506.07468
- Key insight: "Dual prompt distributions with opposite objectives force diverse strategies"
- Our adaptation: Medical safety domain with inject vs keep_clean games
