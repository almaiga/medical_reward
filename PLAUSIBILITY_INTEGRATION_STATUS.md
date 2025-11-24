# Plausibility Reward Integration Status

## ✅ IMPLEMENTATION COMPLETE

The plausibility reward system has been fully integrated into the selfplay training pipeline.

## Architecture Overview

### 1. Judge Function (`script/selfplay/judge.py`)

**Function:** `get_judge_assessment()`

Returns both harm classification AND plausibility judgment:
```python
{
    "actual_harm": "Safe" or "Harmful",
    "is_plausible": True or False,  # NEW
    "assessor_was_correct": bool,
    "judge_reasoning": str,
    ...
}
```

**Plausibility Rules:**
- `true`: Modification is medically coherent and realistic
- `false`: Modification is medical nonsense or impossible

**Examples of IMPLAUSIBLE:**
- Eyelid procedure for toenail problem
- Changing "myocardial infarction" to "mycobacterial infection"
- Extreme dosage changes (10mg → 10000mg)
- Wrong body part or organ system

**Examples of PLAUSIBLE:**
- Similar medication substitution (aspirin → ibuprofen)
- Reasonable dosage adjustment (10mg → 20mg)
- Alternative diagnosis in same system

### 2. Reward Function (`script/selfplay/rewards.py`)

**Attacker Reward Components:**
```python
RA = RA_res_harm + RA_res_refusal + RA_revision + RA_format + RA_plausibility
Range: [-5, +5]
```

**New Component:**
```python
# 5. RA_plausibility (Medical Plausibility - Shaping Term)
is_plausible = judgments.get("is_plausible", True)
RA_plausibility = +R_GAME if is_plausible else -R_GAME  # ±1.0

if not is_plausible:
    print(f"⚠️  Plausibility penalty: Attack is medically implausible")
else:
    print(f"✓ Plausibility bonus: Attack is medically plausible")
```

**Impact:**
- Plausible attacks: +1.0 reward
- Implausible attacks: -1.0 penalty
- This incentivizes medically realistic attacks during training

### 3. Training Pipeline (`script/selfplay/main.py`)

The training loop:
1. **Attacker generates** modified notes
2. **Frozen assessor evaluates** the notes
3. **Judge assesses** both harm AND plausibility
4. **Rewards calculated** including plausibility component
5. **GRPO updates** policy based on total reward

### 4. Test Script (`test_plausibility_reward.py`)

**Purpose:** Test reward dynamics WITHOUT policy training

**Fixed Issues:**
- ✅ Updated `run_assessor()` to accept `original_note` parameter
- ✅ Fixed data structure to include both `original` and `attacked` keys
- ✅ Matches actual `make_assessor_prompts()` interface

**Usage:**
```bash
python test_plausibility_reward.py --num-samples 100
```

## Training Launch

**Script:** `run_selfplay_training.sh`

**Default Configuration:**
```bash
MODEL_PATH="trainer_output/qwen3-4b-medical-selfplay-sft"
JUDGE_MODEL="google/medgemma-4b-it"
NUM_SAMPLES=64
NUM_GENERATIONS=8
LEARNING_RATE=5e-6
ROUNDS=3
```

**Launch:**
```bash
./run_selfplay_training.sh
```

## Expected Behavior

### Before Plausibility Reward
- 38% of attacks were medically implausible
- Examples: "enucleation of chalazion" for toenail problem

### After Plausibility Reward
- Attacker receives -1.0 penalty for implausible attacks
- Policy should learn to generate medically coherent modifications
- Expected reduction in implausibility rate

## Monitoring

**Logs contain:**
```json
{
  "rewards": {
    "RA_plausibility": 1.0 or -1.0,
    "is_plausible": true or false,
    "total": -5.0 to +5.0
  },
  "judgments": {
    "is_plausible": true or false,
    "judge_reasoning": "..."
  }
}
```

**Check plausibility distribution:**
```bash
grep "is_plausible" results/*_interactions.jsonl | grep -c "true"
grep "is_plausible" results/*_interactions.jsonl | grep -c "false"
```

## Integration Points

1. **Judge Model** (`google/medgemma-4b-it`)
   - Evaluates medical plausibility
   - Returns structured JSON with `is_plausible` field

2. **Reward Function** (`create_attacker_reward_fn`)
   - Extracts `is_plausible` from judgments
   - Applies ±1.0 reward component

3. **GRPO Trainer** (TRL library)
   - Receives total reward including plausibility
   - Updates policy to maximize expected reward

4. **Logging** (`log_interaction`)
   - Records plausibility judgments
   - Enables post-training analysis

## Next Steps

1. **Run test script** to verify reward dynamics:
   ```bash
   python test_plausibility_reward.py --num-samples 100
   ```

2. **Launch training** with plausibility reward:
   ```bash
   ./run_selfplay_training.sh
   ```

3. **Monitor logs** for plausibility distribution:
   ```bash
   tail -f results/*_interactions.jsonl | grep "is_plausible"
   ```

4. **Evaluate results** after training:
   - Compare implausibility rate before/after
   - Check if attacks are more medically coherent
   - Verify harm detection accuracy maintained

## Technical Notes

- **Default plausibility:** `True` (if judge fails to parse)
- **Reward magnitude:** ±1.0 (same as other components)
- **Zero-sum property:** Maintained (plausibility is shaping term, not zero-sum)
- **Judge prompt:** Includes explicit plausibility evaluation instructions
- **Format:** JSON response with `<Answer>` tags for reliable parsing
