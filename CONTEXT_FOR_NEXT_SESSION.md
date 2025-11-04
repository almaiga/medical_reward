# Context Summary for Next Session

## What We're Doing
Converting medical error detection self-play training from 3-level classification (Safe/Concerning/Harmful) to **binary classification (Safe/Harmful)** with improved prompts using MEDEC error taxonomy.

## Why
- 3-level classification was too hard - model collapsed to "always Safe"
- Binary is simpler, matches Self-RedTeam paper approach
- Current results: 0% accuracy on Concerning/Harmful classes

## Changes Completed (Steps 1-3)

### ✅ Step 1: Updated Judge (get_judge_assessment)
- Added `game_type` parameter to function signature
- Changed to binary classification (Safe/Harmful only)
- Added rich examples showing how game type affects judgment
- Updated parsing to only look for Safe/Harmful
- Judge now receives: original, attacked, assessor_label, **game_type**

### ✅ Step 2: Updated JudgeValidator
- Changed to track only Safe/Harmful (removed Concerning)
- `self.classifications = {"Safe": 0, "Harmful": 0}`
- Updated warning thresholds for binary

### ✅ Step 3: Updated Data Loading (load_and_prepare_data)
- **Harmful game:** seed_note = Corrected Text (clean), error_example = Text (error version)
- **Safe game:** seed_note = Corrected Text (clean), error_example = ""
- Shows clean→error transformation (attacker learns from real MEDEC errors)
- Few-shot examples updated to use error_example

## Changes Remaining (Steps 4-8)

### Step 4: Update build_attacker_prompts() - Line ~600
**Changes needed:**
- Update few_shot_text to use 'error_example' instead of 'target_note'
- Add 5 MEDEC error types explicitly in system prompt
- Harmful game: Show clean note + error example, ask to introduce similar error
- Update to use `row.get("error_example")` and `row.get("error_type")`
- Rich examples with transformations

### Step 5: Update make_assessor_prompts() - Line ~710
**Changes needed:**
- Binary classification only (Safe/Harmful) - remove all Concerning
- Add 5 MEDEC error types explicitly
- Update all 3 examples to binary (remove Concerning example)
- Add game_type to output dataset: `"game_type": rec.get("game_type", "unknown")`
- Rich examples with detailed reasoning

### Step 6: Update attacker_reward_fn() - Line ~1130
**Changes needed:**
- Pass `game_type` to `get_judge_assessment()` call
- Update faithfulness check: `if actual_harm == "Harmful":` (remove "Concerning")

### Step 7: Update assessor_reward_fn() - Line ~1050
**Changes needed:**
- Get `game_type` from kwargs: `game_types = kwargs.get("game_type", [])`
- Get game_type for each item: `game_type = game_types[i] if i < len(game_types) else "unknown"`
- Pass `game_type` to `get_judge_assessment()` call

### Step 8: Update diversity_stats and tracking - Line ~1010, ~1260
**Changes needed:**
- Remove "concerning" from diversity_stats dict
- Update faithfulness tracking to binary: `is_faithful = actual_harm == "Harmful"` for harmful game

## Key Design Decisions Made

1. **Binary classification** (Safe/Harmful) - simpler, more stable
2. **MEDEC error taxonomy** - 5 explicit error types as guidance
3. **Clean→error transformation** - attacker sees both versions
4. **Game type to judge** - judge knows what attacker was supposed to do
5. **Rich few-shot examples** - detailed reasoning, realistic scenarios

## File to Modify
`script/train_selfplay_advanced.py`

## Next Actions
1. Continue with Step 4: Update build_attacker_prompts()
2. Then Steps 5-8 in sequence
3. Test the updated code

## Important Notes
- The attacker uses `error_example` field (not `target_note`) from data loading
- All "Concerning" references must be removed
- Judge needs game_type passed through the pipeline
- Examples should be realistic with detailed medical reasoning
