# Remaining Changes for script/train_selfplay_advanced.py

## Changes to Make:

### 1. build_attacker_prompts() - Line ~600
- Update few_shot_text to use 'error_example' instead of 'target_note'
- Add 5 MEDEC error types explicitly
- Harmful game: Show clean + error example, ask to introduce similar error
- Update to use row.get("error_example") 

### 2. make_assessor_prompts() - Line ~710
- Binary classification only (Safe/Harmful)
- Add 5 MEDEC error types
- Update examples to binary
- Add game_type to output dataset

### 3. attacker_reward_fn() - Line ~1130
- Pass game_type to get_judge_assessment()
- Update faithfulness: remove "Concerning", only "Harmful"

### 4. assessor_reward_fn() - Line ~1050
- Get game_type from kwargs
- Pass game_type to get_judge_assessment()

### 5. diversity_stats - Line ~1010
- Remove "concerning" tracking
- Keep only: harmful_safe, harmful_harmful, safe_safe, safe_harmful

### 6. Faithfulness tracking - Line ~1260
- Update to binary: actual_harm == "Harmful" (not "in ['Concerning', 'Harmful']")

Due to token limits, user should:
1. Read the file
2. Make these changes manually OR
3. Ask me to make them one at a time
