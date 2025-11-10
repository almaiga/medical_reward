# Implementation Plan

- [x] 1. Create package structure and backup original file
  - Create `script/selfplay/` directory
  - Copy `train_selfplay_advanced.py` to `train_selfplay_advanced_backup.py` as exact backup
  - Create empty `script/selfplay/__init__.py` with package docstring
  - _Requirements: 1.5, 3.4_

- [x] 2. Create utils module with parsing and validation functions
  - Create `script/selfplay/utils.py` with module docstring
  - Extract and move `parse_response()` function
  - Extract and move `check_attack_faithfulness()` function
  - Extract and move `deduplicate_attacked_notes()` function
  - Extract and move `extract_original_from_attacker_prompt()` function
  - Extract and move `extract_attacked_from_assessor_prompt()` function
  - Extract and move `log_interaction()` function
  - Extract and move `patch_tokenizer_for_grpo()` function
  - Add imports (os, re, json, datetime, torch)
  - _Requirements: 1.3, 2.3, 4.1_

- [x] 3. Create judge module with evaluation logic
  - Create `script/selfplay/judge.py` with module docstring
  - Extract and move `get_judge_assessment()` function
  - Extract and move `evaluate_thinking_quality()` function
  - Extract and move `JudgeValidator` class with all methods
  - Add imports (re, json, torch)
  - Import `parse_response` from utils module
  - _Requirements: 1.1, 2.4, 4.1_

- [x] 4. Create data module with MEDEC loading functions
  - Create `script/selfplay/data.py` with module docstring
  - Extract and move `load_and_prepare_data()` function
  - Add imports (pandas, datasets)
  - _Requirements: 1.1, 2.5, 4.1_

- [x] 5. Create prompts module with prompt generation functions
  - Create `script/selfplay/prompts.py` with module docstring
  - Extract and move `build_attacker_prompts()` function
  - Extract and move `make_assessor_prompts()` function
  - Add imports (datasets)
  - _Requirements: 1.2, 2.1, 4.1_

- [x] 6. Create rewards module with reward calculation functions
  - Create `script/selfplay/rewards.py` with module docstring
  - Extract and move reward function creation logic from `main()`
  - Create `create_attacker_reward_fn()` that returns closure
  - Create `create_assessor_reward_fn()` that returns closure
  - Add imports (torch)
  - Import functions from utils, judge, and prompts modules
  - _Requirements: 1.3, 2.2, 4.1_

- [x] 7. Create main module with training orchestration
  - Create `script/selfplay/main.py` with module docstring
  - Extract and move `get_device()` function
  - Extract and move `load_causal_lm()` function
  - Extract and move `main()` function with argument parsing
  - Update `main()` to use imported functions from other modules
  - Add imports from all selfplay modules
  - Add standard imports (os, argparse, datetime, gc, torch, transformers, trl)
  - Add `if __name__ == "__main__"` block
  - _Requirements: 1.6, 2.6, 4.1_

- [x] 8. Update package __init__.py with exports
  - Add imports for key functions from each module
  - Export `main` function
  - Export utility functions
  - Export reward creation functions
  - Add package-level docstring
  - _Requirements: 1.1, 4.5_

- [x] 9. Update train_selfplay_advanced.py to use new modules
  - Replace entire content with imports from `script.selfplay`
  - Import and call `main()` from selfplay.main
  - Preserve command-line interface
  - Add comment explaining refactoring and pointing to backup
  - _Requirements: 3.1, 3.4, 4.1_

- [x] 10. Verify backwards compatibility and functionality
  - Run refactored script with minimal arguments (2 samples, 1 round)
  - Verify it produces output files in expected location
  - Check that command-line arguments work correctly
  - Compare structure of output with backup version expectations
  - _Requirements: 3.1, 3.2, 3.3, 3.5_
