# Implementation Plan

- [x] 1. Update judge prompt with diff-focused comparison
  - Modify `get_judge_assessment()` function in `script/train_selfplay_advanced.py`
  - Replace current system prompt with new diff-focused prompt that emphasizes comparing original vs attacked
  - Add explicit instructions to identify changes and categorize as cosmetic vs medical
  - Include clear examples for Safe/Concerning/Harmful classifications
  - _Requirements: 1.1, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 5.1, 5.2, 5.3_

- [x] 2. Enhance judge response parsing to extract reasoning
  - Update JSON parsing in `get_judge_assessment()` to extract additional fields
  - Parse `differences`, `medical_changes`, and `significance` from judge response
  - Store judge reasoning in `judgments` dictionary for logging
  - Add fallback logic for parsing failures with better error messages
  - _Requirements: 2.4, 5.4_

- [x] 3. Implement judge output validation
  - Create `JudgeValidator` class to track classification distribution
  - Add methods: `add_classification()`, `check_distribution()`, `get_stats()`
  - Implement warning logic for skewed distributions (>70% in one category)
  - Integrate validator into main training loop
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 4. Add validation reporting to training loop
  - Instantiate `JudgeValidator` in `main()` function
  - Call `add_classification()` after each judge assessment
  - Print distribution warnings every 10 steps
  - Log final statistics at end of training
  - _Requirements: 4.1, 4.2, 4.4_

- [x] 5. Enhance interaction logging with judge reasoning
  - Update `log_interaction()` to include judge reasoning fields
  - Add `judge_reasoning`, `identified_changes` to logged data
  - Ensure reasoning is captured from both attacker and assessor phases
  - _Requirements: 5.5_

- [ ]* 6. Create test script to validate judge improvements
  - Create `test_judge_accuracy.py` with known test cases
  - Test cosmetic-only changes → Should classify as "Safe"
  - Test major errors → Should classify as "Harmful"
  - Test minor changes → Should classify as "Concerning"
  - Verify distribution is more balanced than before
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 7. Update documentation with judge improvements
  - Document new judge prompt structure in code comments
  - Add examples of expected judge behavior
  - Document validation thresholds and warning conditions
  - _Requirements: 2.4, 4.2_
