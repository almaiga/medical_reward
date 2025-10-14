# Judge Accuracy Improvements

## Overview

This document describes the improvements made to the judge model to fix the classification distribution issues where 83% of cases were classified as "Concerning" and 0% as "Harmful".

## Problem Statement

**Before improvements:**
- 83% "Concerning" classifications (over-conservative)
- 0% "Harmful" classifications (missing severe cases)
- 17% "Safe" classifications
- Assessor accuracy: 23.1%

**Root causes:**
1. Judge only saw the attacked note, not the original
2. No explicit instruction to compare and identify changes
3. Judge defaulted to "Concerning" when uncertain
4. No validation to detect skewed distributions

## Solution Implemented

### 1. Improved Judge Prompt (Task 1)

**Location:** `get_judge_assessment()` in `script/train_selfplay_advanced.py`

**Changes:**
- Judge now explicitly compares original vs attacked notes
- Step-by-step reasoning process:
  1. Identify ALL differences
  2. Categorize as cosmetic vs medical
  3. Assess clinical significance
  4. Classify harm level
- Clear criteria with examples for each severity level
- Explicit instruction to ignore cosmetic changes (quotes, grammar, spacing)

**Example classifications:**
- Safe: Only cosmetic changes (quote style, grammar)
- Concerning: Minor medical changes (1.5x dosage adjustment)
- Harmful: Major changes (opposite diagnosis, >2x dosage error)

### 2. Enhanced Response Parsing (Task 2)

**Location:** `get_judge_assessment()` parsing section

**Changes:**
- Extract additional fields from judge response:
  - `differences`: List of all changes found
  - `medical_changes`: List of medical changes only
  - `significance`: Explanation of clinical impact
- Better error handling with informative messages
- Fallback logic when parsing fails

### 3. Judge Validation System (Task 3)

**Location:** `JudgeValidator` class

**Features:**
- Tracks classification distribution in real-time
- Warns when distribution is skewed:
  - Over-representation: >70% in one category
  - Under-representation: <5% in any category (when n>50)
- Provides statistics and percentages

**Target distribution:**
- Safe: 30-40%
- Concerning: 30-40%
- Harmful: 20-30%

### 4. Validation Reporting (Task 4)

**Location:** Main training loop after each round

**Features:**
- Prints judge classification distribution
- Shows warnings for skewed distributions
- Logs validation data to JSONL file
- Helps identify when judge needs recalibration

**Example output:**
```
JUDGE CLASSIFICATION DISTRIBUTION
============================================================
Total classifications: 64
  Safe: 20 (31.2%)
  Concerning: 28 (43.8%)
  Harmful: 16 (25.0%)

✅ Judge distribution looks balanced
============================================================
```

### 5. Enhanced Logging (Task 5)

**Location:** `log_interaction()` function

**Changes:**
- Added `judge_reasoning` section to interaction logs
- Captures:
  - Differences identified
  - Medical changes found
  - Significance explanation
  - Chain-of-thought reasoning
- Enables debugging and analysis of judge decisions

## Expected Improvements

### Classification Distribution
- **Before:** 83% Concerning, 0% Harmful, 17% Safe
- **Target:** 30-40% each for Safe/Concerning, 20-30% Harmful

### Assessor Training
- Better labels → Better training signal
- More "Harmful" examples → Learn to detect severe cases
- Balanced distribution → Learn full spectrum of severity

### Transparency
- Judge reasoning is logged
- Can review why classifications were made
- Easier to debug and improve

## Usage

The improvements are automatically active when running training:

```bash
bash run_selfplay_training.sh
```

Monitor the judge distribution output after each round. If you see warnings like:

```
⚠️  JUDGE DISTRIBUTION WARNINGS:
   - Concerning over-represented: 75.0%
   - Harmful under-represented: 2.0%
```

This indicates the judge may need further prompt tuning or temperature adjustment.

## Validation

To validate the improvements:

1. **Check distribution after first round:**
   - Should see more balanced percentages
   - Should see some "Harmful" classifications

2. **Monitor assessor accuracy:**
   - Should improve over rounds as labels get better
   - Target: >50% accuracy after training

3. **Review interaction logs:**
   - Check `*_interactions.jsonl` file
   - Look at `judge_reasoning` field
   - Verify judge is identifying changes correctly

## Tuning Parameters

If distribution is still skewed, you can adjust:

### Judge Temperature
**Location:** `get_judge_assessment()` - `temperature` parameter

- Current: 0.7
- Increase (0.8-0.9): More aggressive classifications, more "Harmful"
- Decrease (0.5-0.6): More conservative, more "Safe"

### Warning Thresholds
**Location:** `JudgeValidator.check_distribution()`

- Current: >70% over-representation, <5% under-representation
- Adjust based on your target distribution

### Classification Criteria
**Location:** Judge system prompt in `get_judge_assessment()`

- Adjust examples to calibrate severity
- Add more specific medical scenarios
- Refine dosage thresholds (currently 2x for Harmful)

## Files Modified

1. `script/train_selfplay_advanced.py`:
   - Updated `get_judge_assessment()` function
   - Added `JudgeValidator` class
   - Enhanced `log_interaction()` function
   - Added validation reporting in main loop

2. `.kiro/specs/improve-judge-accuracy/`:
   - `requirements.md`: Requirements specification
   - `design.md`: Design document
   - `tasks.md`: Implementation tasks

3. `JUDGE_IMPROVEMENTS.md`: This documentation file

## Next Steps

1. Run training and monitor judge distribution
2. If still skewed, adjust temperature or prompt
3. Review interaction logs to understand judge reasoning
4. Consider creating test cases for specific scenarios (optional task 6)
