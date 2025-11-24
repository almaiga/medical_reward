# Test Script Fix Summary

## Problem Identified

The test script `test_plausibility_reward.py` was failing with:
```
KeyError: 'original'
```

This occurred in `script/selfplay/prompts.py` at line 277:
```python
"original_note": rec["original"],
```

## Root Cause

The test script was calling `make_assessor_prompts()` with incomplete data:

**Before (WRONG):**
```python
def run_assessor(attacked_note, policy_model, policy_tok, device):
    assessor_data = [{
        "attacked": attacked_note,
        "game_category": "unknown"
    }]
    assessor_ds = make_assessor_prompts(assessor_data, policy_tok)
```

**Expected by `make_assessor_prompts()`:**
```python
def make_assessor_prompts(records: list, tokenizer):
    """
    Args:
        records: List of dicts with 'original', 'attacked', 'game_category' keys
    """
    for rec in records:
        "original_note": rec["original"],  # <-- Expects 'original' key
        "attacked_note": rec["attacked"],
```

## Solution Applied

Updated `run_assessor()` function signature and implementation:

**After (CORRECT):**
```python
def run_assessor(original_note, attacked_note, policy_model, policy_tok, device):
    assessor_data = [{
        "original": original_note,  # <-- Added original note
        "attacked": attacked_note,
        "game_category": "unknown"
    }]
    assessor_ds = make_assessor_prompts(assessor_data, policy_tok)
```

And updated the call site:
```python
# Before
assessor_label, assessor_thought, assessor_completion = run_assessor(
    attacked_note, policy_model, policy_tok, device
)

# After
assessor_label, assessor_thought, assessor_completion = run_assessor(
    original_note, attacked_note, policy_model, policy_tok, device
)
```

## Why This Matters

The assessor needs both the original and attacked notes to:
1. Compare them for differences
2. Provide context to the judge model
3. Enable proper plausibility assessment

The original note is already available from `run_attacker()`, so we just needed to pass it through.

## Files Modified

- `test_plausibility_reward.py`: Fixed `run_assessor()` function signature and call site

## Next Steps

The script should now run without the KeyError. The test will:
1. Load policy and judge models
2. Generate attacks using the attacker
3. Evaluate attacks using the assessor (now with both original and attacked notes)
4. Judge the results for harm and plausibility
5. Calculate rewards including the new plausibility component
