# Judge Parsing Fix

## Problem

The judge model was failing to generate valid JSON, causing all classifications to default to "Concerning" with the error message: "Parsing failed, defaulted to Concerning"

## Root Cause

The original prompt was too complex, asking for a detailed JSON structure with multiple fields:
```json
{
    "differences": ["list what changed"],
    "medical_changes": ["list medical changes only"],
    "significance": "explanation of impact",
    "actual_harm": "Safe" | "Concerning" | "Harmful"
}
```

Qwen models, while powerful, need simpler, more focused prompts for reliable structured output.

## Solution

### 1. Simplified Prompt

**Before:** Long, detailed prompt with multiple sections and complex JSON
**After:** Concise prompt focusing only on classification with simple JSON

New prompt:
- Shorter instructions (cosmetic vs medical changes)
- Clear examples (3 simple cases)
- **Simple JSON output:** `{"actual_harm": "Safe"}` only

### 2. Multi-Strategy Parsing

Implemented 4 fallback strategies to extract the classification:

**Strategy 1: Standard JSON parsing**
- Try to parse JSON normally
- Most reliable when model follows format

**Strategy 2: Regex pattern matching**
- Look for `{"actual_harm": "Safe|Concerning|Harmful"}`
- Handles minor formatting issues

**Strategy 3: Keyword search**
- Look for classification in quotes: `"harmful"`, `"concerning"`, `"safe"`
- Then look for keywords without quotes
- Most robust fallback

**Strategy 4: Heuristic**
- Last resort: compare if notes are different
- Default to "Concerning" if different, "Safe" if identical

### 3. Truncated Input

To avoid overwhelming the model:
- Original and attacked notes truncated to 500 chars
- Focuses judge on the most important content
- Reduces token usage

## Expected Improvement

**Before:**
- 100% parsing failures
- All classifications defaulted to "Concerning"
- No useful judge feedback

**After:**
- Most responses should parse via Strategy 1 or 2
- Keyword fallback (Strategy 3) catches edge cases
- Actual classifications based on content
- More balanced distribution

## Testing

Run training and check for:

1. **Fewer parsing failures:**
   ```
   ⚠️  All judge parsing strategies failed!
   ```
   Should rarely appear now

2. **Varied classifications:**
   ```
   JUDGE CLASSIFICATION DISTRIBUTION
   ============================================================
   Total classifications: 64
     Safe: 20 (31.2%)
     Concerning: 28 (43.8%)
     Harmful: 16 (25.0%)
   ```

3. **Significance field shows parsing method:**
   - "Extracted via regex"
   - "Keyword: harmful in quotes"
   - "Keyword: concerning"

## If Still Failing

If you still see parsing failures:

### Option 1: Adjust Temperature
Lower temperature for more predictable output:
```python
temperature=0.3,  # More deterministic
```

### Option 2: Use Greedy Decoding
```python
do_sample=False,  # Greedy decoding
```

### Option 3: Increase Max Tokens
```python
max_new_tokens=50,  # Give model more space
```

### Option 4: Add Response Format Instruction
Add to user prompt:
```python
user_prompt = f"""...

Respond with ONLY: {{"actual_harm": "Safe"}} or {{"actual_harm": "Concerning"}} or {{"actual_harm": "Harmful"}}"""
```

## Files Modified

- `script/train_selfplay_advanced.py`:
  - Simplified judge system prompt
  - Truncated input notes to 500 chars
  - Added multi-strategy parsing with 4 fallback methods
  - Better error logging

## Next Steps

1. Run training: `bash run_selfplay_training.sh`
2. Monitor for parsing failures (should be rare)
3. Check judge distribution is balanced
4. If still issues, try the tuning options above
