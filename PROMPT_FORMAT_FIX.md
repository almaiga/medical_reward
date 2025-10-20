# GRPO Prompt Format Fix

## Problem Identified

**Mismatch between adaptation training and GRPO prompts:**

- ✅ Adaptation training used: `Safe<think>reasoning</think>` (POST-FILL)
- ❌ GRPO prompts asked for: `<think>reasoning</think><output>Safe</output>` (PRE-FILL)
- ❌ Model confused, outputs incomplete responses

## Changes Made

### 1. Updated Assessor Prompts (POST-FILL)

**Before:**
```python
system_content = """You classify medical notes for safety errors.

CRITICAL: You MUST use this exact format:
<think>brief analysis</think>
<output>Safe</output> OR <output>Concerning</output> OR <output>Harmful</output>

Example:
<think>The dosage seems incorrect</think>
<output>Concerning</output>"""
```

**After:**
```python
system_content = """You classify medical notes for safety errors.

CRITICAL: You MUST use this exact format:
Safe<think>brief analysis</think>
OR
Concerning<think>brief analysis</think>
OR
Harmful<think>brief analysis</think>

Classifications:
- Safe: No medical errors
- Concerning: Minor medical error
- Harmful: Major medical error"""
```

### 2. Updated Attacker Prompts (POST-FILL)

**Before:**
```python
Format:
<think>what I'm changing</think>
<output>note with error</output>
```

**After:**
```python
# No format instructions in prompt
# Model learned from adaptation: [note]<think>reasoning</think>
```

### 3. Parser Already Updated

The `parse_response()` function now handles both formats:
1. Tries pre-fill format first
2. Tries post-fill format second
3. Falls back to heuristics

## Expected Results

After restarting GRPO with these changes:

✅ **Assessor will output**: `Safe<think>No errors detected</think>`
✅ **Attacker will output**: `[modified note]<think>Changed X to Y</think>`
✅ **Parser will extract**: 
   - thought: "No errors detected"
   - output: "Safe"
✅ **Rewards will be calculated correctly**

## Next Steps

1. **Restart GRPO training** with the updated prompts
2. **Monitor first few iterations** to verify format is correct
3. **Check terminal output** for:
   - "Extracted using POST-FILL format"
   - Complete assessor classifications
   - Higher faithfulness pass rates

## Why This Matters

**Post-fill format is easier for RL** because:
- Model generates answer immediately
- Reasoning comes after (can reflect on output)
- Shorter sequences during generation
- More natural for reward-based learning

This aligns with the Self-RedTeam paper's approach!
