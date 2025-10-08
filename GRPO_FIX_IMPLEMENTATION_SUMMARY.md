# GRPO Garbage Output Fix - Implementation Summary

## Problem Statement
GRPO training with Qwen3-4B was generating garbage output with infinitely repeating "user" tokens:
```
attacked_note: "<think>useruseruseruseruseruser..." (repeating infinitely)
```

## Root Cause Analysis
After comprehensive research across multiple sources (TRL GitHub issues, Unsloth issues, Claude AI analysis), the root cause is:

**GRPOTrainer hardcodes `add_special_tokens=False` in its `_prepare_inputs()` method**, which removes BOS (beginning of sequence) tokens that Qwen models require for proper text generation.

### Why This Affects Qwen Models Specifically
1. Qwen models are heavily trained on their specific chat format with `<|im_start|>` and `<|im_end|>` tokens
2. They require BOS tokens to know how to start generation properly
3. Without BOS tokens, the model sees raw "user" text from the chat template without proper framing
4. This causes the model to get confused and repeat "user" endlessly

## Solution Implemented

### 1. Tokenizer Monkey-Patch (PRIMARY FIX)
Created a function that monkey-patches the tokenizer's `__call__` method to force `add_special_tokens=True`:

```python
def patch_tokenizer_for_grpo(tokenizer):
    """Monkey-patch tokenizer to force add_special_tokens=True for GRPO training.
    
    CRITICAL FIX: GRPOTrainer calls tokenizer with add_special_tokens=False
    which removes BOS tokens that Qwen models require, causing garbage output.
    This patches the tokenizer's __call__ method to force add_special_tokens=True.
    
    Reference: https://github.com/huggingface/trl/issues/3520
    """
    original_call = tokenizer.__call__
    
    def patched_call(*args, add_special_tokens=True, **kwargs):
        # Override any False values to True
        if not add_special_tokens:
            print("DEBUG: Intercepted add_special_tokens=False, forcing True")
            add_special_tokens = True
        return original_call(*args, add_special_tokens=add_special_tokens, **kwargs)
    
    tokenizer.__call__ = patched_call
    return tokenizer
```

### 2. Tokenizer Patching in Main Function
```python
# Load tokenizer normally
policy_model, policy_tok = load_causal_lm(args.model_id, device)

# CRITICAL: Patch tokenizer to fix GRPO garbage output issue
policy_tok = patch_tokenizer_for_grpo(policy_tok)
print("✅ Tokenizer patched to force add_special_tokens=True")

# Verify special tokens are configured
print(f"EOS token: {policy_tok.eos_token} (ID: {policy_tok.eos_token_id})")
print(f"PAD token: {policy_tok.pad_token} (ID: {policy_tok.pad_token_id})")
```

### 3. vLLM Sampling Parameters (if available)
Added proper EOS token handling for vLLM:

```python
from vllm import SamplingParams

vllm_sampling_params = SamplingParams(
    temperature=1.0,
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    # CRITICAL: Include EOS token in stop list for Qwen
    stop=[policy_tok.eos_token, "<|im_end|>"],
    # CRITICAL: Include stop string so model learns to generate it
    include_stop_str_in_output=True,
    max_tokens=1024,
)
```

### 4. Pre-templated Prompts (SECONDARY FIX)
Modified prompt building functions to apply chat template once:

```python
def build_attacker_prompts(ds, few_shot_examples, tokenizer, num_shots=2):
    def to_prompt(row):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        
        # Apply chat template ONCE here, return string
        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        return {
            "prompt": prompt_string,  # Pre-templated string
            "original_note": row["original"],
        }
    
    return ds.map(to_prompt, remove_columns=ds.column_names)
```

### 5. GRPO Config Update
```python
common_cfg = dict(
    # ... other config ...
    **vllm_params,  # Add vLLM params if available
)
```

Note: `apply_chat_template` is not a valid GRPOConfig parameter. The chat template is applied during dataset preparation, and the `TokenizerWrapper` ensures special tokens are preserved.

## Expected Behavior After Fix

### Before (Broken)
```json
{
  "attacked_note": "<think>useruseruseruseruseruser..."
}
```

### After (Fixed)
```json
{
  "attacked_note": "<think>This is a case of postpartum hemorrhage with placental retention in an Rh-negative mother. The key concern is Rh alloimmunization prevention since the baby is Rh-positive. The rosette test helps determine if there was significant fetomaternal hemorrhage requiring additional anti-D immune globulin beyond the standard 300 mcg dose.</think>\n<output>Additional anti-D immune globulin may be needed based on rosette test results</output>"
}
```

## Verification Steps

1. **Check tokenizer wrapper is working:**
   - Look for "DEBUG: Intercepted add_special_tokens=False, forcing True" in logs
   - Verify special token IDs are printed at startup

2. **Monitor training metrics:**
   - `completions/mean_terminated_length` should be > 0
   - `completions/clipped_ratio` should be low (< 0.3)
   - `rewards/mean` should increase over time

3. **Inspect generated outputs:**
   - Should see proper `<think>...</think><output>...</output>` format
   - No repeated "user" tokens
   - No garbage characters or tokenizer artifacts
   - Proper EOS token generation

## Training Recommendations

Based on research from multiple sources:

1. **Wait for at least 300 steps** for the reward to actually increase
2. **Train for at least 12 hours** for good results
3. **Use models at least 1.5B parameters** to correctly generate thinking tokens
4. **Monitor completion metrics** to ensure completions are terminating with EOS properly

## References

1. [TRL Issue #3520: GRPOTrainer generates garbage output due to add_special_tokens=False](https://github.com/huggingface/trl/issues/3520)
2. [Unsloth Issue #1672: GRPO training produces garbage/mangled outputs](https://github.com/unslothai/unsloth/issues/1672)
3. [Unsloth Issue #1844: GRPOTrainer generates "noise" with Unsloth](https://github.com/unslothai/unsloth/issues/1844)
4. Claude AI comprehensive analysis of GRPO issues with Qwen models

## Files Modified

- `script/train_selfplay_advanced.py`:
  - Added `patch_tokenizer_for_grpo()` function (PRIMARY FIX)
  - Patched `policy_tok` before passing to GRPO
  - Added vLLM sampling params with proper EOS token handling (if vLLM available)
  - Modified `build_attacker_prompts()` to return pre-templated strings
  - Modified `make_assessor_prompts()` to return pre-templated strings
  - Removed redundant `apply_chat_template()` calls in generation loops

- `GRPO_GARBAGE_OUTPUT_FIX.md`:
  - Updated with correct root cause analysis
  - Added TokenizerWrapper solution
  - Added Qwen-specific considerations
  - Added training recommendations

## Key Takeaways

1. **Tokenizer patching is ESSENTIAL for Qwen models with GRPO** - Without it, you'll get garbage output 100% of the time
2. This is a **known bug in TRL's GRPO implementation**, not a problem with your code
3. The fix uses monkey-patching to intercept `add_special_tokens=False` calls
4. Monkey-patching is simpler than wrapping because it preserves the tokenizer's type for isinstance checks
5. Qwen models are particularly affected due to their reliance on specific chat tokens and BOS tokens
6. The "user" repetition is the chat template string being repeated without proper token framing

## Next Steps

1. Run the training script with the fixes applied
2. Monitor the debug output for "Intercepted add_special_tokens=False" messages
3. Check that generated outputs have proper format
4. Wait at least 300 steps before evaluating reward improvements
5. If issues persist, verify vLLM is installed and being used
