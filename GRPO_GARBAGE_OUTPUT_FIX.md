# GRPO Garbage Output Fix

## Problem
The GRPO trainer was generating garbage output with repeated "user" tokens and random characters:
```
<think>useruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseru...
```

## Root Cause
**Double chat template application**: 
1. We were passing messages format (`[{"role": "system", "content": "..."}, ...]`) to GRPO
2. GRPO was applying `apply_chat_template()` again during generation
3. This caused the chat template to be applied twice, creating malformed prompts

Reference: https://github.com/unslothai/unsloth/issues/1672

## Solution
Apply the chat template **once** before passing to GRPO, then tell GRPO not to apply it again:

### 1. Add `apply_chat_template=False` to GRPOConfig
```python
common_cfg = dict(
    # ... other config ...
    apply_chat_template=False,  # CRITICAL: We already applied it
)
```

### 2. Pre-template prompts in dataset creation
**Before (WRONG):**
```python
def build_attacker_prompts(ds, few_shot_examples, tokenizer):
    def to_prompt(row):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        return {"prompt": messages}  # Messages format - GRPO will template again!
    return ds.map(to_prompt)
```

**After (CORRECT):**
```python
def build_attacker_prompts(ds, few_shot_examples, tokenizer):
    def to_prompt(row):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        # Apply template ONCE here
        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt_string}  # Pre-templated string
    return ds.map(to_prompt)
```

### 3. Remove redundant templating in generation code
**Before (WRONG):**
```python
attacker_ds = build_attacker_prompts(...)
messages = attacker_ds[0]["prompt"]
prompt_string = policy_tok.apply_chat_template(messages, ...)  # Templating again!
```

**After (CORRECT):**
```python
attacker_ds = build_attacker_prompts(...)
prompt_string = attacker_ds[0]["prompt"]  # Already templated
```

## Files Changed
- `script/train_selfplay_advanced.py`:
  - Modified `build_attacker_prompts()` to return pre-templated strings
  - Modified `make_assessor_prompts()` to return pre-templated strings
  - Added `apply_chat_template=False` to `common_cfg`
  - Removed redundant `apply_chat_template()` calls in generation loops

## Testing
Run the training script and check that:
1. No repeated "user" tokens in output
2. Model generates proper `<think>...</think><output>...</output>` format
3. No garbage characters or tokenizer artifacts
4. EOS tokens are properly generated and respected

## Key Takeaway
When using GRPO with chat models:
- **Either** let GRPO handle templating (pass messages format, don't set `apply_chat_template=False`)
- **Or** pre-template yourself (pass strings, set `apply_chat_template=False`)
- **Never** do both!
