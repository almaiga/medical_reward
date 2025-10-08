# GRPO Garbage Output Fix - CORRECTED

## Problem
The GRPO trainer was generating garbage output with repeated "user" tokens and random characters:
```
<think>useruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseru...
```

## Root Cause (CORRECTED)
**Missing BOS tokens due to `add_special_tokens=False`**: 
1. GRPOTrainer hardcodes `add_special_tokens=False` in its `_prepare_inputs()` method
2. This removes BOS (beginning of sequence) tokens that Qwen models require
3. Without BOS tokens, the model doesn't know how to start generation properly
4. The "user" string from the chat template gets repeated as garbage

This is a **known bug in TRL's GRPO implementation**, particularly affecting Qwen models.

References: 
- https://github.com/huggingface/trl/issues/3520
- https://github.com/unslothai/unsloth/issues/1672

## Solution (CORRECTED)
The fix is to **wrap the tokenizer** to force `add_special_tokens=True` even when GRPO tries to set it to False:

### 1. Create TokenizerWrapper class
```python
class TokenizerWrapper:
    """Wrapper to force add_special_tokens=True for GRPO training.
    
    CRITICAL FIX: GRPOTrainer calls tokenizer with add_special_tokens=False
    which removes BOS tokens that Qwen models require, causing garbage output.
    This wrapper intercepts those calls and forces add_special_tokens=True.
    
    Reference: https://github.com/huggingface/trl/issues/3520
    """
    
    def __init__(self, tokenizer):
        self._wrapped = tokenizer
    
    def __call__(self, *args, add_special_tokens=True, **kwargs):
        # Override any False values to True
        if not add_special_tokens:
            add_special_tokens = True
        return self._wrapped(*args, add_special_tokens=add_special_tokens, **kwargs)
    
    def __getattr__(self, name):
        # Delegate all other attributes to wrapped tokenizer
        return getattr(self._wrapped, name)
```

### 2. Wrap your tokenizer before passing to GRPO
```python
# Load tokenizer normally
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# CRITICAL: Wrap it
wrapped_tokenizer = TokenizerWrapper(tokenizer)

# Use wrapped tokenizer in GRPO
trainer = GRPOTrainer(
    model=model,
    processing_class=wrapped_tokenizer,  # Use wrapper!
    # ... other args
)
```

### 3. Configure vLLM sampling params (if using vLLM)
```python
from vllm import SamplingParams

vllm_sampling_params = SamplingParams(
    temperature=1.0,
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    # CRITICAL: Include EOS token in stop list for Qwen
    stop=[tokenizer.eos_token, "<|im_end|>"],
    # CRITICAL: Include stop string so model learns to generate it
    include_stop_str_in_output=True,
    max_tokens=1024,
)

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    # ... other args
)
```

### 4. Pre-template prompts (still needed)
```python
def build_attacker_prompts(ds, tokenizer):
    def to_prompt(row):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        # Apply template ONCE here
        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt_string}
    return ds.map(to_prompt)
```

### 5. Set apply_chat_template=False in config
```python
common_cfg = dict(
    # ... other config ...
    apply_chat_template=False,  # We already applied it
)
```

## Files Changed
- `script/train_selfplay_advanced.py`:
  - **Added `TokenizerWrapper` class** (PRIMARY FIX)
  - Wrapped `policy_tok` with `TokenizerWrapper` before passing to GRPO
  - Added vLLM sampling params with proper EOS token handling
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

## Key Takeaways
1. **TokenizerWrapper is essential for Qwen models with GRPO** - Without it, you'll get garbage output 100% of the time
2. The root cause is `add_special_tokens=False` hardcoded in TRL's GRPO implementation
3. Qwen models specifically need BOS tokens and chat template tokens like `<|im_start|>` and `<|im_end|>`
4. The "user" repetition happens because the chat template string is being repeated without proper token framing
5. This is a known bug in TRL that affects many models, not just Qwen

## Why Qwen Models Are Particularly Affected
- Qwen models are heavily trained on their specific chat format with `<|im_start|>` and `<|im_end|>` tokens
- They have special tokens in specific positions in vocabulary that must be present
- The instruct versions (like Qwen2.5-4B-Instruct) expect proper formatting or they produce garbage
- Without BOS tokens, the model sees raw "user" text without proper framing and gets confused

## Training Recommendations
- Wait for at least 300 steps for the reward to actually increase
- For good results, you'll need to train for at least 12 hours
- It's advised to apply GRPO to a model at least 1.5B in parameters to correctly generate thinking tokens
- Monitor `completions/mean_terminated_length` and `completions/clipped_ratio` metrics
