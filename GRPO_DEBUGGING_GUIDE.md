# GRPO Garbage Output - Debugging Guide

## Current Status
The monkey-patching approach has been implemented but the garbage output persists. This suggests the issue is more complex than just `add_special_tokens=False`.

## What to Look For in Training Output

### 1. Check if Patches Are Being Triggered
Look for these debug messages:
```
DEBUG: Intercepted __call__ with add_special_tokens=False, forcing True
DEBUG: Intercepted encode with add_special_tokens=False, forcing True
DEBUG: Intercepted encode_plus with add_special_tokens=False, forcing True
DEBUG: Intercepted batch_encode_plus with add_special_tokens=False, forcing True
```

**If you DON'T see these messages:**
- GRPO is using a different tokenization method we haven't patched
- OR the issue isn't with tokenization at all

### 2. Check Tokenizer Configuration
Look for:
```
TOKENIZER CONFIGURATION
============================================================
EOS token: <|im_end|> (ID: 151645)
PAD token: <|endoftext|> (ID: 151643)
BOS token: <|im_start|> (ID: 151644)  # This should be present!
```

**If BOS token is missing:**
- Qwen models need BOS tokens
- The tokenizer might not be configured correctly

### 3. Check Sample Prompt
Look for:
```
SAMPLE ATTACKER PROMPT (first 500 chars)
============================================================
<|im_start|>system
You are a medical editor...
<|im_end|>
<|im_start|>user
Add ONE subtle medical error...
<|im_end|>
<|im_start|>assistant
```

**If you DON'T see `<|im_start|>` and `<|im_end|>` tokens:**
- The chat template isn't being applied correctly
- The prompts are malformed from the start

### 4. Check Tokenizer Patch Test
Look for:
```
TESTING TOKENIZER PATCH
============================================================
Test 1: Calling with add_special_tokens=False
DEBUG: Intercepted __call__ with add_special_tokens=False, forcing True
Result IDs: [151644, 9906, 1879, ...]
```

**If the DEBUG message appears:**
- The patch is working for direct calls
- But GRPO might be bypassing it

## Possible Root Causes

### Cause 1: GRPO Uses Internal Tokenization
GRPO might have its own tokenization logic that doesn't use the tokenizer's standard methods.

**Solution:** Check TRL source code for how GRPO tokenizes prompts internally.

### Cause 2: Prompts Are Already Malformed
The chat template application might not be including special tokens in the string itself.

**Solution:** Verify the prompt strings contain `<|im_start|>` and `<|im_end|>` tokens.

### Cause 3: Model Generation Issue
The model itself might be generating garbage due to:
- Incorrect generation parameters
- Model not properly trained on chat format
- EOS token not being generated

**Solution:** Test generation outside of GRPO to verify model works correctly.

### Cause 4: GRPO's Internal Generation
GRPO generates completions internally during training. If it's not using proper generation parameters, it will produce garbage.

**Solution:** Check if vLLM sampling params are being used correctly.

## Next Steps Based on Output

### If NO "DEBUG: Intercepted" messages appear:
1. GRPO is bypassing our patches
2. Need to find where GRPO actually tokenizes
3. Might need to patch TRL source code directly

### If "DEBUG: Intercepted" messages DO appear:
1. The tokenization patch is working
2. Issue is elsewhere (generation, prompts, model)
3. Focus on generation parameters and prompt format

### If prompts are missing `<|im_start|>` tokens:
1. Chat template isn't working correctly
2. Need to debug `apply_chat_template` call
3. Might need to manually construct prompts

### If everything looks correct but still garbage:
1. The issue might be in GRPO's generation logic
2. Try using vLLM backend if available
3. Consider using a different RL trainer (PPO, DPO)

## Alternative Approaches

### Approach 1: Use vLLM Backend
```python
# Install vLLM
pip install vllm

# GRPO will use vLLM for generation if available
# vLLM has better EOS handling
```

### Approach 2: Modify TRL Source Code
```python
# Find TRL installation
import trl
print(trl.__file__)

# Edit grpo_trainer.py directly
# Change add_special_tokens=False to add_special_tokens=True
```

### Approach 3: Use Different Trainer
```python
# Try PPOTrainer or DPOTrainer instead
from trl import PPOTrainer, PPOConfig

# PPO might not have the same tokenization issues
```

### Approach 4: Pre-tokenize Everything
```python
# Instead of passing strings, pass pre-tokenized input_ids
# This bypasses GRPO's tokenization entirely
def build_attacker_prompts_tokenized(ds, tokenizer):
    def to_prompt(row):
        messages = [...]
        # Tokenize with special tokens
        tokens = tokenizer.apply_chat_template(
            messages, 
            tokenize=True,  # Get tokens, not string
            add_generation_prompt=True,
            return_tensors="pt"
        )
        return {
            "input_ids": tokens[0].tolist(),
            "original_note": row["original"],
        }
    return ds.map(to_prompt)
```

## Diagnostic Commands

### Test Tokenizer Outside GRPO
```python
python script/test_tokenizer_patch.py
```

### Test Model Generation
```python
python script/test_logic_tool.py --model_id your_model --num_examples 1
```

### Check TRL Version
```python
pip show trl
# Make sure you have latest version
pip install --upgrade trl
```

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Expected vs Actual

### Expected Output
```json
{
  "attacked_note": "<think>I'll change the diagnosis</think><output>Patient has pneumonia instead of cystic fibrosis</output>"
}
```

### Actual Output (Garbage)
```json
{
  "attacked_note": "<think>useruseruseruseruseruser..."
}
```

The "user" repetition suggests the chat template's "user" role string is being repeated without proper token framing.

## Contact Points

If all else fails:
1. Open issue on TRL GitHub: https://github.com/huggingface/trl/issues
2. Check Unsloth discussions: https://github.com/unslothai/unsloth/discussions
3. Ask on Hugging Face forums: https://discuss.huggingface.co/

Include:
- TRL version
- Transformers version
- Model name (Qwen3-4B)
- Full error output
- Sample prompt that produces garbage
