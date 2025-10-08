# Troubleshooting Output Guide

## What to Look For in Training Output

With the new debug logging, you'll see detailed information at each stage. Here's how to interpret it:

## 1. Startup Phase

### Tokenizer Configuration
```
TOKENIZER CONFIGURATION
============================================================
EOS token: <|im_end|> (ID: 151645)
PAD token: <|endoftext|> (ID: 151643)
BOS token: <|im_start|> (ID: 151644)
```
✅ **Good**: All tokens present with IDs
❌ **Bad**: Missing BOS token

### Tokenizer Patch Test
```
TESTING TOKENIZER PATCH
============================================================
Test 1: Calling with add_special_tokens=False
DEBUG: Intercepted __call__ with add_special_tokens=False, forcing True
Result IDs: [151644, 9906, 1879, ...]
```
✅ **Good**: See "DEBUG: Intercepted" message
❌ **Bad**: No intercept message

### Sample Prompt
```
SAMPLE ATTACKER PROMPT (first 500 chars)
============================================================
<|im_start|>system
You are a medical editor...
<|im_end|>
<|im_start|>user
Add ONE subtle medical error...
```
✅ **Good**: Contains `<|im_start|>` and `<|im_end|>`
❌ **Bad**: Missing these tokens

### Generation Config
```
GRPO GENERATION CONFIG
============================================================
  max_new_tokens: 1024
  temperature: 0.7
  repetition_penalty: 1.15
  ...
```
✅ **Good**: repetition_penalty is 1.15 or higher
❌ **Bad**: Missing or too low (<1.1)

## 2. Attacker Training Phase

### What GRPO Generates
```
ATTACKER REWARD FUNCTION - Processing 4 items
============================================================

--- Attacker Item 1/4 ---
Prompt preview (first 200 chars): <|im_start|>system...
Completion preview (first 200 chars): <think>I'll change the CSF result</think><output>A 52-year-old woman...
Parsed - Thought: I'll change the CSF result...
Parsed - Attacked note: A 52-year-old woman with HIV infection...
```

✅ **Good Signs**:
- Completion starts with `<think>`
- Has proper medical text
- Parsed attacked note is clean medical text
- No "useruseruser" repetition

❌ **Bad Signs**:
- Completion is `<think>useruseruseruser...`
- Parsed attacked note is garbage
- Repetitive tokens

## 3. Assessor Dataset Creation

```
ASSESSOR DATASET CREATED - 16 items
============================================================
Sample item 0:
  - Prompt (first 300 chars): <|im_start|>system...
  - Original note (first 200 chars): A 14-year-old boy...
  - Attacked note (first 200 chars): A 14-year-old boy... positive for Cryptococcus...
  ✅ Attacked note looks clean
```

✅ **Good**: "Attacked note looks clean"
❌ **Bad**: "⚠️ WARNING: Attacked note contains garbage!"

**This is the KEY checkpoint!** If attacked notes are garbage here, the problem occurred during attacker training.

## 4. Assessor Training Phase

```
ASSESSOR REWARD FUNCTION - Processing 4 items
============================================================

--- Assessor Item 1/4 ---
Prompt preview (first 200 chars): <|im_start|>system...
Completion preview (first 200 chars): <think>This note shows...</think><output>Concerning</output>
Attacked note preview (first 200 chars): A 14-year-old boy...
Parsed - Thought: This note shows...
Parsed - Label: Concerning
```

✅ **Good**: Clean attacked note, proper completion
❌ **Bad**: Garbage in attacked note or completion

## Diagnostic Decision Tree

### If garbage appears in "Attacked note" during assessor dataset creation:
→ Problem is in **attacker training generation**
→ Check GRPO generation config
→ Increase `repetition_penalty` to 1.2-1.3
→ Install vLLM: `pip install vllm`

### If "Completion preview" shows garbage during attacker training:
→ GRPO's generation is broken
→ Check if "DEBUG: Intercepted" messages appear
→ Verify generation_config is being used
→ Try manual generation test outside GRPO

### If prompts are missing `<|im_start|>` tokens:
→ Chat template not working
→ Check tokenizer configuration
→ Verify `apply_chat_template` is being called

### If everything looks good but still fails:
→ Check TRL version: `pip show trl`
→ Update TRL: `pip install --upgrade trl`
→ Try with vLLM: `pip install vllm`
→ Consider using different trainer (PPO, DPO)

## Quick Checks

Run these to verify setup:
```bash
# Test tokenizer patch
python script/test_tokenizer_patch.py

# Test model generation
python script/test_logic_tool.py --model_id your_model --num_examples 1

# Check versions
pip show trl transformers torch
```
