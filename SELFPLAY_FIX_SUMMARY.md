# Selfplay Training Fix Summary

## Problem
During assessor training phase, the reward function was receiving garbage data: `"<think>useruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseruseru"`

## Root Cause
**Double Chat Templating**: The prompts were being templated twice:
1. Once in `make_assessor_prompts()` using `tokenizer.apply_chat_template()`
2. Again by GRPO trainer internally during training

This caused the chat template role markers (`<|im_start|>user`) to be interpreted as literal text, resulting in the repeated "user" strings.

## Solution
Changed the prompt preparation to provide **raw text** to GRPO, letting GRPO handle the templating:

### Changes Made:

1. **`make_assessor_prompts()`** - Now returns raw text instead of templated prompts:
   - Removed `tokenizer.apply_chat_template()` call
   - Returns dataset with `prompt` (raw user content), `system` (system message), `original_note`, and `attacked_note`
   - GRPO will apply chat template during training

2. **`build_attacker_prompts()`** - Same fix:
   - Returns raw text prompts
   - GRPO handles templating

3. **Attacker Reward Function** - Manual templating for generation:
   - When generating assessor responses for reward calculation, manually apply chat template
   - This is needed because we're doing manual generation, not GRPO training

4. **Dataset Access via Closures**:
   - Created closures that capture the dataset for both attacker and assessor reward functions
   - Ensures `train_dataset` is properly passed to reward functions
   - Allows reward functions to access `original_note` and `attacked_note` from dataset

5. **Assessor Reward Function** - Simplified data access:
   - Removed prompt extraction logic
   - Gets both `original_note` and `attacked_note` directly from dataset
   - No more parsing corrupted prompts

## Key Insight from Self-RedTeam Paper
The paper uses **online self-play** where:
- Each training step generates fresh rollouts using the current frozen policy
- Rewards are computed immediately for just-generated interactions
- The policy is updated using these fresh experiences

Our implementation follows this pattern but uses GRPO's internal prompt handling, which requires raw text inputs.

## Testing
Run the training and verify:
1. No more "useruseruseruseru..." garbage in assessor phase
2. Proper medical notes are being evaluated
3. Rewards are calculated correctly with actual harm levels

## Files Modified
- `script/train_selfplay_advanced.py`
