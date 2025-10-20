# Self-RedTeam Paper Analysis & Key Insights

## Critical Discovery: Post-Fill Chain of Thought (CoT)

### What Self-RedTeam Does

**They use "postfill_cot"** - a technique where:

1. **During SFT**: Model learns to generate response FIRST, then reasoning AFTER
2. **Format**: `[Response] <think>[Reasoning]</think>`
3. **During RL**: Model generates both response and reasoning, gets reward on response quality

### Why This Matters for Your Case

**Your current format**: `<think>[reasoning]</think><output>[response]</output>`
- Reasoning comes FIRST (pre-fill)
- This is HARDER for the model during RL because:
  - Must commit to reasoning before seeing the task outcome
  - Can't adjust reasoning based on what it generates
  - Creates longer generation sequences

**Self-RedTeam's format**: `[response]<think>[reasoning]</think>`
- Response comes FIRST (post-fill reasoning)
- This is EASIER during RL because:
  - Model generates the answer immediately
  - Then explains its reasoning afterward
  - Reasoning can be used for training but doesn't block generation

### Their SFT Strategy

From their training script:
```bash
--sft_data "helpsteer3_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl,
            vanilla_benign_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl"
--sft_data_probs "0.5, 0.5"  # 50% harmful, 50% benign
--sft_steps 1
--sft_batches_per_step 1
--postfill_cot_loss_coef 1.0  # Train on reasoning too
```

**Key points**:
1. They generate 15k examples per dataset (30k total)
2. Mix 50/50 harmful and benign prompts
3. Use temperature=0.6, top_p=0.9 for diversity
4. Apply loss to BOTH response and reasoning
5. Do SFT DURING RL training (online adaptation)

### Their Data Generation Process

```bash
python3 -m openrlhf.cli.batch_inference \
   --eval_task generate_vllm_postfill_cot \
   --input_key vanilla \
   --label_key completion \
   --temperature 0.6 \
   --top_p 0.9 \
   --max_samples 15000
```

They:
1. Start with base prompts (vanilla harmful + benign)
2. Generate completions with postfill CoT using base model
3. Filter with WildGuard (their reward model)
4. Use filtered data for SFT during RL

## Implications for Your Medical Error Detection

### Problem with Your Current Approach

**Pre-fill reasoning** (`<think>` first):
- Model must plan the attack/assessment before executing
- During GRPO, this creates long sequences
- Model can't "see" the result before committing to reasoning
- Harder to learn from rewards

### Solution: Adopt Post-Fill Reasoning

**New format**:
```
Attacker: [modified medical note]<think>I changed dosage from 100mg to 200mg</think>
Assessor: Harmful<think>Dosage error detected - 200mg is double the standard</think>
```

**Benefits**:
1. Model generates answer immediately (shorter sequences)
2. Reasoning comes after (can reflect on what was generated)
3. More natural for RL - reward is on the main output
4. Reasoning helps model learn but doesn't block generation

### Revised Adaptation Strategy

**Step 1: Generate Post-Fill CoT Data**
- Take ms-train-733 to ms-train-1532 (800 examples)
- Format with response FIRST, reasoning AFTER
- Use existing MEDEC data (no LLM generation needed)

**Step 2: Create Two Types**

**Attacker Examples**:
```json
{
  "messages": [
    {"role": "system", "content": "Create a medical note variation with ONE error."},
    {"role": "user", "content": "Seed note: [clean note]"},
    {"role": "assistant", "content": "[note with error]<think>Changed X to Y</think>"}
  ]
}
```

**Assessor Examples**:
```json
{
  "messages": [
    {"role": "system", "content": "Classify medical note for errors."},
    {"role": "user", "content": "Note: [note to assess]"},
    {"role": "assistant", "content": "Harmful<think>Dosage error detected</think>"}
  ]
}
```

**Step 3: Mix with Original SFT**
- 50% original educational SFT (ms-train-0 to 732)
- 50% new post-fill CoT format (ms-train-733+)
- This preserves medical knowledge while adding RL-compatible format

**Step 4: Update GRPO to Expect Post-Fill**
- Modify `parse_response()` to extract response first
- Reasoning becomes optional (helps learning but not required)
- Reward based on main output, not reasoning quality

## Action Plan

1. **Create post-fill CoT data generator** (pure reformatting, no LLM)
2. **Generate 800 examples** from unused ms-train data
3. **Run 1-epoch adaptation** with mixed data
4. **Update GRPO parsing** to handle post-fill format
5. **Test** - should see immediate improvement

This aligns your approach with the Self-RedTeam paper's proven strategy!
