# Adaptation Data Generation Complete ✅

## Summary

Successfully generated **400 post-fill CoT adaptation examples** from MEDEC data (ms-train-733 to ms-train-832).

## Data Statistics

- **Total examples**: 400
- **Source**: 100 MEDEC rows with Error Flag = 1
- **Format**: Post-fill Chain of Thought (response first, then `<think>reasoning</think>`)
- **Location**: `data/adaptation/postfill_cot_adaptation.jsonl`

### Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
| **By Role** | | |
| Attacker | 200 | 50% |
| Assessor | 200 | 50% |
| **By Type** | | |
| Harmful/Concerning | 200 | 50% |
| Safe | 200 | 50% |
| **Format** | | |
| Has `<think>` tag | 400 | 100% |

### Error Types

All 100 rows are **management** errors (most common type in ms-train-733+)

## Format Examples

### Attacker Harmful (Introduce Error)
```
System: "Create a variation with ONE substitution error"
User: "Create error variation: [clean note]"
Assistant: "[note with error]<think>Changed management approach: X to Y</think>"
```

### Attacker Safe (Keep Clean)
```
System: "Keep the seed note clean and accurate"
User: "Keep this seed note safe: [clean note]"
Assistant: "[same clean note]<think>Keeping note clean and accurate, no modifications needed</think>"
```

### Assessor Harmful (Detect Error)
```
System: "Classify medical notes for safety errors"
User: "Classify this note: [note with error]"
Assistant: "Harmful<think>Management plan error detected</think>"
```

### Assessor Safe (Recognize Clean)
```
System: "Classify medical notes for safety errors"
User: "Classify this note: [clean note]"
Assistant: "Safe<think>No medical errors detected, note is accurate</think>"
```

## Key Features

✅ **Post-fill CoT format**: Response comes FIRST, reasoning comes AFTER
✅ **Matches GRPO expectations**: Same system prompts and structure as GRPO training
✅ **Template-based reasoning**: Fast, accurate, deterministic
✅ **Leverages MEDEC structure**: Uses both Text (error) and Corrected Text (clean)
✅ **Balanced distribution**: 50/50 split across all dimensions
✅ **No data leakage**: Uses ms-train-733+ (original SFT used ms-train-0 to 732)

## Why This Works

1. **Format Alignment**: Teaches model the exact prompt/response structure GRPO uses
2. **Post-fill CoT**: Easier for RL - model generates answer immediately, reasoning after
3. **Preserves Knowledge**: Model keeps medical knowledge from original SFT
4. **Bridges Gap**: Connects educational SFT format to task-focused GRPO format

## Next Steps

### 1. Run Adaptation Training (1 epoch)

```bash
python3 script/train_qwen3_trl.py \
    --model_id trainer_output/qwen3_trl_20251020_142117 \
    --data_path data/adaptation/postfill_cot_adaptation.jsonl \
    --epochs 1 \
    --batch_size 4 \
    --grad_accumulation 4 \
    --learning_rate 1e-5 \
    --output_dir trainer_output/qwen3_adapted
```

**Settings rationale**:
- **1 epoch**: Prevent overfitting, preserve SFT knowledge
- **LR 1e-5**: Lower than original SFT (2e-5) to gently adapt
- **Small batch**: More frequent updates for format learning

**Expected time**: ~10-15 minutes for 400 examples

### 2. Update GRPO Parser for Post-Fill Format

Modify `parse_response()` in `train_selfplay_advanced.py`:

```python
def parse_response(text):
    """Parse post-fill CoT format: response first, then <think>."""
    
    # Extract thinking (optional)
    thought = ""
    if '<think>' in text and '</think>' in text:
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()
    
    # Extract output (everything before <think>, or full text if no <think>)
    if '<think>' in text:
        output = text.split('<think>')[0].strip()
    else:
        output = text.strip()
    
    return thought, output
```

### 3. Test with GRPO

```bash
bash run_selfplay_training.sh trainer_output/qwen3_adapted_TIMESTAMP
```

**Expected improvements**:
- ✅ Model generates valid responses (not garbage)
- ✅ 60-80% faithfulness pass rate (vs current 0%)
- ✅ Non-zero reward variance (enables learning)
- ✅ No mode collapse

## Comparison: Template vs GPT-Generated

| Aspect | Template-Based | GPT-Generated |
|--------|----------------|---------------|
| **Quality** | Accurate, factual | Slightly more natural |
| **Speed** | Instant | ~13 minutes |
| **Cost** | Free | ~$0.04 |
| **Consistency** | Deterministic | Variable |
| **Sufficient?** | ✅ Yes for format learning | Overkill |

**Decision**: Template-based is sufficient because:
- Adaptation teaches **format**, not **medical knowledge**
- Model already has medical reasoning from original SFT
- Simple descriptions ("Changed X to Y") work fine for format learning
- GRPO rewards will provide the real learning signal

## Files Generated

- `data/adaptation/postfill_cot_adaptation.jsonl` - Full 400 examples
- `data/adaptation/test_postfill.jsonl` - Test with 20 examples
- `script/generate_postfill_adaptation_data.py` - Generation script
- `ADAPTATION_DATA_READY.md` - This summary

Ready to proceed with adaptation training! 🚀
