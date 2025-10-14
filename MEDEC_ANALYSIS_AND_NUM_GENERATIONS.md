# MEDEC Dataset Analysis & num_generations Explanation

## 📊 MEDEC Dataset Structure

### Dataset Overview
- **Total rows**: 574
- **Clean notes (Error Flag = 0)**: 255 notes
- **Notes with errors (Error Flag = 1)**: 319 notes

### Key Columns
1. **Text**: The medical note (may contain error if Error Flag = 1)
2. **Corrected Text**: The corrected version (only populated for Error Flag = 1)
3. **Error Flag**: 0 = clean, 1 = contains error
4. **Error Type**: Type of error (e.g., "causalOrganism")

### Important Discovery
For **Error Flag = 0** (clean notes):
- `Text` contains the clean note
- `Corrected Text` is NaN (empty)

For **Error Flag = 1** (notes with errors):
- `Text` contains the note WITH the error
- `Corrected Text` contains the FIXED version
- These are DIFFERENT texts (not the same)

### Implications for Our Implementation

**Current code issue:**
```python
df["original"] = df.apply(
    lambda r: (r["Corrected Text"].strip() or r["Text"]), axis=1
)
```

This means:
- For Error Flag = 1: Uses "Corrected Text" (the FIXED version) ✅
- For Error Flag = 0: Uses "Text" (the clean version) ✅

**This is actually correct!** We always start with clean notes.

---

## 🎮 num_generations Parameter Explained

### What is num_generations?

`num_generations` is a **GRPO-specific parameter** that controls how many completions the model generates for each prompt during training.

### Current Configuration
```python
common_cfg = dict(
    num_generations=args.num_generations,  # Default: 2
    generation_batch_size=args.num_generations * 2,  # Default: 4
    ...
)
```

### How GRPO Uses It

GRPO (Group Relative Policy Optimization) works by:

1. **For each prompt**, generate `num_generations` different completions
2. **Compare** these completions against each other
3. **Compute relative rewards** - which completion is better?
4. **Update policy** to favor better completions

### Example with num_generations=2

**Prompt**: "Add ONE subtle medical error to this note: [medical note]"

**GRPO generates 2 completions:**
- Completion A: Changes dosage from 100mg to 200mg
- Completion B: Changes diagnosis from "pneumonia" to "bronchitis"

**Reward function evaluates both:**
- Completion A: Assessor detects it → Reward = -2.0
- Completion B: Assessor misses it → Reward = +2.0

**GRPO learns**: Completion B is better, increase probability of similar attacks

### Why Both Attacker and Assessor Use It

**Attacker Training:**
- Generates `num_generations` different attacks per note
- Learns which attack strategies fool the assessor
- Higher diversity = better exploration

**Assessor Training:**
- Generates `num_generations` different assessments per attacked note
- Learns which assessment strategies are more accurate
- Helps avoid overfitting to single response pattern

### Current Problem

Looking at your results:
```
Round 1: Attacker[Deception: 92.2%, Reward: 0.85]
Round 4: Attacker[Deception: 89.1%, Reward: 0.95]
```

With `num_generations=2`, the attacker is generating 2 completions per prompt, but:
- **Both completions likely use the same strategy** (output unchanged note)
- GRPO can't distinguish between them (both get similar rewards)
- No diversity pressure

### Recommendations

1. **Keep num_generations=2** (standard for GRPO)
2. **Add diversity through prompt types** (our dual-game approach)
3. **Consider increasing to num_generations=4** if you have GPU memory
   - More completions = better exploration
   - But 4x more compute per training step

### Memory Impact

```
num_generations=2: 
- 16 prompts × 2 generations = 32 completions per batch
- Memory: ~8GB VRAM

num_generations=4:
- 16 prompts × 4 generations = 64 completions per batch  
- Memory: ~16GB VRAM
```

---

## 🎯 Summary

### MEDEC Dataset
✅ Current usage is correct - we always start with clean notes
✅ We have 255 clean notes available for training

### num_generations
✅ Affects both attacker and assessor training
✅ Controls exploration diversity within GRPO
✅ Current value (2) is reasonable, but diversity comes from prompt types, not just num_generations

### Next Steps
1. Implement dual-game approach (inject vs keep_clean)
2. Use clean notes (Error Flag = 0) for both game types
3. Keep num_generations=2 for now
4. Monitor diversity metrics in logs
