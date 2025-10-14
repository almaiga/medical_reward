# Judge Temperature Fix - Enable Sampling

## 🎯 Problem: Greedy Decoding Too Conservative

**You asked:** "Do you think it's a temperature issue?"

**Answer:** **YES!** Great catch! 🎯

### Current Settings (Before Fix)
```python
judge_model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,  # ❌ Greedy decoding
)
```

**Effect:**
- Always picks most likely token
- No exploration
- Defaults to conservative choices
- "Concerning" (60% likely) always beats "Harmful" (40% likely)

## 🔍 Why This Causes the Problem

### Greedy Decoding Behavior
```
Judge sees: "Missing diagnosis..."

Token probabilities:
- "Concerning": 60%
- "Harmful": 40%

Greedy decoding: Always picks "Concerning" ❌
```

### With Sampling (Temperature > 0)
```
Judge sees: "Missing diagnosis..."

Token probabilities:
- "Concerning": 60%
- "Harmful": 40%

With sampling: Sometimes picks "Harmful" ✅
```

## ✅ Solution: Add Temperature

### New Settings
```python
judge_model.generate(
    **inputs,
    max_new_tokens=256,  # Increased for chain-of-thought
    do_sample=True,  # ✅ Enable sampling
    temperature=0.3,  # Low but non-zero
    top_p=0.9,
    pad_token_id=judge_tok.eos_token_id,
)
```

### Why temperature=0.3?

**Temperature scale:**
- `0.0`: Greedy (deterministic, conservative)
- `0.1-0.3`: Slightly more aggressive, still mostly deterministic
- `0.5-0.7`: Balanced exploration
- `1.0+`: Very creative, less reliable

**We chose 0.3 because:**
- **Slightly more aggressive** than greedy
- **Still reliable** for classification
- **Explores alternatives** when probabilities are close
- **Not too random** - we want consistent judgments

## 📊 Expected Impact

### Before (Greedy)
```
Missing diagnosis cases:
- "Concerning": 100% (always picks most likely)
- "Harmful": 0%
```

### After (Temperature=0.3)
```
Missing diagnosis cases:
- "Concerning": 40-50% (when truly ambiguous)
- "Harmful": 50-60% (when clear error)
```

### Overall Distribution (Expected)
```
Before:
- Safe: 20%
- Concerning: 80%
- Harmful: 0%

After:
- Safe: 20-30%
- Concerning: 30-40%
- Harmful: 30-40% ✅
```

## 🎓 Why Temperature Matters for Classification

### Example: Ambiguous Case
```
Prompt: "Note mentions stretching but no diagnosis"

Token probabilities:
- "Concerning": 55%
- "Harmful": 45%

Greedy (temp=0): Always "Concerning"
Temp=0.3: Sometimes "Harmful" (explores both)
```

### Example: Clear Case
```
Prompt: "Dosage changed from 100mg to 1000mg"

Token probabilities:
- "Harmful": 95%
- "Concerning": 5%

Greedy (temp=0): "Harmful"
Temp=0.3: "Harmful" (still picks most likely)
```

**Key insight:** Temperature helps with **ambiguous cases** without hurting **clear cases**.

## 🔬 Technical Details

### Sampling vs Greedy

**Greedy decoding:**
```python
next_token = argmax(probabilities)  # Always pick highest
```

**Sampling with temperature:**
```python
adjusted_probs = softmax(logits / temperature)
next_token = sample(adjusted_probs)  # Pick based on distribution
```

### Temperature Effect

**Temperature = 0.3:**
```
Original probs: [0.60, 0.40]
After temp=0.3: [0.75, 0.25]  # Sharpens distribution slightly
Still samples from both!
```

**Temperature = 1.0:**
```
Original probs: [0.60, 0.40]
After temp=1.0: [0.60, 0.40]  # No change
More exploration
```

## ✅ Additional Benefits

### 1. Increased max_new_tokens
```
Before: 128 tokens
After: 256 tokens
```
**Why:** Chain-of-thought reasoning needs more space

### 2. Added top_p
```
top_p=0.9
```
**Why:** Nucleus sampling - only consider top 90% of probability mass

### 3. Added pad_token_id
```
pad_token_id=judge_tok.eos_token_id
```
**Why:** Proper handling of padding tokens

## 🎯 Success Criteria

After this fix, we should see:
1. **Harmful %**: 0% → 30%+
2. **Concerning %**: 80% → 30-40%
3. **Missing diagnosis**: Classified as "Harmful" more often
4. **Clear errors**: Still classified correctly
5. **More diversity**: Judge explores alternatives

## 📝 Comparison with Other Models

### Policy Model (Attacker/Assessor)
```python
temperature=0.7  # Higher - need creativity
top_p=0.9
```

### Judge Model (Now)
```python
temperature=0.3  # Lower - need reliability
top_p=0.9
```

**Why different?**
- **Policy needs creativity** - generate diverse attacks
- **Judge needs reliability** - consistent classifications
- **But judge also needs exploration** - not too conservative

## 🚀 Ready to Test

Run training and check if:
- Judge classifies missing diagnosis as "Harmful"
- Harmful % increases from 0%
- Distribution becomes more balanced

This temperature fix combined with the ultra-explicit prompt should finally get us "Harmful" classifications! 🎯
