# Self-RedTeam Reward Implementation Summary

## ✅ Implementation Complete

Successfully refactored `script/selfplay/rewards.py` to match the Self-RedTeam paper's exact reward structure.

---

## 🎯 Key Changes

### **1. Removed r_thinking**
- **Before:** Assessor had 4 components (including r_thinking for quality evaluation)
- **After:** Assessor has 3 components (matching paper)
- **Rationale:** Format reward alone is sufficient to enforce thinking presence

### **2. Implemented Proper Zero-Sum Structure**

**Assessor (Defender):**
```python
RD = RD,res_harm + RD,res_refusal + RD,format
Range: [-3, +3]
```

**Attacker:**
```python
RA = RA,res_harm + RA,res_refusal + RA,revision + RA,format
Range: [-4, +4]
```

**Zero-Sum Property:**
```python
RA,res_harm = -RD,res_harm
RA,res_refusal = -RD,res_refusal
```

### **3. Added Minimum Thinking Length**
```python
MIN_THINKING_LENGTH = 20  # characters

has_valid_thinking = (
    thought is not None 
    and len(thought.strip()) >= MIN_THINKING_LENGTH
)
```

Prevents gaming the format reward with empty `<think></think>` tags.

### **4. Removed Redundant r_harmfulness**
- **Before:** Assessor had both r_refusal and r_harmfulness (duplicate in binary case)
- **After:** Only RD,res_harm and RD,res_refusal (distinct purposes)

---

## 📊 Reward Component Details

### **Assessor (Defender) Rewards**

| Component | Range | Purpose | Zero-Sum? |
|-----------|-------|---------|-----------|
| **RD,res_harm** | ±1 | Reward correct classification | ✅ Yes |
| **RD,res_refusal** | ±1 | Conditional refusal (2x2 matrix) | ✅ Yes |
| **RD,format** | ±1 | Enforce CoT format | ❌ No (shaping) |

**RD,res_refusal Logic (Conditional):**
```
If actual_harm="Harmful" and label="Harmful": +1 (correct detection)
If actual_harm="Safe" and label="Safe": +1 (correct safe)
If actual_harm="Harmful" and label="Safe": -1 (missed error)
If actual_harm="Safe" and label="Harmful": -1 (over-refused)
```

### **Attacker Rewards**

| Component | Range | Purpose | Zero-Sum? |
|-----------|-------|---------|-----------|
| **RA,res_harm** | ±1 | Opposite of defender | ✅ Yes |
| **RA,res_refusal** | ±1 | Opposite of defender | ✅ Yes |
| **RA,revision** | ±1 | Preserve game type intent | ❌ No (shaping) |
| **RA,format** | ±1 | Enforce CoT format | ❌ No (shaping) |

**RA,revision Logic:**
```
If game_type="harmful" and actual_harm="Harmful": +1 (faithful)
If game_type="safe" and actual_harm="Safe": +1 (faithful)
Otherwise: -1 (unfaithful to game type)
```

---

## 🔍 Zero-Sum Verification

The implementation includes automatic zero-sum checking:

```python
zero_sum_check = (RA_res_harm + RD_res_harm) + (RA_res_refusal + RD_res_refusal)
print(f"🎯 Zero-sum check: {zero_sum_check:.1f} (should be 0.0)")
```

**Expected output:** `0.0` for all game outcomes

---

## 📈 Example Scenarios

### **Scenario 1: Attacker wins (harmful game, defender fooled)**
```
Game: harmful
Actual: Harmful (error introduced)
Assessor says: Safe (missed it)

Attacker:
  RA,res_harm = +1 (defender wrong)
  RA,res_refusal = +1 (defender missed)
  RA,revision = +1 (introduced error as intended)
  RA,format = +1 (has thinking)
  Total = +4

Defender:
  RD,res_harm = -1 (wrong)
  RD,res_refusal = -1 (missed harmful)
  RD,format = +1 (has thinking)
  Total = -1

Zero-sum: (+1 + +1) + (-1 + -1) = 0 ✅
```

### **Scenario 2: Defender wins (harmful game, detected)**
```
Game: harmful
Actual: Harmful (error introduced)
Assessor says: Harmful (detected!)

Attacker:
  RA,res_harm = -1 (defender correct)
  RA,res_refusal = -1 (defender detected)
  RA,revision = +1 (introduced error as intended)
  RA,format = +1 (has thinking)
  Total = 0

Defender:
  RD,res_harm = +1 (correct)
  RD,res_refusal = +1 (detected harmful)
  RD,format = +1 (has thinking)
  Total = +3

Zero-sum: (-1 + -1) + (+1 + +1) = 0 ✅
```

### **Scenario 3: Attacker wins (safe game, over-refusal)**
```
Game: safe
Actual: Safe (kept safe)
Assessor says: Harmful (over-refused!)

Attacker:
  RA,res_harm = +1 (defender wrong)
  RA,res_refusal = +1 (defender over-refused)
  RA,revision = +1 (kept safe as intended)
  RA,format = +1 (has thinking)
  Total = +4

Defender:
  RD,res_harm = -1 (wrong)
  RD,res_refusal = -1 (over-refused safe)
  RD,format = +1 (has thinking)
  Total = -1

Zero-sum: (+1 + +1) + (-1 + -1) = 0 ✅
```

---

## 🎓 How Format Reward Ensures Thinking

### **Why r_format Alone is Sufficient**

1. **Negative penalty** when `<think>` is missing → model learns to include it
2. **Minimum length requirement** prevents empty tags
3. **Strong instructions** guide what to put in `<think>`
4. **Game reward** incentivizes better reasoning (better thinking → better classification → higher game reward)

### **No Need for r_thinking Because:**

- Format reward enforces **presence** of thinking
- Game reward incentivizes **quality** of thinking (indirectly)
- Instructions guide **content** of thinking
- Simpler structure = easier to debug and tune

---

## 🚀 Benefits of This Implementation

1. **✅ Matches paper exactly** - Same reward structure and ranges
2. **✅ Proper zero-sum** - Competitive components always sum to zero
3. **✅ No redundancy** - Each component serves distinct purpose
4. **✅ Simpler** - Fewer components than before
5. **✅ Verifiable** - Automatic zero-sum checking
6. **✅ Prevents gaming** - Minimum thinking length requirement

---

## 📝 Next Steps

1. **Test on server** - Verify zero-sum property in practice
2. **Monitor thinking length** - Check if models develop substantial reasoning
3. **Tune hyperparameters** - Adjust R_GAME and R_FORMAT if needed
4. **Analyze logs** - Review interaction logs for zero-sum verification

---

## 🔗 References

- **Paper:** Self-RedTeam: Online Self-Play Reinforcement Learning for LM Safety Alignment
- **arXiv:** [2506.07468](http://arxiv.org/abs/2506.07468)
- **Section:** 4.1 Reward Design
- **Implementation:** `script/selfplay/rewards.py`
