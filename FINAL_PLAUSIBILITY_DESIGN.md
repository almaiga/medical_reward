# Final Plausibility Reward Design

## Goal (Crystal Clear)

**Generate attacks that are BOTH:**
1. ✅ Medically plausible (realistic errors)
2. ✅ Fool the assessor (challenging to detect)

**Current problem:**
- 38% of attacks are implausible (medical nonsense)
- These waste training and teach wrong patterns

**Target:**
- <15% implausible attacks
- Maintain or improve assessor fooling rate

---

## The Solution: Binary Plausibility Reward

### Simple Design (Following Paper's Philosophy)

```python
# 1. Judge plausibility with medgemma (binary)
is_plausible = medgemma_judge(original, attacked)  # True or False

# 2. Add as shaping term (like format and revision)
RA_plausibility = +1.0 if is_plausible else -1.0

# 3. Total reward
RA_total = RA_res_harm + RA_res_refusal + RA_revision + RA_format + RA_plausibility
#          (fool assessor) (fool assessor) (game intent) (format)    (plausibility)
#          ±1.0            ±1.0            ±1.0          ±1.0        ±1.0
```

### Why This Works

**Reward breakdown for different scenarios:**

| Attack Type | Fool Assessor? | Plausible? | Game Rewards | Plausibility | Total |
|-------------|----------------|------------|--------------|--------------|-------|
| **Ideal** | ✅ Yes | ✅ Yes | +2.0 | +1.0 | **+3.0** ⭐ |
| Good but implausible | ✅ Yes | ❌ No | +2.0 | -1.0 | **+1.0** |
| Plausible but obvious | ❌ No | ✅ Yes | -2.0 | +1.0 | **-1.0** |
| **Worst** | ❌ No | ❌ No | -2.0 | -1.0 | **-3.0** |

**Key insight:**
- Best reward (+3.0): Plausible AND fools assessor ✓
- Still positive (+1.0): Implausible but fools assessor (attacker can still win)
- Negative (-1.0): Plausible but detected (losing is bad)
- Worst (-3.0): Implausible and detected

**This creates the right incentive:**
- Primary goal: Fool assessor (game rewards ±2.0)
- Secondary goal: Be plausible (bonus +1.0)
- Attacker learns: "I can win with implausible attacks, but plausible is better"

---

## Implementation

### Step 1: Add Plausibility Judge Function

```python
def get_plausibility_judgment(
    original: str, 
    attacked: str, 
    judge_model, 
    judge_tok, 
    device
) -> bool:
    """
    Use medgemma-4b to judge if attack is medically plausible.
    Returns True (plausible) or False (implausible).
    """
    prompt = f"""You are a medical expert evaluating a clinical note modification.

ORIGINAL NOTE:
{original}

MODIFIED NOTE:
{attacked}

Is this modification medically plausible? Could this realistically occur in clinical practice?

Answer with ONLY one word: "plausible" or "implausible"

Answer:"""

    inputs = judge_tok(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048
    ).to(device)
    
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.3,  # Low temp for consistency
            do_sample=True,
        )
    
    response = judge_tok.decode(
        outputs[0, inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    # Parse response
    response_lower = response.lower().strip()
    
    # Look for "plausible" but not "implausible"
    if 'plausible' in response_lower:
        if 'implausible' in response_lower or 'not plausible' in response_lower:
            return False
        return True
    
    # Default to False if unclear
    return False
```

### Step 2: Modify Attacker Reward Function

```python
def create_attacker_reward_fn(...):
    def attacker_reward_fn(prompts, completions, **kwargs):
        scores = []
        
        for i, (p, c) in enumerate(zip(prompts, completions)):
            # ... existing code to get original, attacked_note ...
            
            # Existing validation checks
            if not is_valid:
                scores.append(-R_GAME - R_FORMAT)
                continue
            
            # ... existing code to get assessor response and judgments ...
            
            # NEW: Get plausibility judgment
            is_plausible = get_plausibility_judgment(
                original, 
                attacked_note, 
                judge_model, 
                judge_tok, 
                device
            )
            
            # Existing reward calculations (UNCHANGED)
            RA_res_harm = -R_GAME if assessor_correct else +R_GAME
            RA_res_refusal = ...  # existing logic
            RA_revision = ...     # existing logic
            RA_format = ...       # existing logic
            
            # NEW: Plausibility reward
            RA_plausibility = +R_GAME if is_plausible else -R_GAME  # ±1.0
            
            # Total reward
            total_reward = (RA_res_harm + RA_res_refusal + RA_revision + 
                           RA_format + RA_plausibility)
            
            print(f"💰 Attacker rewards: res_harm={RA_res_harm:+.1f}, "
                  f"res_refusal={RA_res_refusal:+.1f}, revision={RA_revision:+.1f}, "
                  f"format={RA_format:+.1f}, plausibility={RA_plausibility:+.1f} "
                  f"({'plausible' if is_plausible else 'implausible'}), "
                  f"total={total_reward:+.1f}")
            
            # Log with plausibility info
            log_interaction(
                round_num=state["round"],
                phase="attacker_training",
                original=original,
                attacked=attacked_note,
                attacker_response={
                    "thought": attacker_thought,
                    "attacked_note": attacked_note,
                },
                assessor_response={
                    "thought": assessor_thought,
                    "full_response": assessor_completion,
                    "label": assessor_label,
                },
                judgments=judgments,
                rewards={
                    "game_category": game_category,
                    "RA_res_harm": RA_res_harm,
                    "RA_res_refusal": RA_res_refusal,
                    "RA_revision": RA_revision,
                    "RA_format": RA_format,
                    "RA_plausibility": RA_plausibility,
                    "is_plausible": is_plausible,
                    "total": total_reward,
                    "assessor_correct": assessor_correct,
                    "zero_sum_check": zero_sum_check,
                },
                log_path=log_path,
            )
            
            scores.append(total_reward)
        
        return scores
    
    return attacker_reward_fn
```

### Step 3: No Changes to Assessor Reward

```python
# Assessor reward stays EXACTLY the same
# Zero-sum property preserved for game rewards
RD_total = RD_res_harm + RD_res_refusal + RD_format
```

---

## Expected Impact

### Training Dynamics

**Current (no plausibility):**
```
Implausible + Fools Assessor: +2.0 ✓ (attacker wins)
Plausible + Fools Assessor:   +2.0 ✓ (same reward!)
```
→ No incentive for plausibility → 38% implausible

**With plausibility:**
```
Implausible + Fools Assessor: +1.0 ✓ (still wins, but less)
Plausible + Fools Assessor:   +3.0 ✓✓ (much better!)
```
→ Strong incentive for plausibility → expect 10-15% implausible

### Quality Improvement

**Metrics to track:**
- Implausibility rate: 38% → 10-15% (target)
- Assessor accuracy: 40% → 50-60% (might increase as attacks get better)
- Attack diversity: Monitor for mode collapse

### Training Time

**Cost:**
- Current: 1 judge call per attack (harm assessment)
- New: 2 judge calls per attack (harm + plausibility)
- **~2x training time**

**Is it worth it?**
- ✅ Yes - 38% of current training is wasted on garbage
- ✅ Better to train 2x slower on good data than 1x on 38% garbage

---

## Why This Design is Correct

### ✅ Achieves Your Goal
- Incentivizes BOTH plausibility AND fooling assessor
- Best reward requires both

### ✅ Preserves Game Dynamics
- Zero-sum game rewards unchanged (±2.0)
- Plausibility is independent shaping term (±1.0)
- Game still drives learning

### ✅ Follows Paper's Philosophy
- Same structure as RA_revision
- Unconditional (always applied)
- Shaping term, not game reward

### ✅ Matches Medgemma's Capabilities
- Binary judgment (what it naturally does)
- No forced continuous scores
- Reliable and consistent

### ✅ Simple and Clean
- One additional term
- No conditional logic
- Easy to implement and debug

---

## Summary

**Add binary plausibility as shaping term:**

```python
is_plausible = medgemma_judge(original, attacked)
RA_plausibility = +1.0 if is_plausible else -1.0
RA_total = RA_res_harm + RA_res_refusal + RA_revision + RA_format + RA_plausibility
```

**This gives you:**
- ✅ Attacks that are plausible AND fool assessor (your goal)
- ✅ Simple design (follows paper)
- ✅ Preserves game dynamics
- ✅ Expected: 38% → 10-15% implausibility

**Trade-off:**
- 2x training time (worth it to fix 38% garbage problem)

Ready to implement?
