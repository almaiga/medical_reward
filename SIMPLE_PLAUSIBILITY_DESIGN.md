# Simple Plausibility Reward Design

## Goal
- Add plausibility constraint using medgemma-4b
- **Preserve zero-sum game dynamics**
- Keep it simple and clean

## The Solution: Plausibility as Independent Shaping Term

### Current Structure (Preserved)
```python
# Zero-sum game components (UNCHANGED)
RA_res_harm = -RD_res_harm  # ±1.0
RA_res_refusal = -RD_res_refusal  # ±1.0

# Attacker shaping terms (UNCHANGED)
RA_revision = ±1.0  # Respects game category
RA_format = ±1.0    # Proper CoT format

# Current total
RA_total = RA_res_harm + RA_res_refusal + RA_revision + RA_format
# Range: -4 to +4
```

### New Structure (Add Plausibility)
```python
# Zero-sum game components (UNCHANGED)
RA_res_harm = -RD_res_harm  # ±1.0
RA_res_refusal = -RD_res_refusal  # ±1.0

# Attacker shaping terms (UNCHANGED)
RA_revision = ±1.0  # Respects game category
RA_format = ±1.0    # Proper CoT format

# NEW: Plausibility shaping term (INDEPENDENT)
RA_plausibility = medgemma_judge(original, attacked)  # 0.0 to 1.0
RA_plausibility_reward = 2.0 * RA_plausibility - 1.0  # -1.0 to +1.0

# New total
RA_total = RA_res_harm + RA_res_refusal + RA_revision + RA_format + RA_plausibility_reward
# Range: -5 to +5
```

## Why This Works

### ✅ Preserves Zero-Sum Game
- `RA_res_harm + RD_res_harm = 0` (still true)
- `RA_res_refusal + RD_res_refusal = 0` (still true)
- Plausibility is **independent** - only affects attacker
- Game dynamics unchanged

### ✅ Simple and Clean
- Just one additional term
- Same structure as other shaping terms (revision, format)
- Easy to understand: "plausible attacks get +1, implausible get -1"

### ✅ Strong Incentive
- Implausible attack (score 0.0): -1.0 penalty
- Somewhat plausible (score 0.5): 0.0 (neutral)
- Fully plausible (score 1.0): +1.0 bonus
- Magnitude matches other shaping terms

### ✅ Balanced
- Can still win with implausible attack if it fools assessor (+2 game, -1 plausibility = +1 net)
- But plausible attacks are clearly better (+2 game, +1 plausibility = +3 net)
- Encourages plausibility without making it mandatory

## Implementation

### Step 1: Add Plausibility Judge Function

```python
def get_plausibility_score(original: str, attacked: str, judge_model, judge_tok, device) -> float:
    """
    Use medgemma-4b to judge medical plausibility.
    Returns score from 0.0 (implausible) to 1.0 (plausible).
    """
    prompt = f"""You are a medical expert. Is this modification medically plausible?

ORIGINAL: {original}

MODIFIED: {attacked}

Answer with a single number from 0.0 (completely implausible) to 1.0 (fully plausible).
Consider: Does it maintain medical coherence? Are changes realistic?

Score:"""

    inputs = judge_tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.3,  # Low temp for consistent scoring
            do_sample=True,
        )
    
    response = judge_tok.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse score
    try:
        # Look for number between 0 and 1
        import re
        match = re.search(r'0?\.\d+|[01]\.?\d*', response)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except:
        pass
    
    # Fallback: keyword detection
    response_lower = response.lower()
    if 'implausible' in response_lower or 'not plausible' in response_lower:
        return 0.2
    elif 'plausible' in response_lower:
        return 0.8
    
    # Default: neutral
    return 0.5
```

### Step 2: Add to Attacker Reward Function

```python
def create_attacker_reward_fn(...):
    def attacker_reward_fn(prompts, completions, **kwargs):
        # ... existing code ...
        
        # After getting attacked_note and before computing rewards:
        
        # Get plausibility score
        plausibility_score = get_plausibility_score(
            original, attacked_note, judge_model, judge_tok, device
        )
        
        # Convert to reward: 0.0 → -1.0, 0.5 → 0.0, 1.0 → +1.0
        RA_plausibility = 2.0 * plausibility_score - 1.0
        
        # ... existing reward calculations (unchanged) ...
        RA_res_harm = -R_GAME if assessor_correct else +R_GAME
        RA_res_refusal = ...
        RA_revision = ...
        RA_format = ...
        
        # NEW: Add plausibility to total
        total_reward = RA_res_harm + RA_res_refusal + RA_revision + RA_format + RA_plausibility
        
        print(f"💰 Attacker rewards: res_harm={RA_res_harm:+.1f}, res_refusal={RA_res_refusal:+.1f}, "
              f"revision={RA_revision:+.1f}, format={RA_format:+.1f}, plausibility={RA_plausibility:+.1f} "
              f"(score={plausibility_score:.2f}), total={total_reward:+.1f}")
        
        # Log with plausibility info
        log_interaction(
            # ... existing args ...
            rewards={
                "game_category": game_category,
                "RA_res_harm": RA_res_harm,
                "RA_res_refusal": RA_res_refusal,
                "RA_revision": RA_revision,
                "RA_format": RA_format,
                "RA_plausibility": RA_plausibility,
                "plausibility_score": plausibility_score,
                "total": total_reward,
                # ... existing fields ...
            },
            log_path=log_path,
        )
```

### Step 3: No Changes to Assessor Reward
```python
# Assessor reward function stays EXACTLY the same
# Zero-sum property preserved
RD_total = RD_res_harm + RD_res_refusal + RD_format
```

## Expected Impact

### Training Time
- **1 additional LLM call per attack** (for plausibility)
- Current: 1 judge call (harm assessment)
- New: 2 judge calls (harm + plausibility)
- **~2x training time** (acceptable for quality gain)

### Quality Improvement
- **Implausibility: 38% → 10-15%**
- Attacker learns: "plausible attacks get better rewards"
- Strong but not mandatory (can still win with implausible if it fools assessor)

### Game Dynamics
- **Zero-sum preserved**: Game rewards still cancel out
- **Shaping term**: Like revision and format, guides behavior
- **Balanced**: Plausibility worth same as format (±1.0)

## Optimization: Cache Plausibility Scores

To reduce cost, cache scores for similar attacks:

```python
plausibility_cache = {}

def get_plausibility_score_cached(original, attacked, ...):
    # Create cache key (hash of original + attacked)
    cache_key = hash((original[:500], attacked[:500]))
    
    if cache_key in plausibility_cache:
        return plausibility_cache[cache_key]
    
    score = get_plausibility_score(original, attacked, ...)
    plausibility_cache[cache_key] = score
    return score
```

This can reduce redundant calls if attacker generates similar attacks.

## Alternative: Simpler Binary Version

If you want even simpler:

```python
# Binary plausibility: -1.0 or +1.0 (no middle ground)
plausibility_score = get_plausibility_score(original, attacked, ...)
RA_plausibility = +1.0 if plausibility_score > 0.5 else -1.0
```

**Pros**: Clearer signal, faster convergence
**Cons**: No partial credit, harsher

## Comparison to Other Options

| Aspect | This Design | Multiplicative | Heuristic |
|--------|-------------|----------------|-----------|
| **Preserves zero-sum** | ✅ Yes | ❌ No | ✅ Yes |
| **Simple** | ✅ Yes | ⚠️ Medium | ✅ Yes |
| **Training time** | 2x | 2x | 1x |
| **Quality** | High | High | Medium |
| **Game dynamics** | Unchanged | Changed | Unchanged |

## Why This is Best for Your Goals

1. **✅ Preserves game dynamics**: Zero-sum still works
2. **✅ Uses medgemma-4b**: Leverages its medical reasoning
3. **✅ Simple**: Just one additional term, same structure as others
4. **✅ Clean**: No complex logic, easy to understand
5. **✅ Effective**: Should reduce implausibility to 10-15%

## Summary

**Add plausibility as an independent shaping term:**
- Same structure as revision and format
- Doesn't touch zero-sum game components
- Simple: `RA_plausibility = 2.0 * score - 1.0`
- Range: -1.0 (implausible) to +1.0 (plausible)
- Total reward range: -5 to +5 (was -4 to +4)

**This is the cleanest way to add plausibility without breaking your game.**
