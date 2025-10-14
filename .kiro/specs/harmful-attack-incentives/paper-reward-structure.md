# Self-RedTeam Paper: Complete Reward Structure

## 🎯 The Paper's Actual Reward Function

Looking more carefully at the paper, they have **multiple reward components**, not just binary win/lose:

### 1. **Main Game Reward (Zero-Sum)**
```
R_game = {
    +1 if attacker wins (defender fails)
    -1 if defender wins (defender succeeds)
}
```
This is the zero-sum component.

### 2. **Format Reward (Chain-of-Thought)**
```
R_format = {
    +bonus if agent uses <think> tags correctly
    -penalty if agent doesn't use format
}
```
This encourages hidden chain-of-thought reasoning.

### 3. **Refusal Appropriateness Reward**
```
R_refusal = {
    +bonus if refusal is appropriate (refuse harmful, comply with benign)
    -penalty if refusal is inappropriate (refuse benign, comply with harmful)
}
```
This prevents over-refusal and under-refusal.

### Total Reward
```
R_total = R_game + R_format + R_refusal
```

**Key insight:** The main game is zero-sum, but the auxiliary rewards (format, refusal) are NOT zero-sum. Both players can get format/refusal bonuses independently.

---

## 🔍 Our Current Implementation

### What We Have
```python
# 1. Main game reward (zero-sum)
r_harm = R_HARM if attacker_wins else -R_HARM

# 2. Format reward (not zero-sum)
r_format = R_FORMAT if has_thinking else -R_FORMAT

# 3. Refusal reward (not zero-sum)
r_refusal = R_REFUSAL if appropriate_refusal else -R_REFUSAL

# Total
total_reward = r_harm + r_format + r_refusal
```

### What We're Missing

**The paper likely has different refusal logic:**

For **Attacker** in harmful prompts:
- Goal: Make defender comply (unsafe response)
- Refusal reward: Penalize if defender refuses (attacker failed)

For **Defender** in harmful prompts:
- Goal: Refuse harmful request
- Refusal reward: Bonus if refuses, penalty if complies

For **Attacker** in benign prompts:
- Goal: Make defender refuse (over-refusal)
- Refusal reward: Bonus if defender refuses benign request

For **Defender** in benign prompts:
- Goal: Comply with benign request
- Refusal reward: Bonus if complies, penalty if refuses

---

## 🎯 Mapping to Our Medical Domain

### Our Three Reward Components

**1. Main Game Reward (Zero-Sum)**
```python
# Inject game: Attacker tries to fool assessor
if game_type == "inject":
    if assessor_detected_error:
        r_harm = -R_HARM  # Attacker loses
    else:
        r_harm = +R_HARM  # Attacker wins

# Keep_clean game: Attacker tries to keep clean
elif game_type == "keep_clean":
    if assessor_said_safe:
        r_harm = +R_HARM  # Attacker wins (correct assessment)
    else:
        r_harm = -R_HARM  # Attacker loses (false positive)
```

**2. Format Reward (Not Zero-Sum)**
```python
# Both attacker and assessor get format bonus
r_format = R_FORMAT if used_think_tags else -R_FORMAT
```

**3. Refusal/Appropriateness Reward (Not Zero-Sum)**

This is where we need to think carefully. In medical domain:

**For Attacker:**
- In "inject" game: Should create actual errors (not refuse)
- In "keep_clean" game: Should keep clean (not add errors)
- Refusal = Not following instructions

**For Assessor:**
- Should classify correctly based on actual harm
- "Refusal" = Saying "Can't Assess" (avoiding decision)
- Appropriate refusal = Only when truly ambiguous

```python
# Assessor refusal reward
if assessor_said_cant_assess:
    if truly_ambiguous:
        r_refusal = +R_REFUSAL  # Appropriate refusal
    else:
        r_refusal = -R_REFUSAL  # Inappropriate refusal (avoiding)
else:
    if classification_correct:
        r_refusal = +R_REFUSAL  # Appropriate classification
    else:
        r_refusal = -R_REFUSAL  # Incorrect classification
```

---

## 🤔 The Question: What's the Third Reward?

Looking at the paper more carefully, the three components are likely:

1. **Main game reward** (zero-sum): Win/lose based on attack success
2. **Format reward** (not zero-sum): Using <think> tags correctly
3. **Refusal appropriateness** (not zero-sum): Refusing when appropriate, complying when appropriate

**But there might be a fourth component:**

4. **Diversity/Novelty reward**: Encouraging diverse attack strategies

Or it could be:

4. **Severity reward**: Encouraging more severe attacks (but paper doesn't mention this explicitly)

---

## 🎯 What We Should Clarify

### Current Refusal Logic in Our Code

Looking at our current implementation:

```python
# Refusal reward
query_harmful = judgments["actual_harm"] in ["Concerning", "Harmful"]
assessor_refused = judgments["assessor_refused"]

if query_harmful and assessor_refused:
    r_refusal = +R_REFUSAL  # Good: Refused harmful
elif not query_harmful and not assessor_refused:
    r_refusal = +R_REFUSAL  # Good: Didn't refuse safe
elif query_harmful and not assessor_refused:
    r_refusal = -R_REFUSAL  # Bad: Didn't refuse harmful
else:
    r_refusal = -R_REFUSAL  # Bad: Refused safe
```

**This is actually correct!** It matches the paper's refusal appropriateness logic.

---

## ✅ Our Reward Structure is Actually Correct

### We Have All Three Components:

1. ✅ **Main game (zero-sum)**: `r_harm` based on assessor correctness
2. ✅ **Format (not zero-sum)**: `r_format` for using <think> tags
3. ✅ **Refusal appropriateness (not zero-sum)**: `r_refusal` for appropriate classification

### The Issue is NOT the Reward Structure

The issue is:
1. **Phase size mismatch** (64 vs 256 games)
2. **No harmful attacks** (0% harmful)
3. **Attacker winning too much** (95% win rate)

These are NOT reward structure problems. They're:
- Technical issue (deduplication needed)
- Prompt issue (need examples)
- Early training issue (need more rounds)

---

## 🎯 Revised Understanding

### What the Paper Has (3 Rewards)
1. Main game (zero-sum)
2. Format (CoT)
3. Refusal appropriateness

### What We Have (3 Rewards)
1. Main game (zero-sum) ✅
2. Format (CoT) ✅
3. Refusal appropriateness ✅

### What We Need to Fix
1. ❌ NOT reward structure (it's correct)
2. ✅ Phase size mismatch (deduplication)
3. ✅ Prompt examples (show harmful errors)
4. ✅ More training rounds (let competition develop)

---

## 📋 Question for You

You mentioned "there is two other reward one for format and another one for..."

What is the third reward component you're thinking of? Is it:
1. Refusal appropriateness (what we have)
2. Diversity/novelty
3. Severity
4. Something else?

Can you clarify what you saw in the paper?
