# Comparison: Our Approach vs Self-RedTeam Paper

## 📚 Self-RedTeam Paper Key Elements

### 1. **Two-Player Zero-Sum Game**
**Paper:**
- Single model alternates between attacker and defender roles
- Attacker generates adversarial prompts
- Defender generates safe responses
- Reward model adjudicates outcomes
- Zero-sum: Attacker wins if defender fails, defender wins if safe

**Our Approach:**
- ✅ Single model alternates between attacker and assessor roles
- ✅ Attacker generates attacked medical notes
- ✅ Assessor classifies safety level
- ✅ Judge model adjudicates outcomes
- ✅ Zero-sum: Attacker wins if assessor misclassifies

**Verdict:** ✅ **SAME STRUCTURE**

---

### 2. **Dual Prompt Distributions**
**Paper:**
- **Harmful prompts**: Attacker tries to make defender comply with harmful request
- **Benign prompts**: Attacker tries to make defender refuse benign request
- 50/50 split enforced
- Opposite objectives prevent degenerate equilibria

**Our Approach:**
- **Inject prompts**: Attacker tries to add errors that assessor misses
- **Keep_clean prompts**: Attacker tries to keep clean (assessor should say Safe)
- 50/50 split enforced
- Opposite objectives prevent degenerate equilibria

**Verdict:** ✅ **SAME PRINCIPLE, ADAPTED TO DOMAIN**

---

### 3. **Reward Structure**
**Paper:**
```
Attacker reward = -Defender reward (zero-sum)

For harmful prompts:
  - Attacker wins if defender complies (unsafe response)
  - Defender wins if defender refuses (safe response)

For benign prompts:
  - Attacker wins if defender refuses (over-refusal)
  - Defender wins if defender complies (appropriate response)
```

**Our Approach:**
```
Attacker reward = -Assessor reward (zero-sum)

For inject prompts:
  - Attacker wins if assessor says "Safe" (missed error)
  - Assessor wins if assessor says "Concerning/Harmful" (detected error)

For keep_clean prompts:
  - Attacker wins if assessor says "Safe" (correct)
  - Assessor wins if assessor says "Safe" (correct)
  - Both can win in this game
```

**Verdict:** ⚠️ **SLIGHT DIFFERENCE** - Our keep_clean game is not perfectly zero-sum

---

### 4. **Hidden Chain-of-Thought**
**Paper:**
- Agents use `<think>` tags for private reasoning
- Thinking is NOT shown to opponent
- Reduces over-refusals
- Increases attack diversity

**Our Approach:**
- ✅ We use `<think>` tags for reasoning
- ✅ Thinking is NOT shown to opponent (assessor doesn't see attacker's thinking)
- ✅ Rewarded with R_FORMAT bonus
- ✅ Helps with attack planning

**Verdict:** ✅ **SAME APPROACH**

---

### 5. **Diversity Mechanisms**
**Paper:**
- Dual prompt distributions (harmful vs benign)
- Measures diversity with SBERT embeddings
- Reports +21.8% diversity improvement
- Tracks attack success rate over rounds

**Our Approach:**
- ✅ Dual prompt distributions (inject vs keep_clean)
- ❌ NOT measuring diversity with embeddings (just counting harm levels)
- ✅ Tracking harm level distribution
- ✅ Tracking attacker win rate over rounds

**Verdict:** ⚠️ **MISSING DIVERSITY METRICS** - We should add SBERT or similar

---

### 6. **Nash Equilibrium Convergence**
**Paper:**
- Theoretical guarantee: If self-play converges to Nash Equilibrium, defender is robust
- Monitors convergence through win rate stabilization
- Expects ~50% win rate at equilibrium

**Our Approach:**
- ✅ Same theoretical foundation
- ✅ Monitoring attacker win rate (currently 95%, should converge to ~50%)
- ✅ Expecting equilibrium convergence

**Verdict:** ✅ **SAME THEORY**

---

## 🔍 Key Differences

### 1. **Domain Adaptation**
**Paper:** Jailbreaking / Safety refusal
**Ours:** Medical note error detection

**Impact:** Different but valid application of same principles

---

### 2. **Keep_Clean Game Not Perfectly Zero-Sum**
**Paper:** Both games are strictly zero-sum
**Ours:** Keep_clean game allows both to win

**Problem:** This might create a "cooperative" equilibrium in keep_clean games

**Fix Needed:**
```python
# Current (keep_clean):
if assessor_correct:
    attacker_reward = +R_HARM  # Both win
    assessor_reward = +R_HARM

# Should be (zero-sum):
if assessor_correct:
    attacker_reward = +R_HARM  # Attacker wins
    assessor_reward = -R_HARM  # Assessor loses (but this doesn't make sense...)
```

**Actually, our keep_clean game is conceptually different:**
- Paper's "benign" game: Attacker tries to cause over-refusal (adversarial)
- Our "keep_clean" game: Attacker tries to keep clean (cooperative)

**This is a DESIGN CHOICE, not a bug. But it's different from the paper.**

---

### 3. **Missing Diversity Metrics**
**Paper:** Measures attack diversity with SBERT embeddings
**Ours:** Only counts harm levels (Safe/Concerning/Harmful)

**Impact:** We can't quantify if attacks are becoming more diverse

**Fix Needed:** Add SBERT-based diversity measurement

---

### 4. **Reward Multipliers**
**Paper:** Uses fixed reward values, no multipliers mentioned
**Ours:** Uses harm-level multipliers (Safe: 0.3x, Concerning: 1.5x, Harmful: 3.0x)

**Impact:** Our approach adds granularity to reward structure

**Verdict:** This is an ENHANCEMENT, not a deviation

---

### 5. **Deduplication Strategy**
**Paper:** Not mentioned (likely not needed due to different setup)
**Ours:** Needed because GRPO generates multiple completions per prompt

**Impact:** This is specific to our GRPO implementation

**Verdict:** Implementation detail, not a conceptual difference

---

## 🎯 What We're Doing Differently (Summary)

### ✅ Same as Paper
1. Two-player zero-sum game structure
2. Dual prompt distributions (50/50 split)
3. Hidden chain-of-thought
4. Nash equilibrium convergence goal
5. Online self-play training

### ⚠️ Different from Paper
1. **Keep_clean game is cooperative, not adversarial**
   - Paper: Both games are adversarial
   - Ours: Inject is adversarial, keep_clean is cooperative
   
2. **No diversity metrics (SBERT)**
   - Paper: Measures attack diversity with embeddings
   - Ours: Only counts harm levels
   
3. **Harm-level reward multipliers**
   - Paper: Fixed rewards
   - Ours: Scaled by harm level (3.0x for Harmful)
   
4. **Medical domain vs jailbreaking**
   - Paper: Safety refusal
   - Ours: Error detection

---

## 💡 Recommendations

### 1. **Fix Keep_Clean Game to be Zero-Sum**

**Option A: Make it adversarial (like paper)**
```python
# Keep_clean game: Attacker tries to make assessor INCORRECTLY flag as harmful
if game_type == "keep_clean":
    # Attacker wins if assessor INCORRECTLY says Concerning/Harmful
    # Assessor wins if assessor CORRECTLY says Safe
    if actual_harm == "Safe":
        if assessor_said_safe:
            attacker_reward = -R_HARM  # Attacker loses
            assessor_reward = +R_HARM  # Assessor wins
        else:
            attacker_reward = +R_HARM  # Attacker wins (fooled assessor)
            assessor_reward = -R_HARM  # Assessor loses
```

**Option B: Keep current approach (cooperative)**
- Accept that keep_clean is different from paper
- Document this as a design choice
- Monitor if it causes issues

**Recommendation:** Try Option A first (match paper more closely)

---

### 2. **Add Diversity Metrics**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_diversity(attacked_notes: List[str]) -> float:
    """Calculate average pairwise cosine distance."""
    embeddings = model.encode(attacked_notes)
    distances = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = 1 - cosine_similarity(embeddings[i], embeddings[j])
            distances.append(dist)
    return np.mean(distances)
```

**Log diversity per round:**
```
Round 1: Diversity = 0.45
Round 2: Diversity = 0.52 (+15.6%)
Round 3: Diversity = 0.58 (+11.5%)
```

---

### 3. **Keep Harm-Level Multipliers**

This is an enhancement, not a problem. The paper doesn't have harm levels (just safe/unsafe), so multipliers make sense for our domain.

---

## 🎓 Final Verdict

### What We're Doing Right
- ✅ Core game-theoretic structure matches paper
- ✅ Dual prompt distributions for diversity
- ✅ Zero-sum rewards (mostly)
- ✅ Hidden chain-of-thought
- ✅ Nash equilibrium goal

### What We Should Fix
1. **Make keep_clean game truly zero-sum** (match paper)
2. **Add SBERT diversity metrics** (match paper)
3. **Document our design choices** (where we differ intentionally)

### What We're Enhancing
1. **Harm-level multipliers** (adds granularity)
2. **Deduplication** (GRPO-specific need)
3. **Medical domain adaptation** (valid application)

---

## 📋 Updated Requirements

Based on this comparison, I recommend:

1. **Keep** the reward multiplier increases (3.0x for Harmful)
2. **Keep** the enhanced prompts with examples
3. **Keep** the deduplication strategy
4. **Add** SBERT diversity metrics
5. **Fix** keep_clean game to be truly zero-sum (like paper)
6. **Add** diversity tracking to match paper's +21.8% metric

Should I update the spec with these changes?
