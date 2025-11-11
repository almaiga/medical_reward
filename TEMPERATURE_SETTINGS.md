# Temperature Settings for Self-Play Training

## Overview

Different temperature settings are used for attacker and assessor to optimize their respective objectives.

---

## Temperature Configuration

### **Attacker Temperature: 1.0** 🔥

**Purpose:** Maximize attack diversity and exploration

**Why higher temperature?**
- Encourages diverse attack strategies
- Explores different error types
- Prevents overfitting to specific patterns
- Increases stochasticity in generation

**Expected behavior:**
- More varied attacked notes
- Different error patterns each round
- Higher entropy in policy
- Better generalization

### **Assessor Temperature: 0.7** ❄️

**Purpose:** Consistent and reliable detection

**Why lower temperature?**
- More deterministic classifications
- Consistent detection behavior
- Reduces false positives
- Stable policy convergence

**Expected behavior:**
- Consistent classifications
- Lower variance in responses
- More confident predictions
- Better accuracy

---

## Implementation

### **In Training Loop**

```python
# Temperature constants
ATTACKER_TEMPERATURE = 1.0  # Higher for exploration
ASSESSOR_TEMPERATURE = 0.7  # Lower for consistency

# Before attacker training
policy_model.generation_config.temperature = ATTACKER_TEMPERATURE

# Before assessor training
policy_model.generation_config.temperature = ASSESSOR_TEMPERATURE
```

### **Frozen Assessor**

When attacker queries frozen assessor during training:
```python
frozen_assessor.generate(
    temperature=0.7,  # Consistent with assessor training
    ...
)
```

---

## Rationale from Literature

### **Self-RedTeam Paper**

The paper doesn't specify exact temperatures, but mentions:
> "Self-RedTeam uncovers more diverse attacks (+21.8% SBERT)"

Higher attacker temperature contributes to this diversity.

### **Standard RL Practice**

- **Exploration phase** (attacker): Higher temperature
- **Exploitation phase** (assessor): Lower temperature

### **Medical Safety Context**

- **Attacker:** Needs to explore many error types
- **Assessor:** Needs consistent, reliable detection

---

## Expected Impact

### **Attack Diversity**

With temperature 1.0, attacker should generate:
- More varied error types
- Different phrasing of errors
- Subtle vs obvious errors
- Novel attack patterns

### **Detection Consistency**

With temperature 0.7, assessor should show:
- Stable classifications
- Lower variance in confidence
- Consistent behavior across rounds
- Better convergence

---

## Monitoring

### **Check Entropy**

Higher temperature → Higher entropy:
```
Attacker entropy: ~1.3-1.5 (with temp 1.0)
Assessor entropy: ~1.0-1.2 (with temp 0.7)
```

### **Check Diversity**

Track SBERT similarity between attacks:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(attacked_notes)
# Lower similarity = higher diversity
```

---

## Tuning Guidelines

### **If Attacker Diversity Too Low**

Increase attacker temperature:
```python
ATTACKER_TEMPERATURE = 1.2  # Even more exploration
```

### **If Attacker Too Random**

Decrease attacker temperature:
```python
ATTACKER_TEMPERATURE = 0.9  # More focused
```

### **If Assessor Too Inconsistent**

Decrease assessor temperature:
```python
ASSESSOR_TEMPERATURE = 0.5  # More deterministic
```

### **If Assessor Too Rigid**

Increase assessor temperature:
```python
ASSESSOR_TEMPERATURE = 0.8  # More flexible
```

---

## Comparison

| Aspect | Attacker (1.0) | Assessor (0.7) |
|--------|----------------|----------------|
| **Goal** | Explore errors | Detect errors |
| **Diversity** | High | Moderate |
| **Consistency** | Low | High |
| **Entropy** | ~1.3-1.5 | ~1.0-1.2 |
| **Variance** | High | Low |
| **Exploration** | Maximum | Balanced |

---

## Alternative Approaches

### **Adaptive Temperature**

Decrease temperature over rounds:
```python
# Start high, decrease over time
ATTACKER_TEMPERATURE = 1.0 - (0.1 * round_num / total_rounds)
```

### **Curriculum Learning**

Start with high temperature, gradually decrease:
```python
# Round 1-2: High exploration (1.0)
# Round 3-4: Moderate (0.9)
# Round 5+: Focused (0.8)
```

### **Per-Game-Type Temperature**

Different temperatures for harmful vs safe games:
```python
if game_type == "harmful":
    temperature = 1.0  # More exploration for errors
else:
    temperature = 0.8  # More focused for safe
```

---

## Summary

✅ **Attacker: 1.0** - Maximizes diversity and exploration  
✅ **Assessor: 0.7** - Ensures consistent detection  
✅ **Frozen Assessor: 0.7** - Matches assessor behavior  
✅ **Automatic switching** - Temperature adjusted per phase  

This configuration balances exploration (attacker) with exploitation (assessor) for optimal self-play dynamics.
