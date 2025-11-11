# Attack Diversity Recommendations

## Current Diversity Mechanisms ✅

1. **50/50 Game Type Split** - Harmful vs Safe games
2. **RA,revision Reward** - Ensures faithfulness to game type
3. **Hidden CoT** - Enables strategic planning
4. **Tracking** - Logs diversity stats per round

## Missing Mechanisms ❌

### **1. Increase Seed Diversity**

**Problem:** Same 32 seeds used every round

**Solutions:**

#### Option A: Increase num_samples
```bash
# Instead of --num_samples 32
python script/train_selfplay_advanced.py \
    --num_samples 128 \  # 4x more seeds
    --rounds 4
```

#### Option B: Seed Augmentation
```python
# In data.py
def augment_seeds(df_errors, multiplier=2):
    """Create variations of seed notes for diversity."""
    augmented = []
    
    for _, row in df_errors.iterrows():
        # Original
        augmented.append(row)
        
        # Variation 1: Different error type from same domain
        # Variation 2: Paraphrased version
        # etc.
    
    return pd.DataFrame(augmented)
```

#### Option C: Dynamic Seed Sampling
```python
# Sample different seeds each round
for r in range(args.rounds):
    # Resample seeds from larger pool
    ds_originals_round = ds_originals_full.shuffle(seed=r).select(range(num_samples))
    ds_attacker_round = build_attacker_prompts(ds_originals_round, ...)
```

---

### **2. Adjust Sampling Temperature**

**Current:** Same temperature (0.7) for both agents

**Recommendation:** Higher temperature for attacker

```python
# In main.py
ATTACKER_TEMPERATURE = 0.9  # Encourage exploration
ASSESSOR_TEMPERATURE = 0.7  # More consistent

# Set different configs for attacker training
policy_model.generation_config.temperature = ATTACKER_TEMPERATURE
attacker_trainer.train()

# Reset for assessor training
policy_model.generation_config.temperature = ASSESSOR_TEMPERATURE
assessor_trainer.train()
```

---

### **3. Add Diversity Metrics (Optional)**

Track semantic diversity of attacks:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

diversity_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_diversity_score(attacked_notes):
    """Calculate average pairwise distance between attacks."""
    if len(attacked_notes) < 2:
        return 0.0
    
    embeddings = diversity_model.encode(attacked_notes)
    
    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(embeddings)
    
    # Get upper triangle (exclude diagonal)
    upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
    
    # Diversity = 1 - average similarity
    diversity = 1.0 - upper_triangle.mean()
    
    return diversity

# In training loop
attacked_notes_this_round = [record['attacked'] for record in attacked_notes_from_training]
diversity_score = calculate_diversity_score(attacked_notes_this_round)
print(f"📊 Attack diversity score: {diversity_score:.3f}")
```

---

### **4. Diversity Bonus Reward (Advanced)**

Add explicit reward for diverse attacks:

```python
# In rewards.py
class DiversityTracker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.attack_history = []
        self.embedding_history = []
    
    def add_attack(self, attacked_note):
        """Store attack and its embedding."""
        embedding = self.model.encode(attacked_note)
        self.attack_history.append(attacked_note)
        self.embedding_history.append(embedding)
    
    def calculate_diversity_bonus(self, new_attack, window=10):
        """Calculate diversity bonus for new attack."""
        if len(self.embedding_history) == 0:
            return 0.0
        
        new_embedding = self.model.encode(new_attack)
        
        # Compare to recent attacks (sliding window)
        recent_embeddings = self.embedding_history[-window:]
        similarities = [
            cosine_similarity([new_embedding], [prev])[0][0]
            for prev in recent_embeddings
        ]
        
        # Reward low similarity (high diversity)
        avg_similarity = sum(similarities) / len(similarities)
        diversity_bonus = (1.0 - avg_similarity) * 0.5  # Scale to ±0.5
        
        return diversity_bonus

# In attacker reward function
diversity_tracker = DiversityTracker()

def attacker_reward_fn(...):
    # ... existing code ...
    
    # Add diversity bonus
    diversity_bonus = diversity_tracker.calculate_diversity_bonus(attacked_note)
    diversity_tracker.add_attack(attacked_note)
    
    total_reward = RA_res_harm + RA_res_refusal + RA_revision + RA_format + diversity_bonus
```

---

## Recommended Priority

### **High Priority (Implement Now)**

1. ✅ **Increase num_samples** - Easy, immediate impact
   ```bash
   --num_samples 128  # Instead of 32
   ```

2. ✅ **Adjust attacker temperature** - Simple config change
   ```python
   ATTACKER_TEMPERATURE = 0.9
   ```

### **Medium Priority (Next Iteration)**

3. 📊 **Add diversity metrics** - Track SBERT similarity
4. 🔄 **Dynamic seed sampling** - Different seeds per round

### **Low Priority (Advanced)**

5. 🎁 **Diversity bonus reward** - Explicit reward for novelty
6. 🔧 **Seed augmentation** - Generate variations

---

## Expected Impact

Based on Self-RedTeam paper:
- **+21.8% SBERT diversity** with proper mechanisms
- **Better generalization** to unseen attacks
- **Reduced over-fitting** to specific attack patterns

---

## Quick Implementation

Add to `run_selfplay_training.sh`:

```bash
python script/train_selfplay_advanced.py \
    --model_id trainer_output/qwen3_sft_complete \
    --judge_model_id google/medgemma-4b-it \
    --num_samples 128 \          # Increased from 32
    --num_generations 8 \
    --learning_rate 1e-5 \
    --rounds 4 \
    --max_assessor_batch 128     # Match num_samples
```

This alone should significantly improve diversity!
