# Training Metrics Logging Guide

## Overview

The self-play training now automatically logs all GRPO training metrics to track performance over time.

## Files Generated

For each training run, three files are created in the `results/` directory:

1. **`{experiment}_metrics.jsonl`** - Detailed metrics for each training phase
2. **`{experiment}_metrics_summary.json`** - Human-readable summary
3. **`{experiment}_interactions.jsonl`** - Detailed interaction logs (existing)

---

## Metrics Logged

### **Training Metrics (Per Phase)**

| Metric | Description | Range |
|--------|-------------|-------|
| `loss` | Training loss | Negative = learning |
| `grad_norm` | Gradient norm | ~0.5-2.0 typical |
| `learning_rate` | Current LR | Decreases over time |
| `epoch` | Training epoch | 0.0-1.0 per round |

### **GRPO-Specific Metrics**

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `reward` | Mean reward | Higher = better performance |
| `reward_std` | Reward variance | >0 = diverse outcomes |
| `frac_reward_zero_std` | Fraction with no variance | <0.5 = good exploration |
| `entropy` | Policy entropy | >1.0 = maintains exploration |
| `clip_ratio/*` | Policy clipping ratios | <1% = stable updates |

### **Generation Metrics**

| Metric | Description |
|--------|-------------|
| `completions/mean_length` | Average tokens per generation |
| `completions/min_length` | Shortest generation |
| `completions/max_length` | Longest generation |
| `completions/clipped_ratio` | Fraction truncated (should be 0) |

### **Round Summary Metrics**

| Metric | Description |
|--------|-------------|
| `diversity/*` | Game type distribution |
| `judge/total` | Total judge evaluations |
| `judge/percentages` | Classification distribution |

---

## Viewing Metrics

### **Quick Summary**

```bash
python script/view_training_metrics.py results/20251111_123456_model_grpo_metrics_summary.json
```

**Output:**
```
============================================================
TRAINING METRICS SUMMARY
============================================================
Experiment: 20251111_123456_model_grpo
Start: 2025-11-11T12:34:56
End: 2025-11-11T14:56:78
============================================================

Round 1:
  Attacker:
    Loss: -0.1148
    Reward: -1.1875 ± 0.8413
    Entropy: 1.1313
  Assessor:
    Loss: -0.0892
    Reward: 1.2341 ± 0.7654
    Entropy: 1.0987

...

============================================================
OVERALL STATISTICS
============================================================
Total Rounds: 4

Attacker:
  Avg Loss: -0.1234
  Avg Reward: -1.0567

Assessor:
  Avg Loss: -0.0987
  Avg Reward: 1.1234
============================================================
```

### **Detailed Metrics**

```bash
# View last 10 entries
python script/view_training_metrics.py results/20251111_123456_model_grpo_metrics.jsonl

# View last 20 entries
python script/view_training_metrics.py results/20251111_123456_model_grpo_metrics.jsonl 20
```

---

## Interpreting Metrics

### **Good Training Signs ✅**

1. **Loss decreasing** (becoming more negative)
   - Indicates model is learning from positive rewards

2. **Entropy > 1.0**
   - Model maintains exploration
   - Not collapsed to deterministic policy

3. **Clip ratio < 1%**
   - Stable, conservative updates
   - Good for convergence

4. **Reward std > 0**
   - Diverse outcomes
   - Model exploring different strategies

5. **Balanced win rates**
   - Attacker and assessor both winning ~50%
   - Healthy competition

### **Warning Signs ⚠️**

1. **Loss increasing** (becoming more positive)
   - Model getting worse
   - May need lower learning rate

2. **Entropy < 0.5**
   - Policy collapsed
   - No exploration
   - May need higher temperature

3. **Clip ratio > 5%**
   - Large policy updates
   - May be unstable
   - Reduce learning rate

4. **Reward std = 0**
   - All outcomes identical
   - No exploration
   - Check temperature

5. **One agent dominates (>80% win rate)**
   - Imbalanced training
   - May need to adjust rewards

---

## Programmatic Access

### **Load Summary**

```python
import json

with open('results/experiment_metrics_summary.json', 'r') as f:
    summary = json.load(f)

# Access overall statistics
print(f"Total rounds: {summary['overall']['total_rounds']}")
print(f"Avg attacker loss: {summary['overall']['attacker']['avg_loss']}")
```

### **Load Detailed Metrics**

```python
import json

metrics = []
with open('results/experiment_metrics.jsonl', 'r') as f:
    for line in f:
        metrics.append(json.loads(line))

# Filter by phase
attacker_metrics = [m for m in metrics if m['phase'] == 'attacker']
assessor_metrics = [m for m in metrics if m['phase'] == 'assessor']

# Plot loss over time
import matplotlib.pyplot as plt

rounds = [m['round'] for m in attacker_metrics]
losses = [m['loss'] for m in attacker_metrics]

plt.plot(rounds, losses)
plt.xlabel('Round')
plt.ylabel('Loss')
plt.title('Attacker Loss Over Time')
plt.show()
```

---

## Integration with Existing Tools

### **WandB (Optional)**

To enable WandB logging, change in `main.py`:

```python
common_cfg = dict(
    ...
    report_to="wandb",  # Instead of "none"
)
```

Then set environment variable:
```bash
export WANDB_PROJECT="medical-selfplay"
python script/train_selfplay_advanced.py ...
```

### **TensorBoard (Optional)**

```python
common_cfg = dict(
    ...
    report_to="tensorboard",
    logging_dir="./logs",
)
```

Then view:
```bash
tensorboard --logdir ./logs
```

---

## Example: Tracking Progress

### **After Each Round**

The training will automatically print:

```
============================================================
ATTACKER TRAINING METRICS
============================================================
Loss: -0.1148
Gradient Norm: 0.5664
Learning Rate: 6.67e-06

Reward Statistics:
  Mean: -1.1875
  Std: 0.8413
  Frac Zero Std: 0.50

Generation Statistics:
  Mean Length: 97.1 tokens
  Min Length: 12.5 tokens
  Max Length: 357.5 tokens

Policy Statistics:
  Entropy: 1.1313
  Clip Ratio: 0.0046
============================================================
```

### **After Training**

```
============================================================
METRICS LOGGING COMPLETE
============================================================
Metrics log: results/20251111_123456_model_grpo_metrics.jsonl
Summary: results/20251111_123456_model_grpo_metrics_summary.json
============================================================
```

---

## Troubleshooting

### **No metrics logged**

Check that `metrics_logger` is initialized:
```python
metrics_logger = MetricsLogger(log_dir="results", experiment_name=experiment_name)
```

### **Missing metrics in summary**

Some metrics may be `null` if:
- Training didn't complete
- Metric not available for that phase
- Trainer state doesn't have log_history

### **Large JSONL files**

The `.jsonl` file grows with each phase. To reduce size:
- Decrease `logging_steps` in config
- Only log round summaries (comment out phase logging)

---

## Best Practices

1. **Always check metrics after training**
   ```bash
   python script/view_training_metrics.py results/*_summary.json
   ```

2. **Compare across runs**
   - Keep all `*_summary.json` files
   - Track improvements over time

3. **Monitor during training**
   - Watch for printed metrics summaries
   - Check for warning signs

4. **Archive successful runs**
   ```bash
   mkdir -p archive/successful_runs
   cp results/best_run_* archive/successful_runs/
   ```

---

## Summary

✅ **Automatic logging** - No manual tracking needed  
✅ **Comprehensive metrics** - Loss, rewards, entropy, etc.  
✅ **Easy viewing** - Simple Python script  
✅ **Programmatic access** - JSON/JSONL format  
✅ **Integration ready** - Works with WandB, TensorBoard  

Your training metrics are now tracked and ready for analysis!
