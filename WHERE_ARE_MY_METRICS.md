# Where Are My Training Metrics?

## 📁 File Locations

All training outputs are stored in the **`results/`** directory in your workspace root.

### **Directory Structure**

```
medical_reward/
├── results/
│   ├── 20251111_123456_model_grpo_metrics.jsonl          # ← Detailed metrics
│   ├── 20251111_123456_model_grpo_metrics_summary.json   # ← Human-readable summary
│   └── 20251111_123456_model_grpo_interactions.jsonl     # ← Interaction logs
```

---

## 📊 Three Types of Files

### **1. Metrics Log (JSONL)**

**File:** `{timestamp}_{model}_grpo_metrics.jsonl`

**Contains:**
- Training metrics for each phase (attacker/assessor)
- Loss, rewards, gradients, learning rate
- Generation statistics (length, clipping)
- Policy metrics (entropy, clip ratios)
- Round summaries (diversity, judge stats)

**Format:** One JSON object per line (JSONL)

**Example:**
```json
{"round": 1, "phase": "attacker", "loss": -0.1148, "reward": -1.1875, ...}
{"round": 1, "phase": "assessor", "loss": -0.0892, "reward": 1.2341, ...}
{"round": 1, "phase": "round_summary", "diversity": {...}, "judge": {...}}
```

**View with:**
```bash
python script/view_training_metrics.py results/20251111_123456_model_grpo_metrics.jsonl
```

---

### **2. Metrics Summary (JSON)**

**File:** `{timestamp}_{model}_grpo_metrics_summary.json`

**Contains:**
- Experiment metadata (name, start/end time)
- Round-by-round summary
- Overall statistics (averages across rounds)

**Format:** Single JSON object (human-readable)

**Example:**
```json
{
  "experiment_name": "20251111_123456_model_grpo",
  "start_time": "2025-11-11T12:34:56",
  "end_time": "2025-11-11T14:56:78",
  "rounds": [
    {
      "round": 1,
      "attacker": {"loss": -0.1148, "reward": -1.1875, ...},
      "assessor": {"loss": -0.0892, "reward": 1.2341, ...}
    }
  ],
  "overall": {
    "total_rounds": 4,
    "attacker": {"avg_loss": -0.1234, "avg_reward": -1.0567},
    "assessor": {"avg_loss": -0.0987, "avg_reward": 1.1234}
  }
}
```

**View with:**
```bash
python script/view_training_metrics.py results/20251111_123456_model_grpo_metrics_summary.json
```

Or just open in any text editor!

---

### **3. Interaction Logs (JSONL)**

**File:** `{timestamp}_{model}_grpo_interactions.jsonl`

**Contains:**
- Detailed interaction data for each game
- Original note, attacked note, assessor response
- Judge assessment and reasoning
- Reward breakdown
- Thinking/reasoning from both agents

**Format:** One JSON object per interaction

**Example:**
```json
{
  "round": 1,
  "phase": "attacker_training",
  "original_note": "Patient prescribed...",
  "attacked_note": "Patient prescribed...",
  "attacker_response": {"thought": "...", "attacked_note": "..."},
  "assessor_response": {"thought": "...", "label": "Safe"},
  "judgments": {"actual_harm": "Safe", "assessor_was_correct": true},
  "rewards": {"RA_res_harm": -1.0, "RA_res_refusal": -1.0, ...}
}
```

**View with:**
```bash
# Count interactions
wc -l results/20251111_123456_model_grpo_interactions.jsonl

# View last 5 interactions
tail -n 5 results/20251111_123456_model_grpo_interactions.jsonl | jq
```

---

## 🔍 Quick Access Commands

### **List All Your Training Runs**

```bash
ls -lh results/
```

### **Find Latest Metrics**

```bash
# Latest summary
ls -t results/*_summary.json | head -1

# Latest detailed metrics
ls -t results/*_metrics.jsonl | head -1
```

### **View Latest Training**

```bash
# View summary
python script/view_training_metrics.py $(ls -t results/*_summary.json | head -1)

# View detailed (last 10 entries)
python script/view_training_metrics.py $(ls -t results/*_metrics.jsonl | head -1)
```

### **Compare Multiple Runs**

```bash
# List all summaries
ls results/*_summary.json

# View specific run
python script/view_training_metrics.py results/20251111_123456_model_grpo_metrics_summary.json
```

---

## 📈 What Gets Logged When

### **During Training**

**After Attacker Training:**
```
============================================================
ATTACKER TRAINING METRICS
============================================================
Loss: -0.1148
Reward: -1.1875 ± 0.8413
...
============================================================

📊 Logged attacker metrics for round 1
```

**After Assessor Training:**
```
============================================================
ASSESSOR TRAINING METRICS
============================================================
Loss: -0.0892
Reward: 1.2341 ± 0.7654
...
============================================================

📊 Logged assessor metrics for round 1
```

**After Each Round:**
```
📊 Logged round 1 summary
```

### **After Training Completes**

```
============================================================
METRICS LOGGING COMPLETE
============================================================
Metrics log: results/20251111_123456_model_grpo_metrics.jsonl
Summary: results/20251111_123456_model_grpo_metrics_summary.json
============================================================

📄 Interaction log written to results/20251111_123456_model_grpo_interactions.jsonl
📊 Metrics log written to results/20251111_123456_model_grpo_metrics.jsonl
📊 Metrics summary written to results/20251111_123456_model_grpo_metrics_summary.json
```

---

## 💾 File Sizes (Approximate)

| File | Size (32 samples, 4 rounds) | Size (128 samples, 8 rounds) |
|------|----------------------------|------------------------------|
| **metrics.jsonl** | ~50 KB | ~200 KB |
| **metrics_summary.json** | ~5 KB | ~20 KB |
| **interactions.jsonl** | ~5 MB | ~20 MB |

**Note:** Interaction logs are much larger because they contain full note text and reasoning.

---

## 🗂️ Organizing Your Results

### **Archive Successful Runs**

```bash
mkdir -p archive/successful_runs
cp results/best_run_* archive/successful_runs/
```

### **Clean Up Failed Runs**

```bash
# Remove incomplete runs (no summary file)
for f in results/*_metrics.jsonl; do
    summary="${f/_metrics.jsonl/_metrics_summary.json}"
    if [ ! -f "$summary" ]; then
        echo "Removing incomplete run: $f"
        rm "$f"
    fi
done
```

### **Backup Before New Training**

```bash
# Backup all results
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/
```

---

## 🔧 Programmatic Access

### **Python**

```python
import json
from pathlib import Path

# Find latest summary
summaries = sorted(Path("results").glob("*_summary.json"))
latest = summaries[-1]

# Load summary
with open(latest) as f:
    summary = json.load(f)

print(f"Experiment: {summary['experiment_name']}")
print(f"Total rounds: {summary['overall']['total_rounds']}")
print(f"Avg attacker loss: {summary['overall']['attacker']['avg_loss']}")
```

### **Bash**

```bash
# Extract specific metric from latest run
latest=$(ls -t results/*_summary.json | head -1)
jq '.overall.attacker.avg_loss' "$latest"
```

---

## 📊 Visualization (Optional)

### **Plot Loss Over Time**

```python
import json
import matplotlib.pyplot as plt

# Load metrics
metrics = []
with open('results/20251111_123456_model_grpo_metrics.jsonl') as f:
    for line in f:
        metrics.append(json.loads(line))

# Filter attacker metrics
attacker = [m for m in metrics if m['phase'] == 'attacker']

# Plot
rounds = [m['round'] for m in attacker]
losses = [m['loss'] for m in attacker]

plt.plot(rounds, losses, marker='o')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.title('Attacker Loss Over Time')
plt.grid(True)
plt.savefig('attacker_loss.png')
```

---

## ❓ FAQ

**Q: Where are the model checkpoints?**
A: Currently disabled to save disk space. Set `save_strategy="steps"` in `common_cfg` to enable.

**Q: Can I delete old results?**
A: Yes, but archive important runs first. The `results/` directory can grow large.

**Q: How do I track experiments?**
A: The timestamp in the filename is unique per run. You can also use WandB (see `METRICS_LOGGING_GUIDE.md`).

**Q: What if training crashes?**
A: Metrics are written incrementally. You'll have partial logs up to the crash point.

**Q: Can I resume from metrics?**
A: Metrics are for analysis only. To resume training, you need model checkpoints (currently disabled).

---

## 🎯 Quick Reference

| What | Where | How to View |
|------|-------|-------------|
| **Detailed metrics** | `results/*_metrics.jsonl` | `python script/view_training_metrics.py <file>` |
| **Summary** | `results/*_summary.json` | `python script/view_training_metrics.py <file>` or text editor |
| **Interactions** | `results/*_interactions.jsonl` | `tail -n 10 <file> \| jq` |
| **Latest run** | `ls -t results/*_summary.json \| head -1` | See above |

---

## ✅ Summary

Your training metrics are stored in **`results/`** directory with three files per run:

1. **`*_metrics.jsonl`** - Detailed training metrics (for analysis)
2. **`*_metrics_summary.json`** - Human-readable summary (quick view)
3. **`*_interactions.jsonl`** - Full interaction logs (debugging)

Use `python script/view_training_metrics.py <file>` to view them easily!
