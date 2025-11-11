# Results Directory

This directory contains all training outputs from self-play GRPO training.

## 📁 File Structure

Each training run generates **3 files**:

```
results/
├── {timestamp}_{model}_grpo_metrics.jsonl          # Detailed metrics (JSONL)
├── {timestamp}_{model}_grpo_metrics_summary.json   # Summary (JSON)
└── {timestamp}_{model}_grpo_interactions.jsonl     # Interaction logs (JSONL)
```

**Example:**
```
results/
├── 20251111_123456_qwen3_4b_grpo_metrics.jsonl
├── 20251111_123456_qwen3_4b_grpo_metrics_summary.json
└── 20251111_123456_qwen3_4b_grpo_interactions.jsonl
```

---

## 📊 File Types

### 1. Metrics Log (`.jsonl`)
- Training metrics per phase
- Loss, rewards, gradients, entropy
- One JSON object per line

### 2. Metrics Summary (`.json`)
- Human-readable summary
- Round-by-round overview
- Overall statistics

### 3. Interaction Logs (`.jsonl`)
- Detailed game interactions
- Notes, responses, judgments
- Reward breakdowns

---

## 🔍 Quick View

```bash
# View latest summary
python script/view_training_metrics.py $(ls -t *_summary.json | head -1)

# View detailed metrics
python script/view_training_metrics.py $(ls -t *_metrics.jsonl | head -1)

# Count interactions
wc -l *_interactions.jsonl
```

---

## 📚 Documentation

See `WHERE_ARE_MY_METRICS.md` in the root directory for complete documentation.

---

## 🗑️ Cleanup

To remove old runs:

```bash
# Remove runs older than 7 days
find . -name "*_grpo_*" -mtime +7 -delete

# Archive before cleanup
tar -czf archive_$(date +%Y%m%d).tar.gz *.jsonl *.json
```

---

**Note:** This directory is automatically created by the training script.
