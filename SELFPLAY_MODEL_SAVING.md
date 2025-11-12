# Selfplay Model Saving

## Overview

The selfplay training script now automatically saves the final trained model after all rounds are complete.

## What Gets Saved

After training completes, the following are saved:

1. **Model weights** - The trained PyTorch model
2. **Tokenizer** - The tokenizer configuration
3. **Training info** - JSON file with training parameters

## Default Location

By default, models are saved to:
```
trainer_output/<timestamp>_<model_name>_grpo_final/
```

Example:
```
trainer_output/20251112_143000_qwen3-4b-medical-selfplay-sft_grpo_final/
```

## Custom Output Directory

You can specify a custom output directory in two ways:

### Option 1: Edit the shell script

In `run_selfplay_training.sh`, set the `OUTPUT_DIR` variable:

```bash
# Optional: Set custom output directory for the final model
OUTPUT_DIR="trainer_output/my_custom_model"
```

### Option 2: Pass directly to Python script

```bash
python3 script/train_selfplay_advanced.py \
    --model_id trainer_output/qwen3-4b-medical-selfplay-sft \
    --output_dir trainer_output/my_custom_model \
    [other args...]
```

## Loading the Saved Model

To use the trained model later:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "trainer_output/20251112_143000_qwen3-4b-medical-selfplay-sft_grpo_final"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

## Training Info

The `training_info.json` file contains:
- Original model ID
- Judge model ID
- Training hyperparameters (learning rate, rounds, etc.)
- Timestamp and experiment name

This helps you track which settings were used for each trained model.

## Disk Space

Each saved model will take approximately:
- **Qwen 4B**: ~8GB
- **Qwen 8B**: ~16GB

Make sure you have sufficient disk space before training.

## Notes

- The model is saved **after all rounds complete**, not after each round
- If training is interrupted, the model will not be saved
- The script uses `save_strategy="no"` during training to avoid saving intermediate checkpoints (saves disk space)
