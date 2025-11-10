# Self-Play Training Package

This package contains the refactored modular implementation of the self-play training system for medical error detection.

## Structure

```
script/selfplay/
├── __init__.py          # Package exports
├── main.py              # Training orchestration (400 lines)
├── prompts.py           # Prompt generation (300 lines)
├── rewards.py           # Reward functions (400 lines)
├── utils.py             # Utilities (250 lines)
├── judge.py             # Judge evaluation (200 lines)
└── data.py              # Data loading (100 lines)
```

## Modules

### main.py
Main training loop with self-play rounds. Contains:
- `main()` - Entry point with argument parsing
- `get_device()` - Device selection
- `load_causal_lm()` - Model loading

### prompts.py
Prompt generation for both roles:
- `build_attacker_prompts()` - Generate attacker prompts with few-shot examples
- `make_assessor_prompts()` - Generate assessor prompts for classification

### rewards.py
Reward calculation functions:
- `create_attacker_reward_fn()` - Creates attacker reward function
- `create_assessor_reward_fn()` - Creates assessor reward function

### utils.py
Utility functions:
- `parse_response()` - Parse model outputs (pre-fill and post-fill formats)
- `patch_tokenizer_for_grpo()` - Fix GRPO tokenizer issues
- `check_attack_faithfulness()` - Validate attack quality
- `deduplicate_attacked_notes()` - Remove duplicate attacks
- `log_interaction()` - Log detailed interaction data

### judge.py
Judge model evaluation:
- `get_judge_assessment()` - Get ground-truth harm assessment
- `evaluate_thinking_quality()` - Evaluate reasoning quality
- `JudgeValidator` - Track judge classification distribution

### data.py
Data loading:
- `load_and_prepare_data()` - Load MEDEC data with clean→error transformation

## Usage

### Using the refactored version:
```bash
python script/train_selfplay_advanced.py \
    --model_id Qwen/Qwen2.5-0.5B-Instruct \
    --num_samples 16 \
    --rounds 3
```

### Using the original backup:
```bash
python script/train_selfplay_advanced_backup.py \
    --model_id Qwen/Qwen2.5-0.5B-Instruct \
    --num_samples 16 \
    --rounds 3
```

### Importing as a package:
```python
from selfplay import main, load_and_prepare_data
from selfplay.prompts import build_attacker_prompts
from selfplay.rewards import create_attacker_reward_fn

# Use the functions...
```

## Benefits of Refactoring

1. **Modularity**: Each module has a single, well-defined responsibility
2. **Maintainability**: Easy to find and modify specific functionality
3. **Testability**: Each module can be tested independently
4. **Reusability**: Functions can be imported and used in other scripts
5. **Readability**: Smaller files are easier to understand

## Migration Notes

- The original monolithic file (1000+ lines) is preserved as `train_selfplay_advanced_backup.py`
- The refactored version maintains 100% backwards compatibility
- All command-line arguments work identically
- Output files and logs are identical
- No behavioral changes - pure refactoring

## Development

To modify specific functionality:
- **Prompts**: Edit `prompts.py`
- **Rewards**: Edit `rewards.py`
- **Parsing**: Edit `utils.py`
- **Judge logic**: Edit `judge.py`
- **Data loading**: Edit `data.py`
- **Training loop**: Edit `main.py`
