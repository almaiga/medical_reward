#!/usr/bin/env python3
"""
Self-play training script for medical error detection (Refactored).

This script has been refactored into modular components in the `selfplay` package.
The original monolithic implementation is preserved in `train_selfplay_advanced_backup.py`.

New structure:
    - script/selfplay/main.py - Main training loop
    - script/selfplay/prompts.py - Prompt generation
    - script/selfplay/rewards.py - Reward functions
    - script/selfplay/utils.py - Utilities
    - script/selfplay/judge.py - Judge evaluation
    - script/selfplay/data.py - Data loading

Usage:
    python script/train_selfplay_advanced.py --model_id <model> [options]

For the original implementation, see:
    python script/train_selfplay_advanced_backup.py --model_id <model> [options]
"""

from selfplay.main import main

if __name__ == "__main__":
    main()
