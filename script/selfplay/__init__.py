"""
Self-play training package for medical error detection.

This package contains modular components for training attacker and assessor models
using GRPO (Group Relative Policy Optimization) in a self-play framework.

Modules:
    - prompts: Prompt generation for attacker and assessor roles
    - rewards: Reward calculation functions for both roles
    - utils: Parsing, validation, and helper utilities
    - judge: Judge model evaluation and validation
    - data: Data loading and preparation from MEDEC dataset
    - main: Training orchestration and main entry point
"""

__version__ = "1.0.0"

# Import main entry point
from .main import main, get_device, load_causal_lm

# Import data loading
from .data import load_and_prepare_data

# Import prompt generation
from .prompts import build_attacker_prompts, make_assessor_prompts

# Import reward functions
from .rewards import create_attacker_reward_fn, create_assessor_reward_fn

# Import utilities
from .utils import (
    parse_response,
    patch_tokenizer_for_grpo,
    check_attack_faithfulness,
    deduplicate_attacked_notes,
    log_interaction,
    extract_original_from_attacker_prompt,
    extract_attacked_from_assessor_prompt,
)

# Import judge functions
from .judge import (
    get_judge_assessment,
    evaluate_thinking_quality,
    JudgeValidator,
)

__all__ = [
    # Main entry point
    "main",
    "get_device",
    "load_causal_lm",
    # Data
    "load_and_prepare_data",
    # Prompts
    "build_attacker_prompts",
    "make_assessor_prompts",
    # Rewards
    "create_attacker_reward_fn",
    "create_assessor_reward_fn",
    # Utils
    "parse_response",
    "patch_tokenizer_for_grpo",
    "check_attack_faithfulness",
    "deduplicate_attacked_notes",
    "log_interaction",
    "extract_original_from_attacker_prompt",
    "extract_attacked_from_assessor_prompt",
    # Judge
    "get_judge_assessment",
    "evaluate_thinking_quality",
    "JudgeValidator",
]
