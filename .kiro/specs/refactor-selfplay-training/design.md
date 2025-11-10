# Design Document

## Overview

This design describes the refactoring of `train_selfplay_advanced.py` into a modular package structure. The refactoring will split the monolithic 1000+ line file into 6 focused modules while maintaining 100% backwards compatibility.

## Architecture

### Package Structure

```
script/
├── selfplay/
│   ├── __init__.py          # Package exports
│   ├── prompts.py           # Prompt generation functions
│   ├── rewards.py           # Reward calculation functions
│   ├── utils.py             # Parsing, validation, helpers
│   ├── judge.py             # Judge model evaluation
│   ├── data.py              # Data loading and preparation
│   └── main.py              # Training orchestration
├── train_selfplay_advanced.py        # Updated to use new modules
└── train_selfplay_advanced_backup.py # Original file backup
```

### Module Responsibilities

**prompts.py** (~200 lines)
- `build_attacker_prompts()` - Generate attacker prompts with few-shot examples
- `make_assessor_prompts()` - Generate assessor prompts for classification
- Prompt template constants and few-shot examples

**rewards.py** (~300 lines)
- `attacker_reward_fn()` - Calculate attacker rewards (revision, refusal, harmfulness, format)
- `assessor_reward_fn()` - Calculate assessor rewards (refusal, harmfulness, format, thinking)
- Helper functions for reward component calculation

**utils.py** (~250 lines)
- `parse_response()` - Parse model outputs (pre-fill and post-fill formats)
- `check_attack_faithfulness()` - Validate attack quality
- `deduplicate_attacked_notes()` - Remove duplicate attacks
- `extract_original_from_attacker_prompt()` - Extract original note from prompt
- `extract_attacked_from_assessor_prompt()` - Extract attacked note from prompt
- `log_interaction()` - Log detailed interaction data
- `patch_tokenizer_for_grpo()` - Fix GRPO tokenizer issues

**judge.py** (~200 lines)
- `get_judge_assessment()` - Get ground-truth harm assessment
- `evaluate_thinking_quality()` - Evaluate reasoning quality
- `JudgeValidator` class - Track judge classification distribution

**data.py** (~150 lines)
- `load_and_prepare_data()` - Load MEDEC data with clean→error transformation
- Helper functions for data filtering and splitting

**main.py** (~400 lines)
- `main()` - Main training loop
- `get_device()` - Device selection
- `load_causal_lm()` - Model loading
- Training configuration and orchestration
- Self-play round management

## Components and Interfaces

### 1. Prompts Module

```python
# prompts.py

def build_attacker_prompts(
    ds: Dataset,
    few_shot_examples: Dataset,
    tokenizer,
    num_shots: int = 2
) -> Dataset:
    """Build attacker prompts using clean→error transformation.
    
    Args:
        ds: Dataset with seed notes and game types
        few_shot_examples: Examples showing clean→error transformations
        tokenizer: Tokenizer for chat template
        num_shots: Number of few-shot examples to include
        
    Returns:
        Dataset with pre-templated prompt strings
    """
    pass

def make_assessor_prompts(
    records: list,
    tokenizer
) -> Dataset:
    """Make assessor prompts for binary classification.
    
    Args:
        records: List of dicts with 'original', 'attacked', 'game_type'
        tokenizer: Tokenizer for chat template
        
    Returns:
        Dataset with pre-templated prompt strings
    """
    pass
```

### 2. Rewards Module

```python
# rewards.py

def create_attacker_reward_fn(
    policy_tok,
    judge_model,
    judge_tok,
    device,
    state: dict,
    log_path: str,
    attacked_notes_storage: list,
    diversity_stats: dict,
    judge_validator,
    assessor_snapshot: dict,
    R_GAME: float = 1.0,
    R_FORMAT: float = 1.0
):
    """Create attacker reward function with closure over dependencies.
    
    Returns a function compatible with GRPOTrainer.
    """
    def attacker_reward_fn(prompts, completions, **kwargs):
        # Implementation
        pass
    return attacker_reward_fn

def create_assessor_reward_fn(
    judge_model,
    judge_tok,
    device,
    state: dict,
    log_path: str,
    judge_validator,
    R_GAME: float = 1.0,
    R_FORMAT: float = 1.0
):
    """Create assessor reward function with closure over dependencies.
    
    Returns a function compatible with GRPOTrainer.
    """
    def assessor_reward_fn(prompts, completions, **kwargs):
        # Implementation
        pass
    return assessor_reward_fn
```

### 3. Utils Module

```python
# utils.py

def parse_response(text: str) -> tuple[str, str]:
    """Parse response supporting both pre-fill and post-fill CoT formats.
    
    Returns:
        (thought, output) tuple
    """
    pass

def check_attack_faithfulness(
    attacked: str,
    original: str,
    min_similarity: float = 0.6
) -> tuple[bool, str]:
    """Check if attacked note is a subtle modification.
    
    Returns:
        (is_valid, reason) tuple
    """
    pass

def patch_tokenizer_for_grpo(tokenizer):
    """Monkey-patch tokenizer to force add_special_tokens=True.
    
    Critical fix for Qwen models with GRPO.
    """
    pass

def log_interaction(
    round_num: int,
    phase: str,
    original: str,
    attacked: str,
    attacker_response: dict,
    assessor_response: dict,
    judgments: dict,
    rewards: dict,
    log_path: str
):
    """Log detailed interaction data for analysis."""
    pass
```

### 4. Judge Module

```python
# judge.py

def get_judge_assessment(
    original: str,
    attacked: str,
    assessor_label: str,
    game_type: str,
    judge_model,
    judge_tok,
    device
) -> dict:
    """Use judge model to get ground-truth harm assessment.
    
    Returns:
        Dict with actual_harm, assessor_was_correct, etc.
    """
    pass

def evaluate_thinking_quality(
    thinking: str,
    note: str,
    classification: str,
    judge_model,
    judge_tok,
    device
) -> dict:
    """Evaluate quality of assessor's thinking.
    
    Returns:
        Dict with thinking_quality (+1 or -1) and reasoning
    """
    pass

class JudgeValidator:
    """Track judge classification distribution and warn about skewed results."""
    
    def __init__(self):
        pass
    
    def add_classification(self, classification: str):
        pass
    
    def check_distribution(self) -> dict:
        pass
    
    def get_stats(self) -> dict:
        pass
```

### 5. Data Module

```python
# data.py

def load_and_prepare_data(num_samples: int) -> tuple[Dataset, Dataset]:
    """Load MEDEC data with clean→error transformation approach.
    
    Args:
        num_samples: Number of samples to prepare
        
    Returns:
        (ds_seeds, ds_few_shot) tuple
    """
    pass
```

### 6. Main Module

```python
# main.py

def get_device() -> torch.device:
    """Get best available device for PyTorch."""
    pass

def load_causal_lm(model_id: str, device: torch.device):
    """Load causal language model and tokenizer.
    
    Returns:
        (model, tokenizer) tuple
    """
    pass

def main():
    """Main training loop with self-play rounds."""
    pass

if __name__ == "__main__":
    main()
```

## Data Models

### Shared State Objects

```python
# State dictionary passed between functions
state = {
    "round": int,           # Current round number
    "total_steps": int      # Total training steps
}

# Diversity statistics dictionary
diversity_stats = {
    "harmful_games": int,
    "safe_games": int,
    "harmful_safe": int,
    "harmful_harmful": int,
    "safe_safe": int,
    "safe_harmful": int,
    "harmful_faithful": int,
    "harmful_unfaithful": int,
    "safe_faithful": int,
    "safe_unfaithful": int
}

# Assessor snapshot dictionary
assessor_snapshot = {
    "model": Optional[torch.nn.Module]
}

# Attacked notes storage (list of dicts)
attacked_notes_from_training = [
    {
        "original": str,
        "attacked": str,
        "game_type": str
    }
]
```

## Error Handling

### Module Import Errors
- Each module will have proper error handling for missing dependencies
- The main module will catch import errors and provide helpful messages

### Validation Errors
- `utils.py` will handle parsing failures gracefully
- `judge.py` will handle judge model failures with fallback logic
- `rewards.py` will handle invalid completions with worst-case scores

## Testing Strategy

### Unit Testing
- Test each module independently with mock dependencies
- Test parsing functions with various input formats
- Test reward calculations with known inputs

### Integration Testing
- Test full training loop with small dataset (2 samples, 1 round)
- Verify output files match original implementation
- Compare reward values between original and refactored versions

### Backwards Compatibility Testing
- Run original wrapper script and verify it works
- Compare outputs between wrapper and direct module usage
- Verify command-line arguments work identically

## Migration Strategy

### Phase 1: Create Module Structure
1. Create `script/selfplay/` directory
2. Create empty module files with docstrings
3. Create `__init__.py` with package exports

### Phase 2: Extract Functions
1. Move functions to appropriate modules
2. Update imports and dependencies
3. Test each module independently

### Phase 3: Update Main Script
1. Refactor `main()` to use new modules
2. Test full training loop
3. Verify outputs match original

### Phase 4: Update Main Script
1. Create backup: `train_selfplay_advanced_backup.py` (exact copy of original)
2. Update `train_selfplay_advanced.py` to import from modules
3. Test backwards compatibility
4. Document migration in comments

## Dependencies Between Modules

```
main.py
├── imports: prompts, rewards, utils, judge, data
├── uses: all modules

rewards.py
├── imports: utils, judge, prompts
├── uses: parse_response, get_judge_assessment, evaluate_thinking_quality

prompts.py
├── imports: (none - standalone)
├── uses: tokenizer (passed as argument)

utils.py
├── imports: (none - standalone)
├── uses: standard library only

judge.py
├── imports: utils
├── uses: parse_response (for judge output parsing)

data.py
├── imports: (none - standalone)
├── uses: pandas, datasets
```

## Performance Considerations

- No performance impact expected - pure refactoring
- All functions maintain same signatures and behavior
- No additional overhead from module imports
- Memory usage remains identical

## Security Considerations

- No security changes - maintains existing behavior
- File paths remain the same
- No new external dependencies
- Logging behavior unchanged
