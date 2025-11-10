# Requirements Document

## Introduction

This spec covers the refactoring of the `train_selfplay_advanced.py` script into a modular structure. The current file is over 1000 lines and difficult to maintain. The goal is to split it into logical modules while maintaining all existing functionality.

## Glossary

- **Training Script**: The main executable script that orchestrates the self-play training process
- **Prompt Module**: Contains all prompt generation functions for attacker and assessor roles
- **Reward Module**: Contains reward calculation functions for both attacker and assessor
- **Utils Module**: Contains utility functions for parsing, validation, and data processing
- **Judge Module**: Contains judge model evaluation and validation logic
- **Data Module**: Contains data loading and preparation functions

## Requirements

### Requirement 1

**User Story:** As a developer, I want the code organized into separate modules, so that I can easily find and modify specific functionality without navigating a large monolithic file.

#### Acceptance Criteria

1. WHEN the refactoring is complete, THE System SHALL have separate Python modules for prompts, rewards, utilities, judge logic, and data processing
2. WHEN a developer needs to modify prompt templates, THE System SHALL provide all prompts in a dedicated `prompts.py` file
3. WHEN a developer needs to modify reward functions, THE System SHALL provide all reward logic in a dedicated `rewards.py` file
4. THE System SHALL maintain all existing functionality without behavioral changes
5. THE System SHALL place all new modules in the `script/selfplay/` directory

### Requirement 2

**User Story:** As a developer, I want clear separation of concerns, so that each module has a single, well-defined responsibility.

#### Acceptance Criteria

1. THE System SHALL create a `prompts.py` module containing only prompt generation functions
2. THE System SHALL create a `rewards.py` module containing only reward calculation functions
3. THE System SHALL create a `utils.py` module containing parsing, validation, and helper functions
4. THE System SHALL create a `judge.py` module containing judge model evaluation logic
5. THE System SHALL create a `data.py` module containing data loading and preparation functions
6. THE System SHALL create a `main.py` module containing the training orchestration logic

### Requirement 3

**User Story:** As a developer, I want the refactored code to be backwards compatible, so that existing scripts and workflows continue to work without modification.

#### Acceptance Criteria

1. THE System SHALL maintain the same command-line interface and arguments
2. THE System SHALL produce identical output files and logs
3. THE System SHALL maintain the same training behavior and results
4. THE System SHALL keep the original `train_selfplay_advanced.py` as a thin wrapper that imports from the new modules
5. WHEN the refactored code is run, THE System SHALL produce the same results as the original implementation

### Requirement 4

**User Story:** As a developer, I want clear imports and dependencies, so that I can understand how modules interact with each other.

#### Acceptance Criteria

1. WHEN a module depends on another module, THE System SHALL use explicit imports at the top of the file
2. THE System SHALL minimize circular dependencies between modules
3. THE System SHALL document module dependencies in docstrings
4. THE System SHALL use relative imports within the `selfplay` package
5. THE System SHALL create an `__init__.py` file that exports key functions and classes
