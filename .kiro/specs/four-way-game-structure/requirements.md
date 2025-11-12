# Requirements: Four-Way Game Structure for Self-Play Training

## Introduction

The current self-play training system in `script/selfplay/` always passes prompts through the attacker model, creating only adversarial examples. This makes the attacker's task uniformly difficult. According to the paper, we need to implement a 4-way game structure where the RL prompt composition is 25:25:25:25 to balance task difficulty:

**Current Approach (2-way):**
- 50% harmful game: Attacker gets clean note + error example → must inject similar error (HARD)
- 50% safe game: Attacker gets clean note → must keep it safe (EASY)

**Target Approach (4-way):**
- 25% vanilla_harmful: Attacker receives error note → instructed to copy it exactly as-is (EASY)
- 25% adversarial_harmful: Attacker receives error note → instructed to modify/worsen the error (HARD)
- 25% vanilla_benign: Attacker receives clean note → instructed to copy it exactly as-is (EASY)
- 25% adversarial_benign: Attacker receives clean note → instructed to inject new error (HARD)

This balances the attacker's task difficulty (50% easy vanilla copy-as-is, 50% hard adversarial modification) and ensures the defender sees both unmodified and adversarially modified notes.

**MEDEC Dataset Structure:**
- Rows with Error Flag=1 contain both "Text" (error note) and "Corrected Text" (clean note)
- We'll use "Text" for harmful prompts and "Corrected Text" for benign prompts
- Vanilla prompts bypass the attacker; adversarial prompts go through the attacker

## Glossary

- **Training System**: The self-play training pipeline that generates attacker and assessor interactions
- **Vanilla Prompt**: A prompt used directly without modification by the attacker
- **Adversarial Prompt**: A prompt that serves as a seed for the attacker to revise into a more challenging version
- **Harmful Prompt**: A medical note containing errors that could lead to patient harm
- **Benign Prompt**: A medical note that is safe and contains no errors
- **Attacker**: The model role that attempts to inject errors into medical notes
- **Defender/Assessor**: The model role that attempts to detect errors in medical notes

## Requirements

### Requirement 1: Four-Way Prompt Distribution

**User Story:** As a researcher, I want the training data to follow a 25:25:25:25 distribution across vanilla harmful, adversarial harmful, vanilla benign, and adversarial benign prompts, so that the defender learns to handle both vanilla and adversarial attacks across both categories.

#### Acceptance Criteria

1. WHEN preparing a training batch, THE Training System SHALL allocate exactly 25% of prompts as vanilla harmful
2. WHEN preparing a training batch, THE Training System SHALL allocate exactly 25% of prompts as adversarial harmful (seed prompts for attacker)
3. WHEN preparing a training batch, THE Training System SHALL allocate exactly 25% of prompts as vanilla benign
4. WHEN preparing a training batch, THE Training System SHALL allocate exactly 25% of prompts as adversarial benign (seed prompts for attacker)
5. THE Training System SHALL maintain this distribution across all training rounds

### Requirement 2: Vanilla Game (Copy-As-Is Instructions)

**User Story:** As a training system, I want vanilla games to instruct the attacker to copy notes exactly as-is, so that the attacker has easy tasks balanced with hard adversarial tasks.

#### Acceptance Criteria

1. WHEN a prompt is marked as vanilla_harmful, THE Training System SHALL give the attacker the error note from MEDEC "Text" column with instructions to "copy it exactly as-is without any modifications"
2. WHEN a prompt is marked as vanilla_benign, THE Training System SHALL give the attacker the clean note from MEDEC "Corrected Text" column with instructions to "copy it exactly as-is without any modifications"
3. WHEN a vanilla game is used, THE attacker prompt SHALL explicitly state "Do NOT introduce any medical errors" and "Copy the note exactly"
4. WHEN a vanilla game is used, THE attacker SHALL still generate output but the task is trivial (exact copy)
5. THE Training System SHALL track which prompts are vanilla vs adversarial in the interaction logs with the game_category field showing "vanilla_harmful" or "vanilla_benign"

### Requirement 3: Adversarial Prompt Seeding

**User Story:** As a researcher, I want adversarial prompts to serve as seeds for the attacker to revise, so that the defender faces more challenging and diverse attacks.

#### Acceptance Criteria

1. WHEN a prompt is marked as adversarial_harmful, THE Training System SHALL pass the error note from MEDEC "Text" column to the attacker with instructions to modify or worsen the existing error
2. WHEN a prompt is marked as adversarial_benign, THE Training System SHALL pass the clean note from MEDEC "Corrected Text" column to the attacker with instructions to inject a new error
3. THE Training System SHALL use different attacker prompts for adversarial_harmful vs adversarial_benign seeds
4. THE Training System SHALL track the original seed note and the attacker's revision in logs
5. THE Training System SHALL pass the attacker's revision to the assessor for evaluation

### Requirement 4: Game Category Tracking and Reward Alignment

**User Story:** As a developer, I want clear tracking of the 4 game categories and appropriate reward signals, so that both attacker and assessor learn from balanced examples.

#### Acceptance Criteria

1. WHEN a vanilla_harmful prompt is used, THE Training System SHALL mark it with game_category="vanilla_harmful" and the assessor SHALL be rewarded for detecting the error
2. WHEN an adversarial_harmful prompt is used, THE Training System SHALL mark it with game_category="adversarial_harmful" and both attacker and assessor SHALL receive rewards based on the outcome
3. WHEN a vanilla_benign prompt is used, THE Training System SHALL mark it with game_category="vanilla_benign" and the assessor SHALL be rewarded for confirming safety
4. WHEN an adversarial_benign prompt is used, THE Training System SHALL mark it with game_category="adversarial_benign" and both attacker and assessor SHALL receive rewards based on the outcome
5. THE Training System SHALL maintain the existing reward structure where attacker and assessor have opposing objectives

### Requirement 5: Balanced Dataset Preparation from MEDEC

**User Story:** As a researcher, I want the MEDEC dataset to be split into 4 equal groups (vanilla_harmful, adversarial_harmful, vanilla_benign, adversarial_benign), so that training is balanced across all categories.

#### Acceptance Criteria

1. WHEN loading the MEDEC dataset, THE Training System SHALL use rows with Error Flag=1 that contain both "Text" (error note) and "Corrected Text" (clean note)
2. WHEN preparing the dataset, THE Training System SHALL create 4 equal-sized groups by splitting available rows into quarters
3. WHEN assigning game categories, THE Training System SHALL randomly assign 25% to vanilla_harmful, 25% to adversarial_harmful, 25% to vanilla_benign, and 25% to adversarial_benign
4. THE Training System SHALL shuffle rows before applying the 4-way split to ensure random distribution
5. THE Training System SHALL log the dataset composition showing count for each of the 4 categories at initialization

### Requirement 8: Data Structure for 4-Way Games

**User Story:** As a developer, I want each dataset row to contain all necessary information for its game category, so that the training loop can handle all 4 game types correctly.

#### Acceptance Criteria

1. WHEN creating the dataset in `load_and_prepare_data()`, THE Training System SHALL include the following columns: game_category, error_note, clean_note, error_type
2. WHEN game_category is vanilla_harmful, THE Training System SHALL set error_note from "Text" column and this note SHALL be passed directly to assessor
3. WHEN game_category is adversarial_harmful, THE Training System SHALL set error_note from "Text" column and this note SHALL be passed to attacker as seed
4. WHEN game_category is vanilla_benign, THE Training System SHALL set clean_note from "Corrected Text" column and this note SHALL be passed directly to assessor
5. WHEN game_category is adversarial_benign, THE Training System SHALL set clean_note from "Corrected Text" column and this note SHALL be passed to attacker as seed

### Requirement 9: Judge Context Enhancement

**User Story:** As a judge model, I want to receive game category context when evaluating notes, so that I can make more accurate ground-truth assessments based on what type of game was played.

#### Acceptance Criteria

1. WHEN calling the judge in `get_judge_assessment()`, THE Training System SHALL include the game_category in the judge prompt
2. WHEN game_category is vanilla_harmful, THE judge prompt SHALL indicate "This is an unmodified error note from the dataset - assess the original error"
3. WHEN game_category is adversarial_harmful, THE judge prompt SHALL indicate "An attacker modified an existing error note - assess the final note for harm"
4. WHEN game_category is vanilla_benign, THE judge prompt SHALL indicate "This is an unmodified clean note from the dataset - should be Safe"
5. WHEN game_category is adversarial_benign, THE judge prompt SHALL indicate "An attacker tried to inject errors into a clean note - assess if the attack succeeded"
6. THE judge SHALL still make independent assessments but use the context to understand what to expect

### Requirement 6: Interaction Logging and Tracking

**User Story:** As a researcher, I want detailed logs of which game type each interaction belongs to, so that I can analyze performance across the 4-way structure.

#### Acceptance Criteria

1. WHEN logging an interaction, THE Training System SHALL record the game category (vanilla_harmful, adversarial_harmful, vanilla_benign, adversarial_benign)
2. WHEN logging an interaction, THE Training System SHALL record whether the attacker was involved (vanilla=no, adversarial=yes)
3. WHEN training completes, THE Training System SHALL report statistics for each of the 4 categories
4. THE Training System SHALL track assessor accuracy separately for each category
5. THE Training System SHALL track attacker success rate separately for harmful vs benign seeds

### Requirement 7: Attacker Prompt Differentiation for Adversarial Games

**User Story:** As an attacker model, I want different instructions for adversarial_harmful vs adversarial_benign prompts, so that I understand whether to modify existing errors or inject new ones.

#### Acceptance Criteria

1. WHEN the attacker receives an adversarial_harmful seed (error note from "Text"), THE Training System SHALL instruct it to "modify or worsen the existing medical error while keeping the note structure similar"
2. WHEN the attacker receives an adversarial_benign seed (clean note from "Corrected Text"), THE Training System SHALL instruct it to "inject ONE subtle but medically significant error into the clean note"
3. THE Training System SHALL provide few-shot examples showing error→modified_error transformations for adversarial_harmful games
4. THE Training System SHALL provide few-shot examples showing clean→error transformations for adversarial_benign games
5. THE Training System SHALL use distinct prompt templates in `build_attacker_prompts()` for adversarial_harmful vs adversarial_benign categories

## Success Metrics

- 4-way distribution: Exactly 25% in each category (vanilla_harmful, adversarial_harmful, vanilla_benign, adversarial_benign)
- Vanilla games: Attacker instructed to copy exactly (50% of total prompts are easy copy tasks)
- Adversarial games: Attacker instructed to modify (50% of total prompts are hard modification tasks)
- Assessor accuracy: Tracked separately for all 4 categories
- Attacker task balance: 50% easy (vanilla copy-as-is), 50% hard (adversarial modification)
- Logging completeness: 100% of interactions tagged with game_category field
- Dataset validation: All rows have both error_note and clean_note populated from MEDEC

## Out of Scope

- Changing the reward calculation formulas
- Modifying the judge model or its prompts
- Adding new game types beyond the 4-way structure
- Changing the base model or training hyperparameters
- Implementing the SFT self-distillation process (separate feature)
