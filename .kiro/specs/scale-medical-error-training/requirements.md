# Requirements Document: Scale Medical Error Detection Training Pipeline

## Introduction

This spec defines requirements for scaling the medical error detection training pipeline from the current proof-of-concept (1,592 examples from ~530 MEDEC notes) to a production-ready system using the full MEDEC training set (2,189 notes: 1,219 with errors, 970 clean).

The system trains a Qwen-4B model through a three-stage pipeline:
1. **Educational SFT**: GPT-5 generates reasoning traces teaching medical error analysis
2. **Game Format Adaptation**: Bridges educational format to GRPO game prompts  
3. **Self-Play GRPO**: Attacker/assessor adversarial training with MedGemma-4B judge

**Current Status:**
- ✅ Generated 1,592 educational SFT examples (796 attacker + 796 assessor) from ~530 notes
- ✅ Proven 3-stage pipeline works (educational SFT → adaptation → GRPO)
- ✅ GPT-5 reasoning trace generation script (`script/generate_sft_data.py`)
- ⏳ Need to scale to full 2,189 MEDEC notes → ~6,500 total training examples

**Key Insight from Self-RedTeam Paper:**
- They use 15k examples per dataset (30k total)
- We have 2,189 notes × 3 examples each = ~6,500 examples (comparable scale)
- 50/50 harmful/safe split prevents over-refusal

## Glossary

- **SFT (Supervised Fine-Tuning)**: Initial training phase where model learns medical reasoning and CoT format
- **GRPO (Group Relative Policy Optimization)**: RL training method used for self-play
- **Attacker**: Model role that introduces or keeps medical errors in notes
- **Assessor**: Model role that classifies notes as Safe/Harmful
- **Judge Model**: MedGemma-4B used to evaluate attacker/assessor outputs and assign rewards
- **CoT (Chain-of-Thought)**: Format `<think>reasoning</think><output>response</output>`
- **MEDEC Dataset**: Medical error detection corpus with clean notes and error examples
- **Game Type**: Either "harmful" (introduce error) or "safe" (keep clean)
- **Faithfulness**: Whether attacker respects the game type (harmful→error, safe→no error)

## Requirements

### Requirement 1: Data Pipeline Architecture

**User Story:** As a researcher, I want a clear data pipeline that efficiently processes the full MEDEC dataset, so that I can train models at scale without data bottlenecks.

#### Acceptance Criteria

1. THE Data Pipeline SHALL load the full MEDEC training set (2,189 notes: 1,219 with errors, 970 clean)
2. THE Data Pipeline SHALL split data into educational SFT (70%), adaptation SFT (15%), and GRPO seeds (15%)
3. THE Data Pipeline SHALL generate 3 examples per note: 1 attacker (error analysis), 1 attacker (vanilla/safe), 1 assessor (classification)
4. THE Data Pipeline SHALL produce ~6,500 total examples: ~4,500 educational, ~1,000 adaptation, ~1,000 GRPO seeds
5. THE Data Pipeline SHALL maintain 50/50 balance between harmful and safe game types in adaptation and GRPO data

### Requirement 2: Educational SFT Data Generation

**User Story:** As a model trainer, I want high-quality educational SFT data that teaches medical reasoning, so that the base model understands medical concepts before game-specific training.

#### Acceptance Criteria

1. THE Educational SFT Generator SHALL create ~4,500 examples from 70% of MEDEC data (1,532 notes)
2. FOR EACH note, THE Generator SHALL create 3 examples: attacker-error, attacker-vanilla, assessor-classification
3. THE Generator SHALL use GPT-5 API to generate detailed medical reasoning traces in `<think>` sections
4. THE Generator SHALL use pre-fill CoT format: `<think>reasoning</think><output>content</output>`
5. THE Generator SHALL cover all 5 error types: causalOrganism, diagnosis, dosage, management, medication

### Requirement 3: Game Format Adaptation Data

**User Story:** As a model trainer, I want adaptation data that bridges educational SFT to GRPO format, so that the model can handle game-specific prompts without catastrophic forgetting.

#### Acceptance Criteria

1. THE Adaptation Generator SHALL create ~1,000 examples from 15% of MEDEC data (328 notes)
2. THE Adaptation Generator SHALL use game-style prompts matching GRPO format exactly (system + user messages)
3. THE Adaptation Generator SHALL teach copy/modify skills: exact copying for safe game, one-change modification for harmful game
4. THE Adaptation Generator SHALL maintain pre-fill CoT format consistent with educational SFT
5. THE Adaptation Generator SHALL include few-shot examples showing clean→error transformations in prompts

### Requirement 4: GRPO Seed Data Preparation

**User Story:** As a GRPO trainer, I want seed prompts that enable diverse self-play training, so that the model learns robust error detection through adversarial games.

#### Acceptance Criteria

1. THE GRPO Seed Generator SHALL use remaining 15% of MEDEC data (329 notes) as seeds
2. THE Generator SHALL split seeds 50/50 between harmful and safe game types (~165 each)
3. FOR harmful seeds, THE Generator SHALL include error examples from MEDEC to guide attacker
4. FOR safe seeds, THE Generator SHALL provide only clean notes without error examples
5. THE Generator SHALL ensure no overlap between educational (70%), adaptation (15%), and GRPO seed (15%) splits

### Requirement 5: Training Pipeline Orchestration

**User Story:** As a researcher, I want an automated training pipeline that executes all three stages in sequence, so that I can train models end-to-end without manual intervention.

#### Acceptance Criteria

1. THE Training Pipeline SHALL execute three stages: Educational SFT → Adaptation SFT → GRPO
2. THE Pipeline SHALL validate each stage completes successfully before proceeding
3. THE Pipeline SHALL save checkpoints after each stage with clear naming conventions
4. THE Pipeline SHALL log training metrics (loss, accuracy, faithfulness) at each stage
5. THE Pipeline SHALL support resuming from any stage if interrupted

### Requirement 6: Memory and Compute Optimization

**User Story:** As a model trainer, I want efficient memory usage during training, so that I can train on available hardware without OOM errors.

#### Acceptance Criteria

1. THE Training System SHALL use gradient checkpointing to reduce memory footprint
2. THE Training System SHALL use bf16 precision for 50% memory reduction
3. THE Training System SHALL batch data efficiently (batch_size=4, grad_accum=4)
4. THE Training System SHALL clear CUDA cache between GRPO rounds
5. THE Training System SHALL support training on single GPU (24GB VRAM minimum)

### Requirement 7: Data Quality Validation

**User Story:** As a researcher, I want automated validation of generated training data, so that I can catch quality issues before expensive training runs.

#### Acceptance Criteria

1. THE Validation System SHALL check all examples have valid CoT format
2. THE Validation System SHALL verify attacker examples show actual medical changes (not just cosmetic)
3. THE Validation System SHALL confirm assessor examples have correct labels
4. THE Validation System SHALL validate game type distribution is 50/50
5. THE Validation System SHALL report statistics: example counts, error type distribution, average lengths

### Requirement 8: Experiment Tracking and Reproducibility

**User Story:** As a researcher, I want comprehensive logging and versioning, so that I can reproduce experiments and compare different configurations.

#### Acceptance Criteria

1. THE Logging System SHALL record all hyperparameters (learning rate, batch size, epochs)
2. THE Logging System SHALL save data splits with random seeds for reproducibility
3. THE Logging System SHALL log GRPO metrics: rewards, faithfulness, judge classifications
4. THE Logging System SHALL save interaction logs showing attacker/assessor exchanges
5. THE Logging System SHALL timestamp all outputs with ISO format

### Requirement 9: Incremental Development Support

**User Story:** As a developer, I want to test the pipeline on small subsets before full-scale runs, so that I can iterate quickly and catch bugs early.

#### Acceptance Criteria

1. THE Pipeline SHALL support `--num_samples` parameter to limit data size
2. THE Pipeline SHALL support `--dry_run` mode that validates without training
3. THE Pipeline SHALL provide quick validation scripts that run in <5 minutes
4. THE Pipeline SHALL support resuming from existing checkpoints
5. THE Pipeline SHALL allow testing individual stages independently

### Requirement 10: Performance Monitoring

**User Story:** As a researcher, I want real-time monitoring of training quality, so that I can detect issues like mode collapse or reward hacking early.

#### Acceptance Criteria

1. THE Monitoring System SHALL track attacker faithfulness rate (target: 60-80%)
2. THE Monitoring System SHALL track assessor accuracy (target: >70%)
3. THE Monitoring System SHALL detect judge classification skew (warn if >70% one class)
4. THE Monitoring System SHALL monitor reward variance (warn if near-zero)
5. THE Monitoring System SHALL alert on format violations (target: <10%)
