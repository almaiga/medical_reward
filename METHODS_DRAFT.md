# Methods: Self-Play Reinforcement Learning for Medical Note Safety

## Overview

We adapt the Self-RedTeam framework [Liu et al., 2025] to the medical domain, training a single language model to alternate between attacker and defender roles in a two-player adversarial game for detecting medical errors in clinical notes.

## 3.1 Problem Formulation

We formulate medical note safety as a two-player zero-sum game:

**Attacker Role**: Given a seed medical note (either with or without errors), generate a modified version that:
- In "harmful" games: Introduces subtle medical errors
- In "safe" games: Preserves the original safe content

**Defender Role**: Classify a medical note into three categories:
- Safe: No medical errors
- Concerning: Minor medical errors (e.g., 1.5× dosage change)
- Harmful: Major medical errors (e.g., wrong drug class, opposite diagnosis)

## 3.2 Dataset

**Base Dataset**: MEDEC-MS (Medical Error Detection and Correction - Multi-Specialty) validation set
- 1,532 clinical notes with ground-truth error annotations
- Error types: dosage errors, diagnosis errors, medication errors, clinical finding errors

**Data Split**:
- Educational SFT: ms-train-0 to ms-train-732 (733 examples)
- Adaptation SFT: ms-train-733 to ms-train-1532 (800 examples)
- Evaluation: Held-out test set

**Game Type Distribution**:
- 50% harmful games (seed = note with error)
- 50% safe games (seed = corrected note)

## 3.3 Model Architecture

**Base Model**: Qwen2.5-3B-Instruct
- 3 billion parameters
- Pre-trained on general domain text
- Fine-tuned for medical reasoning

**Judge Model**: MedGemma-4B-IT (Google)
- Medical specialist model for ground-truth harm assessment
- Used to adjudicate game outcomes during training

## 3.4 Training Pipeline

### Phase 1: Educational Supervised Fine-Tuning (SFT)

**Objective**: Teach the model medical knowledge and classification skills

**Data**: 733 examples from MEDEC-MS (ms-train-0 to 732)

**Format**: Pre-fill Chain-of-Thought (CoT)
```
<think>[medical reasoning]</think>
<output>[classification: Safe/Concerning/Harmful]</output>
```

**Training**:
- 3 epochs
- Learning rate: 5e-6
- Batch size: 4 (per device)
- Gradient accumulation: 4 steps

### Phase 2: Format Adaptation SFT

**Objective**: Adapt model to game-playing format while preserving medical knowledge

**Data**: 800 examples from MEDEC-MS (ms-train-733 to 1532)

**Format**: Pre-fill CoT with role-specific prompts
- Attacker: Generate medical note variations
- Defender: Classify notes for errors

**Training**:
- 1 epoch (to avoid catastrophic forgetting)
- Learning rate: 5e-6
- Mixed with 50% educational data

### Phase 3: Self-Play Reinforcement Learning (GRPO)

**Algorithm**: Group Relative Policy Optimization (GRPO)
- On-policy RL algorithm
- Generates multiple completions per prompt
- Computes advantages relative to group mean

**Self-Play Structure**:
```
Round 1:
  1. Snapshot defender model (frozen)
  2. Train attacker against frozen defender
  3. Train defender on attacked notes

Round 2:
  1. Snapshot new defender
  2. Train attacker against updated defender
  3. Train defender on new attacks
  
... (repeat for N rounds)
```

**Hyperparameters**:
- Rounds: 3
- Samples per round: 16 seed notes
- Generations per prompt: 2
- Learning rate: 1e-5
- Batch size: 1 (per device)
- Gradient accumulation: 4 steps
- Max prompt length: 1536 tokens
- Max completion length: 1024 tokens

## 3.5 Reward Structure

We implement the Self-RedTeam reward structure with domain-specific adaptations:

### Attacker Rewards (Total: -3.0 to +3.0)

**1. Faithfulness Reward (RA,revision)**: ±1.0
- Harmful game: +1 if creates Concerning/Harmful, -1 if creates Safe
- Safe game: +1 if keeps Safe, -1 if introduces errors
- **Purpose**: Prevents reward hacking, ensures balanced distribution

**2. Game Reward (RA,game)**: ±1.0
- Harmful game: +1 if defender misclassifies, -1 if defender catches error
- Safe game: +1 if defender correctly says Safe, -1 if defender flags false positive
- **Purpose**: Rewards fooling the defender

**3. Format Reward (RA,format)**: ±1.0
- +1 if uses `<think>` tags correctly, -1 otherwise
- **Purpose**: Ensures proper CoT formatting

### Defender Rewards (Total: -2.0 to +2.0)

**1. Game Reward (RD,game)**: ±1.0
- +1 if classification matches ground truth, -1 otherwise
- **Purpose**: Rewards correct classification

**2. Format Reward (RD,format)**: ±1.0
- +1 if uses `<think>` tags correctly, -1 otherwise
- **Purpose**: Ensures proper CoT formatting

**Ground Truth Adjudication**:
- Judge model (MedGemma-4B) compares original vs. attacked notes
- Classifies actual harm level: Safe, Concerning, or Harmful
- Ignores cosmetic changes (quotes, grammar, spacing)
- Focuses on medical content changes (dosages, diagnoses, medications)

## 3.6 Implementation Details

**Framework**: TRL (Transformer Reinforcement Learning) library with GRPO

**Tokenizer Patching**: 
- Qwen models require BOS tokens for proper generation
- GRPO sets `add_special_tokens=False` by default
- We monkey-patch tokenizer methods to force `add_special_tokens=True`
- Prevents garbage output during RL training

**Memory Optimization**:
- BFloat16 precision (if supported)
- Gradient checkpointing enabled
- Model snapshots cleared after each round
- PyTorch CUDA memory cache cleared between rounds

**Validation**:
- Judge classification distribution monitored
- Faithfulness tracking (% of attacks respecting game type)
- Diversity statistics (Safe/Concerning/Harmful distribution)
- Warnings if distribution becomes skewed (>70% in one category)

## 3.7 Evaluation Metrics

**Attacker Performance**:
- Deception rate: % of attacks that fool the defender
- Faithfulness rate: % of attacks respecting game type
- Diversity: Distribution across Safe/Concerning/Harmful

**Defender Performance**:
- Classification accuracy: % correct vs. judge ground truth
- Per-category accuracy: Safe, Concerning, Harmful
- False positive rate: % safe notes incorrectly flagged
- False negative rate: % harmful notes missed

**Co-Evolution Metrics**:
- Round-over-round improvement
- Attacker win rate vs. defender accuracy
- Distribution balance (target: 33/33/33 across categories)

## 3.8 Key Differences from Original Self-RedTeam

**Domain Adaptation**:
- Medical error detection instead of jailbreak attacks
- Three-class classification (Safe/Concerning/Harmful) vs. binary
- Seed-based prompts (existing notes) vs. generated prompts

**Reward Simplification**:
- Combine `response_refusal` + `response_harmfulness` into single `game` reward
- Appropriate for medical domain where classification captures both aspects

**Format**:
- Pre-fill CoT (reasoning first) vs. post-fill CoT (response first)
- Trade-off: More explicit reasoning vs. shorter generation sequences

**Scale**:
- 3B parameter model vs. 8B in original paper
- 16 samples/round vs. larger batches in original
- 3 rounds vs. extended training in original

---

## References

Liu, M., et al. (2025). Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models. arXiv:2506.07468.

MEDEC Dataset: [Citation needed]

Qwen2.5: [Citation needed]

MedGemma: [Citation needed]
