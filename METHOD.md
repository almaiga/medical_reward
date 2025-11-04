# Method: Self-Play Reinforcement Learning for Medical Error Detection

## Overview

We develop a medical error detection system using self-play reinforcement learning, where a single language model alternates between two adversarial roles: an **Attacker** that introduces medical errors and a **Defender** (Assessor) that detects them. This approach is inspired by Self-RedTeam (Liu et al., 2024) but adapted for medical safety.

## Problem Formulation

### Task Definition

Given a medical note, classify it as:
- **Safe**: No medical errors present
- **Harmful**: Contains medical errors that could harm patients

We focus on 5 types of medical errors from the MEDEC dataset:
1. **Wrong Dosage**: Incorrect medication amounts
2. **Wrong Diagnosis**: Incorrect condition identification
3. **Wrong Medication**: Incorrect drug choice
4. **Wrong Organism**: Incorrect pathogen identification
5. **Wrong Management**: Incorrect treatment plan

### Game-Theoretic Framework

We formulate medical error detection as a two-player game:

**Players:**
- **Attacker (A)**: Modifies clean medical notes to introduce errors
- **Defender (D)**: Classifies notes as Safe or Harmful

**Game Types:**
- **Harmful Game**: Attacker should introduce a medical error; Defender should detect it
- **Safe Game**: Attacker should keep the note safe; Defender should classify it as Safe

This dual-game structure prevents the Defender from over-refusing (classifying all notes as Harmful).

## Architecture

### Shared Policy Model

We use a single language model (Qwen3-4B) that plays both roles:
- **During Attacker phase**: Model generates adversarial medical notes
- **During Assessor phase**: Model classifies notes for errors

### Judge Model

We use MedGemma-4B-IT as an independent judge to:
1. Determine ground-truth harm level (Safe/Harmful)
2. Evaluate thinking quality (specific medical reasoning vs. lazy meta-commentary)
3. Provide reward signals for both players

### Hidden Chain-of-Thought

Both players use a structured reasoning format:
```
<think>[private medical reasoning]</think>
<output>[public action: modified note or classification]</output>
```

The `<think>` section is hidden from the opponent, enabling strategic planning.

## Training Procedure

### Data Preparation

We use the MEDEC-MS validation set, which contains medical notes with documented errors:
- **Clean notes**: Corrected versions (medically correct)
- **Error examples**: Original versions with errors
- **Error types**: Labeled error categories

For each training round:
- 50% Harmful games: Show clean note + error example → ask Attacker to introduce similar error
- 50% Safe games: Show clean note → ask Attacker to keep it safe

### Self-Play Training Loop

Each round consists of two phases:

#### Phase 1: Attacker Training

1. **Input**: Clean medical notes with game type (harmful/safe)
2. **Attacker generates**: Modified notes (with or without errors)
3. **Frozen Defender evaluates**: Classifies each note as Safe/Harmful
4. **Judge assesses**: Determines ground-truth harm level
5. **Compute rewards**: Update Attacker via GRPO

#### Phase 2: Defender Training

1. **Input**: Adversarial notes from Attacker training
2. **Defender generates**: Classifications (Safe/Harmful) with reasoning
3. **Judge assesses**: Determines ground-truth harm level and reasoning quality
4. **Compute rewards**: Update Defender via GRPO

### Reward Structure

We implement a general-sum reward structure based on Self-RedTeam:

#### Attacker Rewards (4 components, range: -4.0 to +4.0)

**1. Revision Faithfulness (R_A,revision)**: ±1.0
- Harmful game: +1 if error introduced, -1 if note unchanged
- Safe game: +1 if note unchanged, -1 if error introduced
- **Purpose**: Ensure Attacker respects game type

**2. Defender Classification (R_A,refusal)**: ±1.0
- Harmful game: +1 if Defender says "Safe" (fooled), -1 if "Harmful" (caught)
- Safe game: +1 if Defender says "Harmful" (over-refused), -1 if "Safe" (correct)
- **Purpose**: Reward Attacker for fooling Defender

**3. Actual Harm Level (R_A,harmfulness)**: ±1.0
- +1 if note contains error (Harmful), -1 if safe
- **Purpose**: Reward Attacker for creating challenging adversarial examples

**4. Format Reward (R_A,format)**: ±1.0
- +1 if correct CoT format, -1 otherwise
- **Purpose**: Ensure structured reasoning

#### Defender Rewards (4 components, range: -4.0 to +4.0)

**1. Classification Correctness (R_D,refusal)**: ±1.0
- +1 if classification matches ground truth, -1 otherwise
- **Purpose**: Reward correct classification

**2. Always Detect Errors (R_D,harmfulness)**: ±1.0
- +1 if classification is correct, -1 otherwise
- **Purpose**: Emphasize error detection (matches paper structure)

**3. Format Reward (R_D,format)**: ±1.0
- +1 if correct CoT format, -1 otherwise
- **Purpose**: Ensure structured reasoning

**4. Thinking Quality (R_D,thinking)**: ±1.0
- +1 if reasoning contains specific medical details, -1 if lazy meta-commentary
- Evaluated by Judge model
- **Purpose**: Encourage thorough medical analysis

### Optimization

We use **Group Relative Policy Optimization (GRPO)** with:
- Learning rate: 1e-5
- Generations per prompt: 4
- Temperature: 0.7
- Top-p: 0.9
- Repetition penalty: 1.15

## Anti-Reward-Hacking Mechanisms

### 1. Faithfulness Constraint

The R_A,revision component prevents the Attacker from:
- Always introducing errors (would fail safe games)
- Always keeping notes safe (would fail harmful games)
- Ignoring the game type

### 2. Multi-Component Rewards

Multiple conflicting objectives prevent single-objective optimization:
- Attacker must balance: faithfulness, fooling Defender, creating errors
- Defender must balance: classification accuracy, error detection, reasoning quality

### 3. Two Game Types

Dual games prevent overfitting:
- Attacker can't learn "always attack" or "always copy"
- Defender can't learn "always refuse" or "always accept"

### 4. Judge-Based Evaluation

External judge provides objective ground truth that players cannot manipulate.

### 5. Thinking Quality Evaluation

Judge evaluates reasoning quality to prevent lazy responses like:
- "I'll analyze this systematically" (meta-commentary)
- Short responses without medical details

Good reasoning must include:
- Specific medical details (dosages, findings, lab values)
- Error type checks with ✓/✗ marks
- Medical reasoning (e.g., "10mg is within 10-40mg range")

## Prompt Design

### Attacker Prompt (Harmful Game)

```
You are testing medical safety by introducing realistic medical errors.

Clean note (CORRECT):
[clean medical note]

Error example showing [error_type]:
[note with error]

YOUR TASK: Introduce a similar [error_type] error to the clean note above.
Make ONE change that introduces a medical error. Keep everything else identical.

<think>[Explain your error]</think>
<output>[Note with error]</output>
```

### Attacker Prompt (Safe Game)

```
You are ensuring medical note safety.

Clean note (CORRECT):
[clean medical note]

YOUR TASK: Keep this note SAFE. Copy it exactly without changing any medical information.

<think>[Confirm it's correct]</think>
<output>[Exact copy]</output>
```

### Defender Prompt

```
You are a medical safety classifier reviewing clinical notes for errors.

Review this medical note for errors:
[medical note]

Explain your medical reasoning with specific details:
- What is this note about?
- What are the key medical details (dosages, diagnoses, treatments)?
- Are they correct or incorrect? Why?

Then classify as Safe or Harmful.

<think>[explain your reasoning with specific details]</think>
<output>Safe/Harmful</output>
```

The Defender prompt emphasizes:
- Natural medical reasoning (not rigid templates)
- Specific details from the note
- Explaining WHY, not just stating conclusions

## Training Dynamics

### Co-Evolution

Over training rounds:
1. **Attacker learns** to introduce subtle, realistic errors
2. **Defender learns** to detect increasingly sophisticated attacks
3. **Both improve** through adversarial interaction

### Diversity Metrics

We track:
- **Faithfulness**: % of games where Attacker respects game type
- **Judge distribution**: Balance of Safe/Harmful classifications
- **Thinking quality**: % of responses with specific medical reasoning

### Convergence

Training continues until:
- Attacker faithfulness stabilizes (>70% for both game types)
- Defender accuracy plateaus
- Judge distribution is balanced (40-60% for each class)

## Implementation Details

- **Base model**: Qwen2.5-3B-Instruct (SFT-adapted for medical CoT)
- **Judge model**: MedGemma-4B-IT
- **Dataset**: MEDEC-MS validation set
- **Training rounds**: 4
- **Samples per round**: 32 (16 harmful + 16 safe)
- **Framework**: TRL (Transformer Reinforcement Learning)
- **Hardware**: Single GPU (CUDA/MPS)

## Key Innovations

1. **Medical domain adaptation** of Self-RedTeam framework
2. **Thinking quality evaluation** to prevent lazy reasoning
3. **Clean→error transformation** approach for teaching realistic error patterns
4. **Dual-game structure** to prevent over-refusal
5. **Judge-based ground truth** for objective evaluation

## References

- Liu et al. (2024). Self-RedTeam: Self-Play Reinforcement Learning for Language Model Safety. arXiv:2506.07468
- MEDEC: Medical Error Detection and Correction dataset
- GRPO: Group Relative Policy Optimization (Shao et al., 2024)
