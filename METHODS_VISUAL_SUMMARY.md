# Visual Summary of Methods

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Educational SFT                      │
│                         (3 epochs)                               │
├─────────────────────────────────────────────────────────────────┤
│  Input: 733 medical classification examples                     │
│  Format: <think>reasoning</think><output>Safe/Concerning/       │
│          Harmful</output>                                        │
│  Goal: Learn medical knowledge + classification                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 2: Adaptation SFT                        │
│                         (1 epoch)                                │
├─────────────────────────────────────────────────────────────────┤
│  Input: 800 game-format examples + 50% educational data         │
│  Roles: Attacker (modify notes) + Defender (classify)           │
│  Goal: Learn game mechanics without forgetting medicine         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 3: Self-Play RL (GRPO)                    │
│                         (3 rounds)                               │
└─────────────────────────────────────────────────────────────────┘
```

## Self-Play Round Structure

```
Round N:
  ┌──────────────────────────────────────────────────────────┐
  │ 1. SNAPSHOT DEFENDER                                     │
  │    Defender_frozen ← copy(Policy_model)                  │
  └──────────────────────────────────────────────────────────┘
                         ↓
  ┌──────────────────────────────────────────────────────────┐
  │ 2. TRAIN ATTACKER                                        │
  │    ┌────────────────────────────────────────────┐       │
  │    │ For each seed note:                        │       │
  │    │   • Attacker generates 2 variations        │       │
  │    │   • Defender_frozen classifies them        │       │
  │    │   • Judge determines ground truth          │       │
  │    │   • Compute attacker rewards:              │       │
  │    │     - Faithfulness: ±1                     │       │
  │    │     - Game: ±1                             │       │
  │    │     - Format: ±1                           │       │
  │    │   • Update Policy_model (attacker)         │       │
  │    └────────────────────────────────────────────┘       │
  └──────────────────────────────────────────────────────────┘
                         ↓
  ┌──────────────────────────────────────────────────────────┐
  │ 3. TRAIN DEFENDER                                        │
  │    ┌────────────────────────────────────────────┐       │
  │    │ For each attacked note:                    │       │
  │    │   • Defender generates 2 classifications   │       │
  │    │   • Judge determines ground truth          │       │
  │    │   • Compute defender rewards:              │       │
  │    │     - Game: ±1                             │       │
  │    │     - Format: ±1                           │       │
  │    │   • Update Policy_model (defender)         │       │
  │    └────────────────────────────────────────────┘       │
  └──────────────────────────────────────────────────────────┘
```

## Game Types

```
┌─────────────────────────────────────────────────────────────┐
│                      HARMFUL GAME                           │
├─────────────────────────────────────────────────────────────┤
│  Seed: Medical note WITH error                             │
│  Attacker Goal: Keep/introduce error to fool defender      │
│  Defender Goal: Detect and classify error correctly        │
│                                                             │
│  Example:                                                   │
│    Seed: "Patient prescribed 200mg aspirin"  [ERROR]       │
│    Attack: "Patient prescribed 200mg aspirin" [KEEP ERROR] │
│    Defender: "Harmful" ✓ (correct)                         │
│    OR                                                       │
│    Defender: "Safe" ✗ (fooled!)                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       SAFE GAME                             │
├─────────────────────────────────────────────────────────────┤
│  Seed: Medical note WITHOUT error                          │
│  Attacker Goal: Keep note safe (no changes)                │
│  Defender Goal: Correctly identify as safe                 │
│                                                             │
│  Example:                                                   │
│    Seed: "Patient prescribed 100mg aspirin"  [CORRECT]     │
│    Attack: "Patient prescribed 100mg aspirin" [NO CHANGE]  │
│    Defender: "Safe" ✓ (correct)                            │
│    OR                                                       │
│    Defender: "Concerning" ✗ (false positive!)              │
└─────────────────────────────────────────────────────────────┘
```

## Reward Structure

```
ATTACKER REWARDS (Total: -3 to +3)
┌──────────────────────────────────────────────────────────────┐
│ Component         │ Value │ Purpose                          │
├──────────────────────────────────────────────────────────────┤
│ Faithfulness      │  ±1   │ Respects game type               │
│ (RA,revision)     │       │ • Harmful: create error          │
│                   │       │ • Safe: keep safe                │
├──────────────────────────────────────────────────────────────┤
│ Game              │  ±1   │ Fools defender                   │
│ (RA,game)         │       │ • Harmful: defender misses       │
│                   │       │ • Safe: defender says safe       │
├──────────────────────────────────────────────────────────────┤
│ Format            │  ±1   │ Uses <think> tags correctly      │
│ (RA,format)       │       │                                  │
└──────────────────────────────────────────────────────────────┘

DEFENDER REWARDS (Total: -2 to +2)
┌──────────────────────────────────────────────────────────────┐
│ Component         │ Value │ Purpose                          │
├──────────────────────────────────────────────────────────────┤
│ Game              │  ±1   │ Classifies correctly             │
│ (RD,game)         │       │ • Matches judge ground truth     │
├──────────────────────────────────────────────────────────────┤
│ Format            │  ±1   │ Uses <think> tags correctly      │
│ (RD,format)       │       │                                  │
└──────────────────────────────────────────────────────────────┘
```

## Judge Model Role

```
┌─────────────────────────────────────────────────────────────┐
│                    JUDGE (MedGemma-4B)                       │
│                  Ground Truth Adjudicator                    │
├─────────────────────────────────────────────────────────────┤
│  Input:                                                      │
│    • Original note                                           │
│    • Attacked note                                           │
│    • Defender's classification                               │
│                                                              │
│  Process:                                                    │
│    1. Compare notes                                          │
│    2. Identify changes                                       │
│    3. Ignore cosmetic (quotes, grammar, spacing)            │
│    4. Focus on medical (dosage, diagnosis, medication)      │
│    5. Classify actual harm: Safe/Concerning/Harmful         │
│                                                              │
│  Output:                                                     │
│    • actual_harm: Safe | Concerning | Harmful               │
│    • assessor_was_correct: True | False                     │
└─────────────────────────────────────────────────────────────┘
```

## Key Metrics Tracked

```
ATTACKER METRICS:
  • Deception rate: % attacks that fool defender
  • Faithfulness rate: % attacks respecting game type
  • Diversity: Distribution across Safe/Concerning/Harmful

DEFENDER METRICS:
  • Overall accuracy: % correct vs. judge
  • Per-category accuracy: Safe, Concerning, Harmful
  • False positive rate: % safe notes flagged
  • False negative rate: % harmful notes missed

CO-EVOLUTION METRICS:
  • Round-over-round improvement
  • Distribution balance (target: 33/33/33)
  • Judge validation warnings
```

## Data Flow

```
MEDEC-MS Dataset (1,532 notes)
         ↓
    ┌────┴────┐
    │  Split  │
    └────┬────┘
         ↓
    ┌────┴──────────────────────────────┐
    │                                   │
Educational SFT        Adaptation SFT   │
(733 examples)         (800 examples)   │
    │                       │           │
    └───────┬───────────────┘           │
            ↓                           │
    Pre-trained Model                   │
            ↓                           │
    Self-Play RL (GRPO)                 │
            ↓                           │
    Final Model ←───────────────────────┘
                                  (Evaluation)
```
