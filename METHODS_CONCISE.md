# Methods (Concise Version for Paper)

## 3. Methods

### 3.1 Problem Formulation

We adapt Self-RedTeam [1] to medical error detection, formulating it as a two-player game where a single model alternates between attacker and defender roles. The attacker generates medical note variations (introducing or preserving errors), while the defender classifies notes as Safe, Concerning, or Harmful.

### 3.2 Dataset and Model

**Dataset**: MEDEC-MS validation set (1,532 clinical notes with error annotations). We split data into educational SFT (733 examples), adaptation SFT (800 examples), and evaluation sets. Game types are balanced 50/50 between harmful (seed = error note) and safe (seed = corrected note).

**Models**: Qwen2.5-3B-Instruct (policy model) and MedGemma-4B-IT (judge for ground-truth adjudication).

### 3.3 Training Pipeline

**Phase 1 - Educational SFT**: 3-epoch fine-tuning on 733 medical classification examples using pre-fill Chain-of-Thought format: `<think>[reasoning]</think><output>[classification]</output>`.

**Phase 2 - Adaptation SFT**: 1-epoch adaptation on 800 game-format examples (mixed 50/50 with educational data) to teach role-specific behaviors while preserving medical knowledge.

**Phase 3 - Self-Play RL**: 3 rounds of Group Relative Policy Optimization (GRPO) where:
1. Defender model is snapshotted (frozen)
2. Attacker trains against frozen defender (16 seeds, 2 generations each)
3. Defender trains on generated attacks

### 3.4 Reward Structure

Following Self-RedTeam's general-sum formulation:

**Attacker** (range: -3 to +3):
- Faithfulness (±1): Respects game type (prevents reward hacking)
- Game (±1): Fools defender
- Format (±1): Proper CoT usage

**Defender** (range: -2 to +2):
- Game (±1): Correct classification vs. judge ground truth
- Format (±1): Proper CoT usage

The judge model compares original vs. attacked notes, ignoring cosmetic changes and focusing on medical content (dosages, diagnoses, medications).

### 3.5 Implementation

We use TRL's GRPO implementation with key modifications: (1) tokenizer patching to force BOS tokens for Qwen models, (2) memory optimization via gradient checkpointing and model snapshot clearing, (3) distribution monitoring to detect reward hacking. Training uses learning rates of 5e-6 (SFT) and 1e-5 (RL), batch size 4 with 4-step gradient accumulation, and BFloat16 precision.

---

**Key Differences from Original Self-RedTeam**: We adapt to medical domain with three-class classification, combine response_refusal and response_harmfulness into single game reward, use pre-fill CoT format, and operate at smaller scale (3B model, 16 samples/round, 3 rounds).

---

[1] Liu, M., et al. (2025). Chasing Moving Targets with Online Self-Play RL for Safer LMs. arXiv:2506.07468.
