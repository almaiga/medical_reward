# Methods: Bullet Point Summary

## What We Did

### Problem Setup
- Adapted Self-RedTeam framework to medical error detection
- Two-player game: single model alternates between attacker and defender
- Attacker: modifies medical notes (introduce/preserve errors)
- Defender: classifies notes (Safe/Concerning/Harmful)
- Judge: MedGemma-4B provides ground truth

### Data
- MEDEC-MS validation set: 1,532 clinical notes with error annotations
- Split: 733 educational + 800 adaptation + evaluation
- Game types: 50% harmful (error notes) + 50% safe (corrected notes)
- Error types: dosage, diagnosis, medication, clinical findings

### Model
- Policy: Qwen2.5-3B-Instruct
- Judge: MedGemma-4B-IT
- Format: Pre-fill CoT `<think>reasoning</think><output>response</output>`

### Training (3 Phases)

**Phase 1: Educational SFT**
- 733 medical classification examples
- 3 epochs, LR 5e-6
- Goal: Learn medical knowledge + classification

**Phase 2: Adaptation SFT**
- 800 game-format examples (mixed 50/50 with educational)
- 1 epoch, LR 5e-6
- Goal: Learn game mechanics without forgetting medicine

**Phase 3: Self-Play RL (GRPO)**
- 3 rounds of self-play
- Each round:
  1. Snapshot defender (frozen)
  2. Train attacker vs. frozen defender (16 seeds × 2 generations)
  3. Train defender on generated attacks
- LR 1e-5, batch size 4, gradient accumulation 4

### Rewards

**Attacker (-3 to +3):**
- Faithfulness ±1: Respects game type (prevents reward hacking)
- Game ±1: Fools defender
- Format ±1: Proper CoT usage

**Defender (-2 to +2):**
- Game ±1: Correct classification vs. judge
- Format ±1: Proper CoT usage

### Key Implementation Details
- Tokenizer patching: Force BOS tokens for Qwen models
- Memory optimization: Gradient checkpointing, snapshot clearing
- Validation: Distribution monitoring, faithfulness tracking
- Judge: Compares notes, ignores cosmetic changes, focuses on medical content

### Differences from Original Self-RedTeam
- Domain: Medical error detection (not jailbreaks)
- Classification: 3-class (Safe/Concerning/Harmful) not binary
- Rewards: Combined response_refusal + response_harmfulness into single game reward
- Format: Pre-fill CoT (not post-fill)
- Scale: 3B model, 16 samples/round, 3 rounds (smaller than original)

## Key Innovations

1. **Faithfulness Reward**: Prevents attacker from always choosing easy path (e.g., always outputting safe notes)
2. **Balanced Game Types**: 50/50 harmful/safe ensures defender can't exploit distribution
3. **Judge Validation**: Medical specialist model provides reliable ground truth
4. **Format Adaptation**: Separate SFT phase prevents catastrophic forgetting
5. **Distribution Monitoring**: Detects and warns about reward hacking

## Metrics

**Attacker:**
- Deception rate (% fool defender)
- Faithfulness rate (% respect game type)
- Diversity (Safe/Concerning/Harmful distribution)

**Defender:**
- Overall accuracy (% match judge)
- Per-category accuracy
- False positive/negative rates

**Co-Evolution:**
- Round-over-round improvement
- Distribution balance (target: 33/33/33)

## Expected Outcomes

**Round 1:**
- Attacker learns faithfulness (respect game type)
- Defender learns basic classification

**Round 2:**
- Attacker learns to create subtle errors
- Defender improves detection

**Round 3:**
- Both agents reach equilibrium
- Balanced, diverse attacks
- Robust defender

## Why This Matters

- **Medical Safety**: Automated detection of subtle medical errors
- **Scalability**: Single model plays both roles (no separate red team needed)
- **Robustness**: Co-evolution prevents overfitting to static attacks
- **Interpretability**: CoT format shows reasoning
- **Efficiency**: 3B model suitable for deployment
