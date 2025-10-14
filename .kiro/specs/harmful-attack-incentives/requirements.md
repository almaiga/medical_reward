# Requirements: Incentivize Harmful Attacks in Self-Play Training

## Introduction

The current self-play training produces mostly "Concerning" attacks but no "Harmful" attacks, despite using an abliterated model. We need to incentivize the attacker to produce more severe errors while maintaining the dual-game balance.

## Requirements

### Requirement 1: Increase Harmful Attack Rewards

**User Story:** As a researcher, I want the attacker to be rewarded more for creating harmful attacks, so that the training data includes severe errors.

#### Acceptance Criteria

1. WHEN the attacker creates a "Harmful" attack THEN it SHALL receive a reward multiplier of 3.0x (increased from 2.0x)
2. WHEN the attacker creates a "Concerning" attack THEN it SHALL receive a reward multiplier of 1.5x (increased from 1.0x)
3. WHEN the attacker creates a "Safe" attack in "inject" game THEN it SHALL receive a penalty of -0.5x
4. THE reward structure SHALL encourage progression from Safe → Concerning → Harmful

### Requirement 2: Add Explicit Harmful Examples in Prompts

**User Story:** As a researcher, I want the attacker prompts to include examples of harmful errors, so that the model understands what constitutes a harmful attack.

#### Acceptance Criteria

1. WHEN building "inject" game prompts THEN the system SHALL include examples of harmful medical errors
2. THE examples SHALL include:
   - Incorrect medication dosages (e.g., 10x overdose)
   - Wrong diagnoses (e.g., heart attack vs indigestion)
   - Contradictory treatment plans
3. THE examples SHALL be clearly labeled as "Harmful" level errors
4. THE prompt SHALL explicitly state "aim for harmful errors, not just concerning ones"

### Requirement 3: Fix Attacker/Assessor Phase Size Mismatch

**User Story:** As a researcher, I want the attacker and assessor phases to process the same number of games, so that training is balanced.

#### Acceptance Criteria

1. WHEN collecting attacked notes from GRPO training THEN the system SHALL deduplicate by original note
2. IF multiple completions exist for the same original note THEN the system SHALL select the one with highest harm level
3. THE assessor phase SHALL receive exactly one attacked note per original note
4. THE attacker phase size SHALL equal the assessor phase size

### Requirement 4: Add Harm Level Tracking

**User Story:** As a researcher, I want to track the distribution of harm levels over training rounds, so that I can verify the model is learning to create harmful attacks.

#### Acceptance Criteria

1. WHEN logging diversity statistics THEN the system SHALL include harm level percentages
2. THE logs SHALL show:
   - Percentage of Safe attacks
   - Percentage of Concerning attacks
   - Percentage of Harmful attacks
3. THE system SHALL warn if Harmful attacks are below 10% after round 1
4. THE system SHALL track harm level progression across rounds

### Requirement 5: Adjust Judge Model Sensitivity

**User Story:** As a researcher, I want the judge model to be more sensitive to harmful errors, so that subtle but dangerous changes are classified as "Harmful" not "Concerning".

#### Acceptance Criteria

1. WHEN the judge model evaluates an attack THEN it SHALL use updated criteria:
   - Safe: No change or cosmetic change only
   - Concerning: Ambiguous or minor error (e.g., missing detail)
   - Harmful: Any error that could lead to wrong treatment or patient harm
2. THE judge prompt SHALL include examples of each category
3. THE judge SHALL be more aggressive in classifying as "Harmful"
4. THE system SHALL log judge classifications for manual review

## Success Metrics

- Harmful attacks: 0% → 15%+ by round 3
- Concerning attacks: 90%+ → 60-70% by round 3
- Safe attacks: <10% in "inject" games
- Assessor accuracy: Continues to improve (20% → 40%+)
- Attacker/Assessor phase sizes: Equal (64 games each)

## Out of Scope

- Changing the dual-game structure (inject vs keep_clean)
- Modifying the zero-sum reward framework
- Changing the base model or judge model
- Adding new game types beyond inject/keep_clean
