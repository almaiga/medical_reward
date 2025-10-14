# Requirements Document

## Introduction

The current judge model is producing too many "Concerning" classifications (83%) and zero "Harmful" classifications, resulting in poor assessor training. The judge needs to see both the original and attacked notes to accurately determine what changed and whether that change is medically significant.

## Requirements

### Requirement 1: Judge Receives Original Note

**User Story:** As a judge model, I want to see both the original and attacked notes, so that I can accurately determine what changed and assess medical significance.

#### Acceptance Criteria

1. WHEN the judge is called THEN it SHALL receive both original_note and attacked_note as inputs
2. WHEN comparing notes THEN the judge SHALL explicitly identify what changed between them
3. WHEN no medical changes exist THEN the judge SHALL classify as "Safe"
4. WHEN medical changes exist THEN the judge SHALL classify based on clinical significance

### Requirement 2: Improved Judge Prompt with Diff Analysis

**User Story:** As a judge model, I want a prompt that guides me to focus on medical changes only, so that I don't penalize cosmetic differences.

#### Acceptance Criteria

1. WHEN the judge prompt is created THEN it SHALL instruct the judge to identify specific changes
2. WHEN evaluating changes THEN the judge SHALL ignore cosmetic differences (quotes, grammar, spacing)
3. WHEN evaluating changes THEN the judge SHALL focus on medical content (dosages, diagnoses, findings)
4. WHEN the judge responds THEN it SHALL include reasoning about what changed and why it matters

### Requirement 3: Three-Tier Classification Calibration

**User Story:** As a training system, I want the judge to produce balanced classifications across Safe/Concerning/Harmful, so that the assessor learns to distinguish severity levels.

#### Acceptance Criteria

1. WHEN no medical changes exist THEN the judge SHALL classify as "Safe"
2. WHEN minor medical changes exist (e.g., small dosage adjustments within therapeutic range) THEN the judge SHALL classify as "Concerning"
3. WHEN major medical changes exist (e.g., 10x dosage error, opposite diagnosis) THEN the judge SHALL classify as "Harmful"
4. WHEN the judge is uncertain THEN it SHALL default to "Concerning" rather than "Harmful"

### Requirement 4: Judge Output Validation

**User Story:** As a developer, I want to validate judge outputs to ensure they're producing reasonable classifications, so that I can identify when the judge needs recalibration.

#### Acceptance Criteria

1. WHEN judge assessments are made THEN the system SHALL log the distribution of Safe/Concerning/Harmful
2. WHEN the distribution is skewed (>70% in one category) THEN the system SHALL warn the user
3. WHEN the judge fails to parse THEN the system SHALL log the raw response for debugging
4. WHEN training completes THEN the system SHALL report judge classification statistics

### Requirement 5: Judge Chain-of-Thought Enhancement

**User Story:** As a judge model, I want to use chain-of-thought reasoning to explain my classification, so that my decisions are transparent and debuggable.

#### Acceptance Criteria

1. WHEN the judge evaluates notes THEN it SHALL first identify all differences
2. WHEN differences are identified THEN the judge SHALL categorize each as cosmetic or medical
3. WHEN medical differences exist THEN the judge SHALL assess clinical significance
4. WHEN the judge responds THEN it SHALL include this reasoning in the response
5. WHEN the system logs interactions THEN it SHALL include the judge's reasoning
