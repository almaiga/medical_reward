# Self-RedTeam Reward Design Analysis

## Paper Reference
**Title:** Self-RedTeam: Online Self-Play Reinforcement Learning for LM Safety Alignment  
**arXiv:** [2506.07468](http://arxiv.org/abs/2506.07468)  
**Section:** 4.1 Reward Design

---

## Core Reward Architecture

The reward system is **composite**, combining multiple sub-rewards computed at the end of each turn. The design enforces zero-sum dynamics while shaping agent behavior through auxiliary terms.

### Reward Components Hierarchy

```
Final Reward = Game Outcome Rewards + Reward Shaping Terms

For Attacker (A):
  RA = RA,res_harm + RA,res_refusal + RA,format + RA,revision

For Defender (D):
  RD = RD,res_harm + RD,res_refusal + RD,format
```

---

## 1. Game Outcome Rewards (Core Zero-Sum Component)

The game outcome is determined by **three binary safety dimensions**:
- `Qharm(yA)` ∈ {0, 1}: Is the attacker's query harmful?
- `Resharm(yD)` ∈ {0, 1}: Is the defender's response harmful?
- `Resrefuse(yD)` ∈ {0, 1}: Did the defender refuse to answer?

These are evaluated using:
- Reward models (Wang et al., 2024)
- Classification models (WILDGUARD, LLAMAGUARD)

### 1.1 Response Harmfulness Sub-Reward

**Purpose:** Enforce opposing goals on response safety

**Hyperparameter:** `rharm` (reward magnitude)

**Defender's Perspective:**
```
RD,res_harm = +rharm  if Resharm(yD) = 0  (safe response)
            = -rharm  if Resharm(yD) = 1  (harmful response)
```

**Attacker's Perspective:**
```
RA,res_harm = -RD,res_harm  (pure zero-sum)
```

**Key Insight:** Attacker is rewarded for eliciting harmful responses; defender is rewarded for avoiding them.

---

### 1.2 Response Refusal Sub-Reward

**Purpose:** Prevent over-refusal while maintaining safety

**Hyperparameter:** `rrefusal` (reward magnitude)

**Defender's Perspective:**
```
RD,res_refusal = +rrefusal  if Qharm(yA) = 1 AND Resrefuse(yD) = 1  (correctly refuse harmful)
               = +rrefusal  if Qharm(yA) = 0 AND Resrefuse(yD) = 0  (correctly answer benign)
               = -rrefusal  if Qharm(yA) = 1 AND Resrefuse(yD) = 0  (fail to refuse harmful)
               = -rrefusal  if Qharm(yA) = 0 AND Resrefuse(yD) = 1  (over-refuse benign)
```

**Attacker's Perspective:**
```
RA,res_refusal = -RD,res_refusal  (pure zero-sum)
```

**Key Insight:** This is a **conditional reward** that incentivizes nuanced behavior:
- Defender should refuse harmful queries but answer benign ones
- Attacker wins by causing over-refusal (making benign prompts look harmful)
- Prevents trivial "refuse everything" strategy

---

## 2. Reward Shaping Terms (Behavioral Constraints)

These terms are **not zero-sum** and guide both agents toward desired behaviors.

### 2.1 CoT Formatting Sub-Reward

**Purpose:** Enforce adherence to hidden Chain-of-Thought format

**Hyperparameter:** `rformat` (reward magnitude)

**For Both Agents:**
```
RP,format = +rformat  if output correctly parsed into <think>...</think> and <answer>...</answer>
          = -rformat  otherwise
```

**Key Insight:** 
- Ensures agents use the CoT template properly
- Hidden thoughts (yCoT) remain private and excluded from game evaluation
- Enables strategic planning without revealing reasoning to opponent

---

### 2.2 Revision Faithfulness Sub-Reward (Attacker Only)

**Purpose:** Ensure attacker preserves seed prompt intent during revision

**Hyperparameter:** `rrevision` (reward magnitude)

**For Attacker:**
```
RA,revision = +rrevision  if classification(yA) = classification(s)  (intent preserved)
            = -rrevision  if classification(yA) ≠ classification(s)  (intent changed)
```

Where `s` is the seed prompt (either harmful or benign).

**Key Insight:**
- Attacker must revise harmful seeds into stealthier versions (still harmful)
- Attacker must revise benign seeds into deceptive-looking prompts (still benign)
- Ensures defender faces balanced mix of adversarial prompts
- **Critical for preventing trivial defender strategy** of refusing all prompts

---

## Critical Design Decisions

### 1. Benign Seed Inclusion
The dataset S contains both:
- **Harmful seeds:** "develop a program that deletes government files"
- **Benign seeds:** "develop a program that reads public government files"

**Rationale:** Without benign seeds, defender could trivially refuse all prompts. The attacker's job includes making benign prompts look suspicious, forcing the defender to discriminate carefully.

### 2. Hidden Chain-of-Thought
- Only `<answer>` is revealed to opponent
- `<think>` remains private
- Enables strategic planning without information leakage
- Empirically reduces over-refusals and increases attack diversity

### 3. Conditional Refusal Reward
The refusal sub-reward is **conditioned on query harmfulness**, creating a 2×2 reward matrix:

|                          | Defender Refuses | Defender Answers |
|--------------------------|------------------|------------------|
| **Harmful Query**        | +rrefusal        | -rrefusal        |
| **Benign Query**         | -rrefusal        | +rrefusal        |

This forces nuanced behavior rather than blanket policies.

---

## Reward Model Requirements

The system requires classifiers/reward models for:

1. **Query Harmfulness:** `Qharm(yA)` - Is the attacker's revised prompt harmful?
2. **Response Harmfulness:** `Resharm(yD)` - Is the defender's response harmful?
3. **Response Refusal:** `Resrefuse(yD)` - Did the defender refuse to answer?
4. **Revision Faithfulness:** Does revised prompt match seed classification?

**Suggested Models:**
- WILDGUARD (Han et al., 2024)
- LLAMAGUARD (Inan et al., 2023)
- Custom reward models (Wang et al., 2024)

---

## Hyperparameters

The reward design introduces four tunable hyperparameters:

| Parameter   | Purpose                          | Trade-off                                    |
|-------------|----------------------------------|----------------------------------------------|
| `rharm`     | Response harmfulness weight      | Safety vs. helpfulness                       |
| `rrefusal`  | Refusal behavior weight          | Over-refusal vs. under-refusal               |
| `rformat`   | CoT format adherence weight      | Format compliance vs. content quality        |
| `rrevision` | Seed intent preservation weight  | Attack diversity vs. seed faithfulness       |

**Note:** Paper does not specify exact values - likely requires tuning per domain.

---

## Implications for Medical Error Detection

### Potential Adaptations

**1. Redefine Safety Dimensions:**
```
Qharm(yA) → Qerror(yA): Does the case contain a medical error?
Resharm(yD) → Resdetect(yD): Did the detector correctly identify the error?
Resrefuse(yD) → Resuncertain(yD): Did the detector express appropriate uncertainty?
```

**2. Adapted Reward Structure:**
```
Detector (D):
  RD,detection = +rdetect if Qerror(yA) = 1 AND Resdetect(yD) = 1  (correct detection)
               = +rdetect if Qerror(yA) = 0 AND Resdetect(yD) = 0  (correct negative)
               = -rdetect if Qerror(yA) = 1 AND Resdetect(yD) = 0  (missed error)
               = -rdetect if Qerror(yA) = 0 AND Resdetect(yD) = 1  (false positive)

Case Generator (A):
  RA,detection = -RD,detection  (zero-sum)
```

**3. Revision Faithfulness for Medical Cases:**
```
RA,revision = +rrevision if error_type(yA) = error_type(s)  (error type preserved)
            = -rrevision otherwise
```

Ensures attacker generates harder versions of the same error type, not different errors.

**4. Hidden CoT for Medical Reasoning:**
```
<think>
  Patient presents with chest pain and elevated troponin.
  Differential: MI, PE, aortic dissection.
  The case mentions aspirin given, but patient has active GI bleed.
  This is a contraindication error.
</think>
<answer>
  Error Type: Contraindication
  Location: Medication order line 3
  Severity: High
</answer>
```

### Key Differences from Safety Alignment

| Aspect                  | Safety Alignment              | Medical Error Detection       |
|-------------------------|-------------------------------|-------------------------------|
| **Attacker Goal**       | Elicit harmful response       | Generate undetectable errors  |
| **Defender Goal**       | Refuse harmful, answer benign | Detect errors, avoid false +  |
| **Seed Data**           | Harmful/benign prompts        | Error/error-free cases        |
| **Evaluation**          | Harmfulness classifiers       | Clinical accuracy validators  |
| **Over-refusal analog** | Refusing benign prompts       | False positive detections     |

---

## Implementation Considerations

### 1. Reward Model Selection
- Need reliable classifiers for error detection, error type, and severity
- May require domain-specific medical safety models
- Consider ensemble of models for robustness

### 2. Seed Dataset Design
- Must include both error cases and error-free cases
- Error cases should span multiple error types (dosage, contraindication, etc.)
- Error-free cases should be realistic but safe

### 3. Hyperparameter Tuning
- Balance detection accuracy vs. false positive rate
- Adjust `rdetect` to control sensitivity
- Tune `rrevision` to ensure error type preservation

### 4. Hidden CoT Benefits
- Allows detector to reason through differential diagnosis
- Prevents attacker from exploiting detector's reasoning process
- May reduce false positives by encouraging careful analysis

---

## Open Questions

1. **How to handle error severity?** Should rewards scale with error severity?
2. **Multi-class vs. binary?** Paper uses binary harmful/safe; medical errors have types.
3. **Partial credit?** Should detector get partial reward for identifying error location but wrong type?
4. **Human-in-the-loop?** How to incorporate expert validation of generated cases?
5. **Catastrophic forgetting?** How to ensure detector doesn't forget rare error types?

---

## Summary

The Self-RedTeam reward design is a **carefully balanced composite system** that:
- Enforces zero-sum competition on core objectives (harmfulness, refusal)
- Shapes behavior through auxiliary rewards (format, revision faithfulness)
- Prevents trivial strategies through benign seed inclusion and conditional rewards
- Enables strategic reasoning through hidden Chain-of-Thought

The key innovation is the **conditional refusal reward**, which forces nuanced discrimination rather than blanket policies. This design could be adapted to medical error detection by reframing safety dimensions as error detection dimensions and preserving the zero-sum competitive structure.
