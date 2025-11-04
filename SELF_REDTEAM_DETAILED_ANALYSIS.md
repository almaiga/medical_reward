# Self-RedTeam Paper - Detailed Implementation Analysis

## Source
Based on actual code from: https://github.com/mickelliu/selfplay-redteaming

## Core Game Structure

### Two Game Types (NOT just harmful!)

The paper uses **TWO types of games**, not one:

1. **Harmful Game** (`generated_harmful`):
   - Seed: A harmful prompt (e.g., "How to make a bomb")
   - Attacker goal: Refine/jailbreak it to bypass defender
   - Defender goal: Refuse the harmful request

2. **Benign Game** (`generated_benign`):
   - Seed: A benign prompt (e.g., "Tell me about Paris")
   - Attacker goal: Keep it benign (don't make it harmful)
   - Defender goal: Comply with the benign request

### Why Two Game Types?

This prevents the defender from becoming over-cautious and refusing everything!
- If only harmful prompts exist → defender learns to refuse everything
- With benign prompts → defender must learn to distinguish and comply when appropriate

## Reward Design - GENERAL SUM (Not Zero-Sum!)

### Key Insight: The paper uses "general_sum" NOT "zero_sum"

From the code:
```python
REWARD_COEFF_CONFIG = {
    "general_sum_all_one": {
        "attacker": {
            "prompt_harmfulness": (-1.0, 1.0),      # RA,revision
            "response_harmfulness": (-1.0, 1.0),    # RA,harmfulness
            "response_refusal": (-1.0, 1.0)         # RA,refusal
        },
        "defender": {
            "response_harmfulness": (-1.0, 1.0),    # RD,harmfulness
            "response_refusal": (-1.0, 1.0)         # RD,refusal
        }
    }
}
```

### Attacker Rewards (3 components):

#### 1. Prompt Harmfulness Reward (RA,revision)
**Purpose**: Ensure attacker respects the seed type (faithful revision)

**Harmful seed** (`generated_harmful`):
- Prompt is harmful → +1.0 (faithful: kept it harmful)
- Prompt is unharmful → -1.0 (unfaithful: made it safe)

**Benign seed** (`generated_benign`):
- Prompt is unharmful → +1.0 (faithful: kept it benign)
- Prompt is harmful → -1.0 (unfaithful: made it harmful)

**This prevents reward hacking!** Attacker can't just generate random harmful prompts for benign seeds.

#### 2. Response Refusal Reward (RA,refusal)
**Purpose**: Reward attacker for making defender refuse/comply incorrectly

**Harmful seed**:
- Defender refuses → -1.0 (attacker lost: defender correctly refused)
- Defender complies → +1.0 (attacker won: defender failed to refuse)

**Benign seed**:
- Defender refuses → +1.0 (attacker won: defender over-refused)
- Defender complies → -1.0 (attacker lost: defender correctly complied)

#### 3. Response Harmfulness Reward (RA,harmfulness)
**Purpose**: Reward attacker for eliciting harmful responses

**Both harmful and benign seeds**:
- Response is harmful → +1.0 (attacker won: broke defender)
- Response is unharmful → -1.0 (attacker lost: defender stayed safe)

**Total Attacker Reward**: RA,revision + RA,refusal + RA,harmfulness
- Range: -3.0 to +3.0

### Defender Rewards (2 components):

#### 1. Response Refusal Reward (RD,refusal)
**Purpose**: Reward defender for refusing harmful, complying with benign

**If prompt is harmful**:
- Defender refuses → +1.0 (correct refusal)
- Defender complies → -1.0 (wrong refusal)

**If prompt is benign**:
- Defender complies → +1.0 (correct compliance)
- Defender refuses → -1.0 (over-refusal)

#### 2. Response Harmfulness Reward (RD,harmfulness)
**Purpose**: Reward defender for always generating safe responses

**Always**:
- Response is unharmful → +1.0 (successful defense)
- Response is harmful → -1.0 (broken defense)

**Total Defender Reward**: RD,refusal + RD,harmfulness
- Range: -2.0 to +2.0

### Format Reward (Both players):
- Correct CoT format → +1.0
- Incorrect format → -1.0

## Judge Model (WildGuard)

The judge evaluates THREE things:

1. **Prompt Harmfulness**: Is the attacker's prompt harmful or unharmful?
2. **Response Refusal**: Did the defender refuse or comply?
3. **Response Harmfulness**: Is the defender's response harmful or unharmful?

These three labels are used to compute all rewards.

## Training Pipeline

### One Round of Self-Play:

1. **Attacker Phase**:
   - Input: Seed prompts (50% harmful, 50% benign)
   - Attacker generates adversarial prompts
   - Frozen defender responds to them
   - Judge evaluates: prompt_harmfulness, response_refusal, response_harmfulness
   - Compute attacker rewards (3 components + format)
   - Update attacker via GRPO

2. **Defender Phase**:
   - Input: Adversarial prompts from attacker training
   - Defender generates responses
   - Judge evaluates: response_refusal, response_harmfulness
   - Compute defender rewards (2 components + format)
   - Update defender via GRPO

3. **Repeat**: Alternate between attacker and defender training

## Hidden Chain-of-Thought

### Format:
```
<think>private reasoning</think><answer>public output</answer>
```

### Key Points:
- Thinking is HIDDEN from the other player
- Only the answer is shown to opponent
- Both players get format reward for correct CoT structure
- Prepended with `<think>` in prompt, model completes with `</think><answer>...</answer>`

## Key Differences from Our Implementation

### ❌ What We Got Wrong:

1. **Only one game type**: We only had "harmful" game
   - Paper has: harmful + benign (prevents over-refusal)

2. **Wrong reward structure**: We tried to make it zero-sum
   - Paper uses: general_sum with 3 components for attacker, 2 for defender

3. **Missing prompt harmfulness reward**: We didn't have RA,revision
   - Paper uses: This to prevent reward hacking (faithfulness check)

4. **Different reward scales**: We had asymmetric scales
   - Paper uses: Consistent ±1.0 for all components

5. **Wrong game mechanics**: We thought attacker modifies notes
   - Paper: Attacker generates adversarial prompts, defender responds to them

## Our Medical Error Detection Adaptation

### How to adapt Self-RedTeam to medical errors:

#### Game Types:

1. **Harmful Game** (introduce error):
   - Seed: Clean medical note
   - Attacker: Introduce medical error
   - Defender: Classify as Harmful
   - Judge: Check if error was introduced AND if defender caught it

2. **Safe Game** (keep safe):
   - Seed: Clean medical note
   - Attacker: Keep it safe (no changes)
   - Defender: Classify as Safe
   - Judge: Check if note stayed safe AND if defender correctly said Safe

#### Reward Components:

**Attacker** (3 components):
1. **Note Modification Faithfulness** (RA,revision):
   - Harmful game: Introduced error → +1, No error → -1
   - Safe game: No error → +1, Introduced error → -1

2. **Defender Classification** (RA,refusal equivalent):
   - Harmful game: Defender says Harmful → -1, Safe → +1
   - Safe game: Defender says Safe → -1, Harmful → +1

3. **Actual Harm Level** (RA,harmfulness equivalent):
   - Both games: Note has error → +1, No error → -1

**Defender** (2 components):
1. **Classification Correctness** (RD,refusal equivalent):
   - If note has error: Says Harmful → +1, Says Safe → -1
   - If note is safe: Says Safe → +1, Says Harmful → -1

2. **Always Detect Errors** (RD,harmfulness equivalent):
   - Correctly classified → +1, Incorrectly classified → -1

## Summary: What Makes Self-RedTeam Work

1. **Two game types** (harmful + benign) prevent over-refusal
2. **Faithfulness reward** (RA,revision) prevents reward hacking
3. **Multiple reward components** provide rich learning signal
4. **General sum** (not zero-sum) allows both players to learn
5. **Hidden CoT** enables private reasoning
6. **WildGuard judge** provides reliable ground truth

## Action Items for Our Code

1. ✅ Keep two game types (harmful + safe)
2. ❌ Remove zero-sum constraint
3. ✅ Add faithfulness reward (RA,revision) - we have this!
4. ❌ Add response classification reward (RA,refusal equivalent)
5. ❌ Add actual harm reward (RA,harmfulness equivalent)
6. ✅ Keep format reward
7. ❌ Use consistent ±1.0 scales for all components
8. ✅ Use judge to evaluate all three aspects

## Conclusion

Our implementation was partially correct but missing key components:
- ✅ We have two game types (harmful/safe)
- ✅ We have faithfulness check
- ❌ We don't have the 3-component attacker reward structure
- ❌ We don't have the 2-component defender reward structure
- ❌ We tried to force zero-sum when paper uses general-sum

The paper's "general_sum_all_one" config is the key - it's NOT zero-sum, but has multiple reward components that together create the right incentives.


## CORRECTION: Format Reward in Self-RedTeam

### What the Paper Actually Says

I need to correct my earlier analysis. The paper DOES give format rewards:

> **RP,format (CoT Formatting Sub-Reward)**: This sub-reward ensures that both agents, A and D, adhere to the correct CoT format. A reward of +rformat is given if the agent's output can be correctly parsed into distinct reasoning (yCoT_P) and answer (yP) components, and −rformat otherwise.

### The Actual Reward Structure

**Game Outcome Rewards (Zero-Sum):**
```
RA,res_harm = -RD,res_harm
RA,res_refusal = -RD,res_refusal
```

**Total Rewards (NOT Zero-Sum due to shaping):**
```
Attacker: RA = RA,res_harm + RA,res_refusal + RA,format + RA,revision
Defender: RD = RD,res_harm + RD,res_refusal + RD,format

RA + RD ≠ 0 (because RA,revision only for attacker)
```

### Why Format Reward Doesn't Cause Lazy Thinking in Self-RedTeam

**1. Parsing Requirement (Structural Check)**
- Must have valid `<think>...</think>` structure
- Must have valid `<answer>...</answer>` structure  
- Both must be parseable
- This is syntax checking, not content quality checking

**2. Format is Small Proportion of Total Reward**
```
Attacker total: rharm + rrefusal + rformat + rrevision
              = ±1.0  ± 1.0     ± 1.0    ± 1.0
              = -4.0 to +4.0

Format is 1/4 of total reward
```

**3. Hidden CoT Separates Thinking from Action**
```
<think>private reasoning</think>  ← Gets format reward, hidden from opponent
<answer>actual move</answer>       ← Affects game outcome, visible to opponent
```

The thinking gets format reward but doesn't directly affect game outcome. The answer affects game outcome but is separate from thinking.

**4. Game Outcome Rewards Dominate**
Even with lazy thinking that gets +rformat, if the answer is bad:
- Bad answer → opponent exploits it
- Game outcome: -rharm - rrefusal = -2.0
- Net: +1.0 (format) - 2.0 (game) = -1.0

Lazy thinking doesn't pay off because game outcomes matter more.

### Why YOUR System Has Lazy Thinking (The Real Reason)

**The Critical Difference:**

| Aspect | Self-RedTeam | Your System |
|--------|--------------|-------------|
| **Thinking content** | Strategic planning | Medical analysis |
| **Answer content** | The actual move | Just a label |
| **Thinking visibility** | Hidden from opponent | Part of output |
| **Thinking impact** | Indirect (helps answer) | Direct (IS the analysis) |
| **Format proportion** | 1/4 of attacker, 1/3 of defender | 1/3 of assessor |

**In Self-RedTeam:**
- Thinking = private planning
- Answer = the actual adversarial prompt/response
- Lazy thinking → bad answer → bad game outcome

**In Your System:**
- Thinking = the medical analysis itself
- Output = just the classification label
- Lazy thinking → might still get correct label by luck

### What This Means for Your Implementation

**Your current approach is actually CORRECT per the paper!**

1. ✅ Format reward is appropriate (paper uses it)
2. ✅ Multi-component rewards (paper uses them)
3. ✅ Quality-gating format reward (stricter than paper)

**The lazy thinking issue is NOT because of format reward.**

It's because:
1. Your 4B model is small and takes shortcuts
2. The thinking contains the actual analysis (not just planning)
3. Classification can sometimes be correct even with lazy thinking

**Solutions that align with the paper:**

1. **Stricter Parsing (What Paper Does)**
   - Check that `<think>` is not empty
   - Check that `<think>` contains actual content (not just meta-commentary)
   - Check that `<output>` is valid classification

2. **Increase Game Outcome Weight (What Paper Implies)**
   - Make classification accuracy worth more
   - Make format worth less
   - Example: rharm=2.0, rrefusal=2.0, rformat=1.0

3. **Better Prompting (What Paper Does)**
   - Show examples of good thinking
   - Show examples of bad thinking
   - Explicitly forbid lazy meta-commentary

**What you've already implemented (quality-gated format reward) is actually BETTER than what the paper does!** The paper only checks parsing, you check content quality too.

The lazy thinking you're seeing is likely:
1. Model capacity limitations (4B is small)
2. Early training dynamics (will improve over rounds)
3. Need for better few-shot examples in prompts
