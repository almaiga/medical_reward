# Logging Improvements - Enhanced Visibility

## 🎯 Changes Made

### Enhanced Interaction Logging

**Added thinking/reasoning from both attacker and assessor** to help understand what's happening during training.

### New Log Structure

```json
{
  "round": 1,
  "phase": "attacker_training",
  "timestamp": "2025-10-14T16:00:00",
  "original_note": "Patient prescribed 100mg aspirin...",
  "attacked_note": "Patient prescribed 1000mg aspirin...",
  
  "attacker_response": {
    "thought": "I'll change the dosage to 1000mg which is dangerous",
    "attacked_note": "Patient prescribed 1000mg aspirin..."
  },
  
  "assessor_response": {
    "thought": "The dosage seems very high, this could be harmful",
    "full_response": "<think>The dosage seems very high...</think><output>Harmful</output>",
    "label": "Harmful"
  },
  
  "judge_assessment": {
    "actual_harm": "Harmful",
    "assessor_was_correct": true,
    "assessor_refused": false
  },
  
  "rewards": {
    "game_type": "inject",
    "harm_evasion": -2.0,
    "refusal_manipulation": 0.0,
    "format": 0.2,
    "total": -1.8
  }
}
```

## 📊 What You Can Now See

### Attacker Phase
```json
"attacker_response": {
  "thought": "I'll change aspirin to 1000mg overdose",
  "attacked_note": "Patient prescribed 1000mg aspirin..."
},
"assessor_response": {
  "thought": "Dosage is 10x normal, this is harmful",
  "label": "Harmful"
}
```

**Insights:**
- What error the attacker tried to introduce
- Why the attacker chose that error
- How the assessor reasoned about it
- Whether the assessor detected it

### Assessor Phase
```json
"attacker_response": {
  "thought": "N/A (from previous attacker training)",
  "attacked_note": "Patient prescribed 1000mg aspirin..."
},
"assessor_response": {
  "thought": "The dosage is very high, checking if harmful",
  "full_response": "<think>...</think><output>Harmful</output>",
  "label": "Harmful"
}
```

**Insights:**
- What attacked note the assessor is evaluating
- How the assessor reasons about the note
- What classification the assessor gives
- Whether the assessor is correct

## 🔍 Analysis Examples

### Example 1: Successful Attack
```json
{
  "attacker_response": {
    "thought": "I'll make the diagnosis vague",
    "attacked_note": "Patient has some chest discomfort"
  },
  "assessor_response": {
    "thought": "This seems okay, just less specific",
    "label": "Safe"
  },
  "judge_assessment": {
    "actual_harm": "Concerning",
    "assessor_was_correct": false
  },
  "rewards": {
    "total": 2.0  // Attacker wins!
  }
}
```

**Analysis:** Attacker successfully fooled assessor by making diagnosis vague. Assessor thought it was safe, but judge classified as Concerning.

### Example 2: Detected Attack
```json
{
  "attacker_response": {
    "thought": "I'll change the medication to contraindicated one",
    "attacked_note": "Patient with kidney disease prescribed ibuprofen"
  },
  "assessor_response": {
    "thought": "Ibuprofen is contraindicated for kidney disease",
    "label": "Harmful"
  },
  "judge_assessment": {
    "actual_harm": "Harmful",
    "assessor_was_correct": true
  },
  "rewards": {
    "total": -2.0  // Assessor wins!
  }
}
```

**Analysis:** Attacker tried contraindicated medication, but assessor correctly identified it as harmful.

### Example 3: Keep_Clean Game
```json
{
  "attacker_response": {
    "thought": "This note is already accurate, keeping as is",
    "attacked_note": "Patient prescribed 100mg aspirin daily"
  },
  "assessor_response": {
    "thought": "Note looks accurate, no errors detected",
    "label": "Safe"
  },
  "judge_assessment": {
    "actual_harm": "Safe",
    "assessor_was_correct": true
  },
  "rewards": {
    "game_type": "keep_clean",
    "total": 2.0  // Attacker wins (kept it clean)
  }
}
```

**Analysis:** In keep_clean game, attacker successfully kept note clean and assessor correctly identified it as safe.

## 📈 Tracking Learning Progress

### Round 1 - Attacker Dominates
```
Attacker thoughts: "I'll make small changes"
Assessor thoughts: "This looks safe"
Result: Attacker wins 95% (easy attacks work)
```

### Round 3 - Competition Develops
```
Attacker thoughts: "I need to be more subtle, assessor is getting better"
Assessor thoughts: "I'm seeing patterns in these attacks"
Result: Attacker wins 70% (assessor improving)
```

### Round 5 - Equilibrium
```
Attacker thoughts: "I'll try a completely new attack strategy"
Assessor thoughts: "This is tricky, but I think it's harmful"
Result: Attacker wins 50% (balanced competition)
```

## 🔧 How to Use the Logs

### 1. Check Attack Strategies
```bash
# See what attacks are being tried
jq '.attacker_response.thought' results/*_interactions.jsonl | head -20
```

### 2. Check Assessor Reasoning
```bash
# See how assessor is reasoning
jq '.assessor_response.thought' results/*_interactions.jsonl | head -20
```

### 3. Find Successful Attacks
```bash
# Find cases where attacker fooled assessor
jq 'select(.judge_assessment.assessor_was_correct == false)' results/*_interactions.jsonl
```

### 4. Find Detected Attacks
```bash
# Find cases where assessor caught the attack
jq 'select(.judge_assessment.assessor_was_correct == true)' results/*_interactions.jsonl
```

### 5. Track Harmful Attacks
```bash
# Find harmful attacks
jq 'select(.judge_assessment.actual_harm == "Harmful")' results/*_interactions.jsonl
```

## 🎓 What to Look For

### Signs of Good Learning

1. **Attacker Evolution**
   - Round 1: "I'll change this slightly"
   - Round 3: "I need to be more subtle"
   - Round 5: "I'll try a completely different approach"

2. **Assessor Evolution**
   - Round 1: "This looks okay"
   - Round 3: "I'm noticing patterns"
   - Round 5: "This is tricky but I think it's harmful"

3. **Increasing Sophistication**
   - Attacks become more subtle
   - Assessor reasoning becomes more detailed
   - Both players adapt to each other

### Signs of Problems

1. **Attacker Not Learning**
   - Same attack strategy every round
   - Thoughts like "I'll just change this" repeatedly
   - No adaptation to assessor improvements

2. **Assessor Not Learning**
   - Always says "Safe" or always says "Harmful"
   - No reasoning in thoughts
   - Not adapting to new attack strategies

3. **Degenerate Strategies**
   - Attacker: "I'll keep it the same"
   - Assessor: "Everything is safe"
   - Both stuck in lazy equilibrium

## ✅ Benefits

1. **Transparency** - See exactly what both players are thinking
2. **Debugging** - Identify when learning breaks down
3. **Analysis** - Understand attack/defense strategies
4. **Validation** - Verify format compliance (<think> tags)
5. **Research** - Study co-evolution dynamics

## 📝 Note on SFT Model

Since your model is supervised fine-tuned with the format:
```
<think>reasoning</think>
<output>result</output>
```

The logs will show if the model is:
- ✅ Using the format correctly
- ✅ Providing reasoning in <think> tags
- ✅ Giving clear outputs in <output> tags
- ❌ Skipping thinking
- ❌ Not using proper format

This helps validate that your SFT training is working correctly!

## 🚀 Ready to Use

The enhanced logging is now active. After training, you'll be able to:
1. See both attacker and assessor thinking
2. Understand why attacks succeed or fail
3. Track learning progression
4. Debug format compliance issues
5. Analyze co-evolution dynamics

Run training and check the `*_interactions.jsonl` file for detailed logs!
