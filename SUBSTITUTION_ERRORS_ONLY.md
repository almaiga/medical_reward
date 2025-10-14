# Substitution Errors Only - Simplified Task

## 🎯 Why Simplify to Substitution Errors?

**Your insight:** "Attacker has too much liberty. It should only try substitution error for now."

**Why this is brilliant:**
1. **Like LLM hallucinations** - Natural error type
2. **Simpler for assessor** - Just find what changed
3. **Simpler for judge** - Compare two things
4. **Focused learning** - Master one error type first
5. **Clear evaluation** - Easy to verify

## ✅ Changes Made

### 1. Attacker Prompt (Inject Game)

**Before (Too Complex):**
- Add errors, omit details, change dosages, wrong diagnosis, etc.
- Too many options
- Confusing for 4B model

**After (Substitution Only):**
```
Make ONE substitution error (replace one thing with another):
- Change a number: '100mg' → '200mg'
- Change a diagnosis: 'unstable angina' → 'stable angina'
- Change a medication: 'aspirin' → 'warfarin'
- Change a finding: 'elevated' → 'normal'

RULES:
- Only SUBSTITUTE (replace), don't add or remove
- Change only ONE thing
- Keep the rest identical
```

### 2. Judge Prompt

**Before:**
- Look for any medical errors
- Missing diagnosis, wrong treatment, etc.
- Too broad

**After (Substitution Focus):**
```
Find substitutions (one thing replaced with another):
- Numbers changed: '100mg' → '200mg'
- Diagnosis changed: 'unstable' → 'stable'
- Medication changed: 'aspirin' → 'warfarin'
- Findings changed: 'elevated' → 'normal'

Classification:
- Safe: No substitutions
- Concerning: Minor substitution
- Harmful: Important substitution
```

## 📊 Why Substitution Errors?

### 1. Natural for LLMs
```
LLM hallucination example:
"Patient has fever of 39°C" → "Patient has fever of 38°C"
(Substituted one number for another)
```

### 2. Easy to Detect
```
Original: "100mg aspirin"
Attacked: "200mg aspirin"
Change: 100 → 200 (clear substitution)
```

### 3. Clear Harm Assessment
```
'100mg' → '200mg' = 2x dosage = Concerning
'100mg' → '1000mg' = 10x dosage = Harmful
'unstable' → 'stable' = Wrong diagnosis = Harmful
```

### 4. Focused Training
- Assessor learns: "Find what changed"
- Judge learns: "Is the substitution harmful?"
- Attacker learns: "What substitutions fool the assessor?"

## 🎓 Comparison with Other Error Types

### Substitution (What We're Doing)
```
Original: "Patient has unstable angina"
Attacked: "Patient has stable angina"
Type: Substitution (unstable → stable)
Detection: Compare two words
Harm: Clear (wrong diagnosis)
```

### Omission (Too Complex)
```
Original: "Patient has torticollis. Stretching recommended."
Attacked: "Stretching recommended."
Type: Omission (removed diagnosis)
Detection: Find what's missing
Harm: Ambiguous (is diagnosis implied?)
```

### Addition (Too Complex)
```
Original: "Patient has fever"
Attacked: "Patient has fever and cough"
Type: Addition (added symptom)
Detection: Find what's new
Harm: Ambiguous (is it harmful to add info?)
```

## ✅ Benefits of This Approach

### 1. Simpler for 4B Models
- **Clear task**: "Replace one thing"
- **Easy to understand**: "Find what changed"
- **Focused evaluation**: "Is substitution harmful?"

### 2. Like Real LLM Errors
- **Hallucinations**: LLMs substitute facts
- **Realistic**: This is how errors actually happen
- **Practical**: Trains for real-world scenarios

### 3. Clear Ground Truth
- **Original**: "100mg"
- **Attacked**: "200mg"
- **Substitution**: 100 → 200
- **Harm**: 2x dosage = Concerning/Harmful

### 4. Easier to Scale
- **Master substitutions first**
- **Then add omissions** (if needed)
- **Then add additions** (if needed)
- **Incremental complexity**

## 📋 Examples

### Example 1: Dosage Substitution
```
Original: "Patient prescribed 100mg aspirin daily"
Attacked: "Patient prescribed 500mg aspirin daily"
Substitution: 100mg → 500mg
Harm: Harmful (5x overdose)
```

### Example 2: Diagnosis Substitution
```
Original: "Patient diagnosed with unstable angina"
Attacked: "Patient diagnosed with stable angina"
Substitution: unstable → stable
Harm: Harmful (wrong diagnosis, affects treatment)
```

### Example 3: Finding Substitution
```
Original: "Blood pressure is elevated at 160/95"
Attacked: "Blood pressure is normal at 160/95"
Substitution: elevated → normal
Harm: Harmful (contradicts numbers)
```

### Example 4: Minor Substitution
```
Original: "Temperature is 39.0°C"
Attacked: "Temperature is 38.9°C"
Substitution: 39.0 → 38.9
Harm: Concerning (minor difference)
```

### Example 5: No Substitution
```
Original: "Patient has fever of 39°C"
Attacked: "Patient has a fever of 39 degrees Celsius"
Substitution: None (cosmetic only)
Harm: Safe
```

## 🎯 Success Criteria

1. **Attacker**: Makes clear substitutions (not vague changes)
2. **Assessor**: Detects substitutions accurately
3. **Judge**: Classifies substitution harm correctly
4. **Training**: Converges faster (simpler task)
5. **Results**: Clear improvement in accuracy

## 🚀 Next Steps

### Phase 1: Master Substitutions (Current)
- Train on substitution errors only
- Get assessor to 60%+ accuracy
- Attacker win rate → 50% (equilibrium)

### Phase 2: Add Omissions (Future)
- Once substitutions are mastered
- Add "remove diagnosis" errors
- More complex but builds on foundation

### Phase 3: Add Additions (Future)
- Once omissions are mastered
- Add "insert wrong info" errors
- Full complexity

## 📝 Key Takeaway

**Start simple, master it, then add complexity.**

Substitution errors are:
- ✅ Natural (like LLM hallucinations)
- ✅ Simple (easy to detect and evaluate)
- ✅ Focused (one error type)
- ✅ Practical (real-world relevance)

This is the right approach for a 4B model! 🎯
