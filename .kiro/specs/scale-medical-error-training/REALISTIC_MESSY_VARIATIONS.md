# Realistic Messy Variations: Critical Design Decision

## The Problem We Discovered

Your adaptation data had **generic template-based reasoning** that was causing your model to learn lazy patterns like "I'll alter management plan" instead of real medical analysis.

## Root Cause Analysis

### Self-RedTeam vs Your Use Case

**Self-RedTeam (LLM Safety):**
- Clean, well-formed prompts
- Goal: Distinguish malicious intent from benign requests
- "Deceptive-looking benign": Make benign prompts look suspicious to test over-refusal

**Your Use Case (Medical Error Detection):**
- MEDEC data is TOO CLEAN (research-quality, standardized)
- Real-world clinical notes are MESSY (abbreviations, typos, formatting issues)
- Goal: Detect medical ERRORS, not formatting/style variations

## The Critical Insight

**Real clinical documentation is messy:**
- Abbreviations (BP, HR, Temp, RR, etc.)
- Inconsistent formatting
- Typos and shorthand
- Non-standard terminology
- Rushed documentation

**Your model needs to learn:**
- Focus on MEDICAL errors, not presentation quality
- Handle real-world variability
- Distinguish "poorly written" from "medically wrong"

## The Solution: Realistic Messy Variations

### Attacker-Safe Strategy (Updated)

**OLD (Template-based):**
```
Thinking: "This note is already clean and accurate. I'll output it unchanged."
Output: [exact copy of clean note]
```

**NEW (GPT-5 generated realistic variations):**
```
Thinking: "I'll introduce realistic formatting variations and abbreviations commonly 
seen in clinical notes: use 'Temp' instead of 'temperature', 'BP' for blood pressure, 
vary spacing and punctuation. These changes make it look like real clinical documentation 
but don't affect medical accuracy."

Output: "65M presents w/ 2d lethargy/confusion. Wife reports 5d nausea + polyuria, 
1wk cough. PMH: CVA 3y ago, HTN 10y. Meds: lisinopril, ASA. VS: Temp 38.5C, HR 114, 
RR 15, BP 108/75. Somnolent, oriented x1. Dry MM, poor turgor. Crackles L base..."
```

### All 4 Example Types Now Use GPT-5

**1. Attacker-Harmful** (was already GPT-5):
- Generate medical errors with detailed reasoning
- Explain what changed and why it's dangerous

**2. Attacker-Safe** (NOW GPT-5):
- Generate realistic messy variations
- Use abbreviations, formatting changes, shorthand
- Keep ALL medical information accurate
- Explain what variations were made

**3. Assessor-Harmful** (was already GPT-5):
- Detailed analysis of medical errors
- Systematic review of what's wrong

**4. Assessor-Safe** (NOW GPT-5):
- Detailed systematic review
- Explain what was checked (meds, dosages, diagnoses)
- Confirm no medical errors despite messiness

## Implementation Changes

### Updated `generate_sft_data.py`

**Vanilla/Safe prompt:**
```python
"""Create a realistic variation of this medical note that mimics real-world clinical documentation:

Transform it to look like a real clinician wrote it quickly:
- Use medical abbreviations (BP, HR, Temp, RR, etc.)
- Vary formatting (spacing, punctuation, capitalization)
- Use shorthand where appropriate (↑, ↓, +, -, etc.)
- Reorder information naturally
- Keep ALL medical information accurate (no medical errors)"""
```

### Updated `generate_game_format_adaptation.py`

**Added GPT-5 functions:**
- `generate_gpt_attacker_safe_reasoning()` - Realistic messy variations
- `generate_gpt_assessor_safe_reasoning()` - Detailed safe analysis

**Updated generation loop:**
- ALL 4 types now use GPT-5 when `--use_gpt` flag is set
- Total API calls: 4 per note (was 2 per note)

## Cost & Time Impact

### Adaptation Generation (299 notes)

**OLD (2 GPT calls per note):**
- Harmful examples: GPT-5
- Safe examples: Templates
- Total: 598 API calls
- Time: ~3-4 hours
- Cost: ~$0.10-0.15

**NEW (4 GPT calls per note):**
- ALL examples: GPT-5
- Total: 1,196 API calls
- Time: ~6-8 hours
- Cost: ~$0.20-0.30

### Educational Generation (Already Done)

Your educational data needs the same fix for vanilla examples, but you already generated it. We'll need to regenerate with the updated prompts.

## Benefits

1. **Realistic Training Data**: Model learns to handle messy real-world notes
2. **Better Discrimination**: Focuses on medical errors, not formatting
3. **Consistent Quality**: All examples have detailed GPT-5 reasoning
4. **Production Ready**: Prepares model for actual clinical deployment

## Next Steps

1. ✅ Scripts updated with realistic messy variation prompts
2. ⏳ Regenerate adaptation data (299 notes with --use_gpt)
3. ⏳ Regenerate educational data (522 notes with updated vanilla prompts)
4. ⏳ Merge and validate all data
5. ⏳ Train with consistent high-quality data

## Example Transformations

**Clean → Realistic Messy:**

```
CLEAN:
"His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, 
and blood pressure is 98/62 mm Hg."

MESSY:
"Temp 38.5C, HR 96, RR 26, BP 98/62"
```

```
CLEAN:
"Patient has no cough, hoarseness, or rhinorrhea"

MESSY:
"No cough/hoarseness/rhinorrhea noted"
```

```
CLEAN:
"Examination shows increased fremitus and bronchial breath sounds"

MESSY:
"Exam: ↑ fremitus, bronchial BS"
```

These variations teach the model that messy formatting ≠ medical error!
