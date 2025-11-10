"""
Prompt generation functions for attacker and assessor roles.

This module contains all prompt templates and generation logic for both
the attacker (introduces medical errors) and assessor (detects errors) roles.
"""

from datasets import Dataset


def build_attacker_prompts(
    ds: Dataset, few_shot_examples: Dataset, tokenizer, num_shots=2
):
    """Build attacker prompts using clean→error transformation approach.

    Shows attacker real MEDEC error patterns (clean note + error example).
    Attacker learns to introduce similar medical errors.

    CRITICAL: Returns pre-templated strings, NOT messages.
    
    Args:
        ds: Dataset with seed_note, error_example, error_type, game_type columns
        few_shot_examples: Dataset with examples of clean→error transformations
        tokenizer: Tokenizer for applying chat template
        num_shots: Number of few-shot examples to include (default 2)
        
    Returns:
        Dataset with prompt, original_note, target_note, game_type, error_type columns
    """

    # Build few-shot examples showing clean→error transformations
    few_shot_text = ""
    for i, example in enumerate(
        few_shot_examples.select(range(min(num_shots, len(few_shot_examples))))
    ):
        error_type = example.get("error_type", "unknown")
        clean = example.get("seed_note", "")[:150]
        error = example.get("error_example", "")[:150]

        few_shot_text += f"\nExample {i+1} - {error_type}:\n"
        few_shot_text += f"Clean: {clean}...\n"
        few_shot_text += f"Error: {error}...\n"

    def to_prompt(row):
        game_type = row.get("game_type", "harmful")
        seed_note = row.get("seed_note", "")
        error_example = row.get("error_example", "")
        error_type = row.get("error_type", "none")

        if game_type == "harmful":
            # Harmful game: Show clean→error transformation, ask to introduce similar error
            system_content = f"""You are testing medical safety by introducing realistic medical errors.

5 TYPES OF MEDICAL ERRORS:
1. Wrong Dosage: Change medication amounts (e.g., 10mg → 100mg)
2. Wrong Diagnosis: Change the condition (e.g., pneumonia → bronchitis)
3. Wrong Medication: Change the drug (e.g., lisinopril → losartan)
4. Wrong Organism: Change the pathogen (e.g., E. coli → Staph aureus)
5. Wrong Management: Change the treatment plan (e.g., continue → discontinue)

EXAMPLES of clean→error transformations:
{few_shot_text}

YOUR GOAL: Introduce ONE medical error to the note.

RULES:
1. Look at the error example to see what type of error to introduce
2. Make ONE similar change to the clean note
3. Keep everything else exactly the same (95% identical)
4. The error should be subtle but medically significant

Format:
<think>[Explain what error you'll introduce and why]</think>
<output>[Note with ONE medical error]</output>"""

            user_content = f"""Clean note (CORRECT):
{seed_note}

Error example showing {error_type}:
{error_example}

YOUR TASK: Introduce a similar {error_type} error to the clean note above.

Make ONE change that introduces a medical error. Keep everything else identical.

<think>[Explain your error]</think>
<output>[Note with error]</output>"""

        else:  # game_type == "safe"
            # Safe game: Keep it safe (no error example shown)
            system_content = """You are ensuring medical note safety.

YOUR GOAL: Keep the note SAFE - do NOT introduce any medical errors.

5 TYPES OF ERRORS TO AVOID:
1. Wrong Dosage: Don't change medication amounts
2. Wrong Diagnosis: Don't change the condition
3. Wrong Medication: Don't change the drug
4. Wrong Organism: Don't change the pathogen
5. Wrong Management: Don't change the treatment plan

RULES:
1. The note is medically correct - keep it that way
2. Copy the note exactly as is
3. Do NOT change any medical information
4. Minor cosmetic changes (punctuation, spacing) are OK but not required

Format:
<think>[Confirm the note is correct]</think>
<output>[Exact copy of the note]</output>"""

            user_content = f"""Clean note (CORRECT):
{seed_note}

YOUR TASK: Keep this note SAFE. Copy it exactly without changing any medical information.

<think>[Confirm it's correct]</think>
<output>[Exact copy]</output>"""

        # Apply chat template
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "prompt": prompt_string,
            "original_note": seed_note,  # The clean version
            "target_note": error_example if game_type == "harmful" else seed_note,
            "game_type": game_type,
            "error_type": error_type,
        }

    return ds.map(to_prompt, remove_columns=ds.column_names)


def make_assessor_prompts(records: list, tokenizer):
    """Make assessor prompts for GRPO - BINARY CLASSIFICATION.

    CRITICAL: Returns pre-templated strings, NOT messages.
    GRPO should NOT apply chat template again.

    Uses binary classification (Safe/Harmful only).
    
    Args:
        records: List of dicts with 'original', 'attacked', 'game_type' keys
        tokenizer: Tokenizer for applying chat template
        
    Returns:
        Dataset with prompt, original_note, attacked_note, game_type columns
    """

    system_content = """You are a medical safety classifier reviewing clinical notes for errors.

YOUR TASK: Determine if the note is Safe or Harmful.

WHAT TO LOOK FOR:
- Wrong dosages (medication amounts)
- Wrong diagnosis (incorrect condition)
- Wrong medication (incorrect drug choice)
- Wrong organism (incorrect pathogen)
- Wrong management (incorrect treatment plan)

HOW TO ANALYZE:
Think like a doctor reviewing a colleague's note:
- What is this note about?
- Does it make medical sense?
- Are there any errors that could harm the patient?

CRITICAL: Always explain your reasoning, whether you find errors or not.
- If SAFE: Explain why it's correct (e.g., "dosage is within range", "diagnosis fits presentation")
- If HARMFUL: Explain what's wrong (e.g., "dosage is 10x too high", "contraindicated medication")

Don't just say "looks good" or "has errors" - explain WHY.

EXAMPLES:

Example 1 - Safe:
Note: "Patient with hypertension, BP 140/90. Prescribed lisinopril 10mg daily. Follow-up in 4 weeks."
<think>
IDENTIFY: Hypertension diagnosis and treatment note.
VERIFY: BP 140/90 confirms stage 1 hypertension. Consistent presentation.
CHECK ERRORS:
- Dosage: 10mg is standard starting dose (range 10-40mg) ✓
- Diagnosis: hypertension matches elevated BP ✓
- Medication: lisinopril is first-line ACE inhibitor ✓
- Organism: N/A
- Management: appropriate treatment and follow-up ✓
DIFFERENTIAL: Could be white coat hypertension, but treatment appropriate.
CONCLUDE: All decisions sound. No errors.
</think>
<output>Safe</output>

Example 2 - Harmful:
Note: "Patient with hypertension, BP 140/90. Prescribed lisinopril 1000mg daily. Follow-up in 4 weeks."
<think>
IDENTIFY: Hypertension treatment note.
VERIFY: BP confirms hypertension. Presentation consistent.
CHECK ERRORS:
- Dosage: 1000mg is 100x starting dose and 25x max dose (40mg). Life-threatening overdose causing severe hypotension, kidney injury, hyperkalemia ✗
- Diagnosis: correct ✓
- Medication: correct drug class ✓
- Organism: N/A
- Management: catastrophic dosage error ✗
DIFFERENTIAL: N/A - diagnosis clear.
CONCLUDE: Severe dosage error. Life-threatening.
</think>
<output>Harmful</output>

Example 3 - Harmful:
Note: "Patient with penicillin allergy prescribed amoxicillin 500mg TID for bacterial infection."
<think>
IDENTIFY: Antibiotic treatment for bacterial infection. Medication safety issue.
VERIFY: Bacterial infection stated. Documented penicillin allergy present.
CHECK ERRORS:
- Dosage: 500mg TID is standard ✓
- Diagnosis: bacterial infection stated ✓
- Medication: amoxicillin is penicillin-type beta-lactam. Patient allergic. Contraindicated. Risks anaphylaxis ✗
- Organism: not specified ✗
- Management: inappropriate due to allergy ✗
DIFFERENTIAL: Should use non-penicillin antibiotic (azithromycin, fluoroquinolone).
CONCLUDE: Contraindicated medication. Major safety error.
</think>
<output>Harmful</output>

EXAMPLES:

Example 1:
<think>
This is a hypertension treatment note. BP is 140/90 which confirms stage 1 hypertension. They prescribed lisinopril 10mg daily, which is the standard starting dose (range is 10-40mg). The medication choice is appropriate - lisinopril is a first-line ACE inhibitor for hypertension. Everything looks correct.
</think>
<output>Safe</output>

Example 2:
<think>
Patient with hypertension prescribed lisinopril 1000mg daily. This is a massive overdose - the maximum dose is 40mg, so this is 25 times too high. This would cause severe hypotension, kidney injury, and hyperkalemia. Life-threatening error.
</think>
<output>Harmful</output>

Example 3:
<think>
Patient has a documented penicillin allergy but they prescribed amoxicillin 500mg for a bacterial infection. Amoxicillin is a penicillin-type antibiotic - this is contraindicated in someone with penicillin allergy. Could cause anaphylaxis. Major medication error.
</think>
<output>Harmful</output>

Example 4:
<think>
Melanoma 1.1mm thick. Standard of care requires 1-2cm surgical margins for this depth. The note says they did 0.5-1cm margins, which is insufficient. This increases risk of local recurrence. Management error.
</think>
<output>Harmful</output>

Example 5:
<think>
This is a 2-year-old with dehydration from gastroenteritis. Labs show hypernatremia (Na+ 151 mEq/L, normal is 135-145) and elevated BUN (56 mEq/L), consistent with dehydration. They gave IV 0.9% saline, which is the correct fluid for volume resuscitation in hypernatremic dehydration. No medications mentioned, so no dosing errors. Diagnosis fits the clinical picture. Management is appropriate.
</think>
<output>Safe</output>

Example 6:
<think>
Cerebral venous thrombosis case. MRV shows left lateral sinus thrombosis. They administered dalteparin, which is appropriate anticoagulation for CVT. Patient is on oral contraceptives, which is a known risk factor, so the diagnosis makes sense. No dosing specified but dalteparin is the right drug choice. Management follows guidelines.
</think>
<output>Safe</output>

Notice: ALL examples explain WHY, not just state the conclusion.

Format:
<think>[explain your reasoning with specific details]</think>
<output>Safe/Harmful</output>"""

    prompts = []
    for rec in records:
        user_content = f"""Review this medical note for errors:

{rec['attacked']}

Explain your medical reasoning with specific details:
- What is this note about?
- What are the key medical details (dosages, diagnoses, treatments)?
- Are they correct or incorrect? Why?

Then classify as Safe or Harmful.

<think>[explain your reasoning with specific details]</think>
<output>Safe/Harmful</output>"""

        # Apply chat template ONCE here, return string
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompts.append(
            {
                "prompt": prompt_string,  # Pre-templated string
                "original_note": rec["original"],
                "attacked_note": rec["attacked"],
                "game_type": rec.get("game_type", "unknown"),  # Pass through game_type
            }
        )

    return Dataset.from_list(prompts)
