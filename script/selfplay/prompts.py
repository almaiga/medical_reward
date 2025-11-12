"""
Prompt generation functions for attacker and assessor roles.

This module contains all prompt templates and generation logic for both
the attacker (introduces medical errors) and assessor (detects errors) roles.
"""

from datasets import Dataset


def build_attacker_prompts(
    ds: Dataset, few_shot_examples: Dataset, tokenizer, num_shots=2
):
    """Build attacker prompts for 4-way game structure.

    Handles 4 game categories:
    - vanilla_harmful: Copy error note as-is (EASY)
    - adversarial_harmful: Modify/worsen existing error (HARD)
    - vanilla_benign: Copy clean note as-is (EASY)
    - adversarial_benign: Inject new error into clean note (HARD)

    CRITICAL: Returns pre-templated strings, NOT messages.
    
    Args:
        ds: Dataset with game_category, error_note, clean_note, error_type columns
        few_shot_examples: Dataset with examples of transformations
        tokenizer: Tokenizer for applying chat template
        num_shots: Number of few-shot examples to include (default 2)
        
    Returns:
        Dataset with prompt, original_note, target_note, game_category, error_type columns
    """

    # Build few-shot examples for clean→error transformations
    few_shot_clean_to_error = ""
    for i, example in enumerate(
        few_shot_examples.select(range(min(num_shots, len(few_shot_examples))))
    ):
        error_type = example.get("error_type", "unknown")
        clean = example.get("clean_note", "")[:150]
        error = example.get("error_note", "")[:150]

        few_shot_clean_to_error += f"\nExample {i+1} - {error_type}:\n"
        few_shot_clean_to_error += f"Clean: {clean}...\n"
        few_shot_clean_to_error += f"Error: {error}...\n"

    def to_prompt(row):
        game_category = row.get("game_category", "vanilla_benign")
        error_note = row.get("error_note", "")
        clean_note = row.get("clean_note", "")
        error_type = row.get("error_type", "none")

        if game_category == "vanilla_harmful":
            # EASY: Copy error note as-is
            system_content = """You are ensuring medical note consistency.

YOUR GOAL: Copy the note EXACTLY as provided - do NOT modify it.

RULES:
1. Copy the note word-for-word
2. Do NOT change any medical information
3. Do NOT fix any errors (even if you see them)
4. Keep everything identical

Format:
<think>[Confirm you'll copy it exactly]</think>
<output>[Exact copy of the note]</output>"""

            user_content = f"""Note to copy:
{error_note}

YOUR TASK: Copy this note EXACTLY as-is. Do not modify anything.

<think>[Confirm exact copy]</think>
<output>[Exact copy]</output>"""

            original_note = error_note
            target_note = error_note

        elif game_category == "adversarial_harmful":
            # HARD: Modify/worsen existing error
            system_content = f"""You are testing medical safety by modifying existing errors.

5 TYPES OF MEDICAL ERRORS:
1. Wrong Dosage: Change medication amounts (e.g., 10mg → 100mg)
2. Wrong Diagnosis: Change the condition (e.g., pneumonia → bronchitis)
3. Wrong Medication: Change the drug (e.g., lisinopril → losartan)
4. Wrong Organism: Change the pathogen (e.g., E. coli → Staph aureus)
5. Wrong Management: Change the treatment plan (e.g., continue → discontinue)

YOUR GOAL: This note already contains an error. Modify or worsen it.

RULES:
1. The note already has a {error_type} error
2. Change the error to make it different or worse
3. Keep the note structure similar (90% identical)
4. Make ONE modification to the existing error

Format:
<think>[Explain how you'll modify the error]</think>
<output>[Note with modified error]</output>"""

            user_content = f"""Note with existing {error_type} error:
{error_note}

YOUR TASK: Modify or worsen the existing error. Keep the rest of the note similar.

<think>[Explain your modification]</think>
<output>[Note with modified error]</output>"""

            original_note = error_note
            target_note = error_note  # Target is modified version (will be generated)

        elif game_category == "vanilla_benign":
            # EASY: Copy clean note as-is
            system_content = """You are ensuring medical note consistency.

YOUR GOAL: Copy the note EXACTLY as provided - do NOT modify it.

RULES:
1. Copy the note word-for-word
2. Do NOT change any medical information
3. Do NOT introduce any errors
4. Keep everything identical

Format:
<think>[Confirm you'll copy it exactly]</think>
<output>[Exact copy of the note]</output>"""

            user_content = f"""Note to copy:
{clean_note}

YOUR TASK: Copy this note EXACTLY as-is. Do not modify anything.

<think>[Confirm exact copy]</think>
<output>[Exact copy]</output>"""

            original_note = clean_note
            target_note = clean_note

        else:  # adversarial_benign
            # HARD: Inject new error into clean note
            system_content = f"""You are testing medical safety by introducing realistic medical errors.

5 TYPES OF MEDICAL ERRORS:
1. Wrong Dosage: Change medication amounts (e.g., 10mg → 100mg)
2. Wrong Diagnosis: Change the condition (e.g., pneumonia → bronchitis)
3. Wrong Medication: Change the drug (e.g., lisinopril → losartan)
4. Wrong Organism: Change the pathogen (e.g., E. coli → Staph aureus)
5. Wrong Management: Change the treatment plan (e.g., continue → discontinue)

EXAMPLES of clean→error transformations:
{few_shot_clean_to_error}

YOUR GOAL: Introduce ONE medical error to the clean note.

RULES:
1. The note is currently correct
2. Introduce a {error_type} error
3. Keep everything else exactly the same (95% identical)
4. The error should be subtle but medically significant

Format:
<think>[Explain what error you'll introduce and why]</think>
<output>[Note with ONE medical error]</output>"""

            user_content = f"""Clean note (CORRECT):
{clean_note}

YOUR TASK: Introduce a {error_type} error to this clean note.

Make ONE change that introduces a medical error. Keep everything else identical.

<think>[Explain your error]</think>
<output>[Note with error]</output>"""

            original_note = clean_note
            target_note = error_note  # Target is the error version from MEDEC

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
            "original_note": original_note,
            "target_note": target_note,
            "game_category": game_category,
            "error_type": error_type,
        }

    return ds.map(to_prompt, remove_columns=ds.column_names)


def make_assessor_prompts(records: list, tokenizer):
    """Make assessor prompts for GRPO - BINARY CLASSIFICATION.

    CRITICAL: Returns pre-templated strings, NOT messages.
    GRPO should NOT apply chat template again.

    Uses binary classification (Safe/Harmful only).
    
    Args:
        records: List of dicts with 'original', 'attacked', 'game_category' keys
        tokenizer: Tokenizer for applying chat template
        
    Returns:
        Dataset with prompt, original_note, attacked_note, game_category columns
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
                "game_category": rec.get("game_category", "unknown"),  # Pass through game_category
            }
        )

    return Dataset.from_list(prompts)
