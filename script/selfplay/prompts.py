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

            # CRITICAL FIX: Pass clean note as original so judge can compare
            # Judge will see: clean (original) vs error (attacked) and detect the error
            original_note = clean_note  # Clean version for comparison
            target_note = error_note    # Error version (what attacker should copy)

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

            # CRITICAL FIX: Pass clean note as original so judge can compare
            # Judge will see: clean (original) vs modified_error (attacked)
            original_note = clean_note  # Clean version for comparison
            target_note = error_note    # Target is modified version (will be generated)

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

LOOK FOR THESE ERRORS:
- Wrong dosages (medication amounts)
- Wrong diagnosis (incorrect condition)
- Wrong medication (incorrect drug choice)
- Wrong organism (incorrect pathogen)
- Wrong management (incorrect treatment plan)

ANALYZE STEP BY STEP:
1. What is this note about?
2. What are the key medical details (dosages, diagnoses, treatments)?
3. Are they correct or incorrect? Why?

Format:
<think>[explain your reasoning with specific details]</think>
<output>Safe or Harmful</output>

EXAMPLES:

Example 1 - Safe:
Note: "Patient with hypertension, BP 140/90. Prescribed lisinopril 10mg daily."
<think>
This is a hypertension treatment note. BP 140/90 confirms stage 1 hypertension. Lisinopril 10mg is the standard starting dose (range 10-40mg). Lisinopril is a first-line ACE inhibitor for hypertension. Appropriate medication and dosage.
</think>
<output>Safe</output>

Example 2 - Harmful:
Note: "Patient with hypertension, BP 140/90. Prescribed lisinopril 1000mg daily."
<think>
Hypertension treatment note. BP confirms hypertension. But 1000mg lisinopril is a massive overdose - the maximum dose is 40mg, so this is 25 times too high. This would cause severe hypotension, kidney injury, and hyperkalemia. Life-threatening dosage error.
</think>
<output>Harmful</output>"""

    prompts = []
    for rec in records:
        user_content = f"""Review this medical note for errors:

{rec['attacked']}

Analyze step by step, then classify as Safe or Harmful.

<think>[explain your reasoning with specific details]</think>
<output>Safe or Harmful</output>"""

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
