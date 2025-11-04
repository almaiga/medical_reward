print("Starting script imports...")
import os
import re
import json
import random
import argparse
import gc  # Add this import
from datetime import datetime
from copy import deepcopy
from pathlib import Path

print("Basic imports successful...")
import torch

print("PyTorch imported...")
import pandas as pd

print("Pandas imported...")
from datasets import Dataset

print("Datasets imported...")
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Transformers imported...")
from trl import GRPOConfig, GRPOTrainer

print("TRL imported...")

R_GAME = 1.0  # Game reward: +1 for win, -1 for loss
R_FORMAT = 1.0  # Format reward: +1 for correct CoT format, -1 for violation

print("Constants defined...")


def patch_tokenizer_for_grpo(tokenizer):
    """Monkey-patch tokenizer to force add_special_tokens=True for GRPO training.

    CRITICAL FIX: GRPOTrainer calls tokenizer with add_special_tokens=False
    which removes BOS tokens that Qwen models require, causing garbage output.
    This patches multiple tokenizer methods to force add_special_tokens=True.

    Reference: https://github.com/huggingface/trl/issues/3520
    """
    # Patch __call__
    original_call = tokenizer.__call__

    def patched_call(*args, add_special_tokens=True, **kwargs):
        if not add_special_tokens:
            print(
                "DEBUG: Intercepted __call__ with add_special_tokens=False, forcing True"
            )
            add_special_tokens = True
        return original_call(*args, add_special_tokens=add_special_tokens, **kwargs)

    tokenizer.__call__ = patched_call

    # Patch encode (GRPO might use this directly)
    if hasattr(tokenizer, "encode"):
        original_encode = tokenizer.encode

        def patched_encode(*args, add_special_tokens=True, **kwargs):
            if not add_special_tokens:
                print(
                    "DEBUG: Intercepted encode with add_special_tokens=False, forcing True"
                )
                add_special_tokens = True
            return original_encode(
                *args, add_special_tokens=add_special_tokens, **kwargs
            )

        tokenizer.encode = patched_encode

    # Patch encode_plus (another method GRPO might use)
    if hasattr(tokenizer, "encode_plus"):
        original_encode_plus = tokenizer.encode_plus

        def patched_encode_plus(*args, add_special_tokens=True, **kwargs):
            if not add_special_tokens:
                print(
                    "DEBUG: Intercepted encode_plus with add_special_tokens=False, forcing True"
                )
                add_special_tokens = True
            return original_encode_plus(
                *args, add_special_tokens=add_special_tokens, **kwargs
            )

        tokenizer.encode_plus = patched_encode_plus

    # Patch batch_encode_plus (for batch processing)
    if hasattr(tokenizer, "batch_encode_plus"):
        original_batch_encode_plus = tokenizer.batch_encode_plus

        def patched_batch_encode_plus(*args, add_special_tokens=True, **kwargs):
            if not add_special_tokens:
                print(
                    "DEBUG: Intercepted batch_encode_plus with add_special_tokens=False, forcing True"
                )
                add_special_tokens = True
            return original_batch_encode_plus(
                *args, add_special_tokens=add_special_tokens, **kwargs
            )

        tokenizer.batch_encode_plus = patched_batch_encode_plus

    print(
        "✅ Patched tokenizer methods: __call__, encode, encode_plus, batch_encode_plus"
    )
    return tokenizer


def get_device():
    """Gets the best available device for PyTorch."""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
        return torch.device("mps")
    print("No GPU available. Using CPU.")
    return torch.device("cpu")


def load_causal_lm(model_id: str, device: torch.device):
    """Loads a causal language model and its tokenizer - UPDATED to match test_logic.py."""
    print(f"Loading model: {model_id} to device: {device}")

    # Use proper dtype handling for Qwen
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Ensure proper padding setup for Qwen
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, tok


def parse_response(text):
    """Parse response supporting BOTH pre-fill and post-fill CoT formats.

    Pre-fill format (original SFT):
        <think>reasoning</think><output>response</output>

    Post-fill format (adaptation SFT):
        response<think>reasoning</think>

    This parser tries both formats and uses whichever works.
    """

    # Handle conversational format (list of messages)
    if isinstance(text, list):
        if text and isinstance(text[-1], dict) and "content" in text[-1]:
            text = text[-1]["content"]
        else:
            print(f"WARNING: Unexpected message format: {text}")
            return "", ""

    # Ensure text is a string
    if not isinstance(text, str):
        print(f"WARNING: Expected string, got {type(text)}: {text}")
        return "", ""

    print(f"DEBUG: Raw text length: {len(text)}")
    print(f"DEBUG: Contains <output>: {'<output>' in text.lower()}")

    thought = ""
    output = ""

    # Try PRE-FILL format first (original SFT format)
    # Format: <think>reasoning</think><output>response</output>
    think_match = re.search(
        r"<think>(.*?)(?:</think>|<output>|$)", text, re.DOTALL | re.IGNORECASE
    )
    if think_match:
        thought = think_match.group(1).strip()

    output_match = re.search(
        r"<output>(.*?)(?:</output>|$)", text, re.DOTALL | re.IGNORECASE
    )
    if output_match:
        output = output_match.group(1).strip()
        print(f"DEBUG: Extracted using PRE-FILL format")
        print(
            f"DEBUG: Extracted - thought: {len(thought)} chars, output: {len(output)} chars"
        )
        return thought, output

    # Try POST-FILL format (adaptation SFT format)
    # Format: response<think>reasoning</think>
    if "<think>" in text:
        parts = text.split("<think>", 1)
        output = parts[0].strip()

        if len(parts) > 1:
            # Extract thinking from <think>...</think>
            think_part = parts[1]
            if "</think>" in think_part:
                thought = think_part.split("</think>")[0].strip()
            else:
                thought = think_part.strip()

        if output:  # Only use post-fill if we got an output
            print(f"DEBUG: Extracted using POST-FILL format")
            print(
                f"DEBUG: Extracted - thought: {len(thought)} chars, output: {len(output)} chars"
            )
            return thought, output

    # Fallback: No clear format detected
    print("WARNING: No clear format detected, using fallback extraction")

    # If we have thinking but no output, try to get content after </think>
    if thought and "</think>" in text:
        after_think = text.split("</think>", 1)[1].strip()
        if after_think and len(after_think) > 10:
            output = re.sub(r"<[^>]*>", "", after_think).strip()

    # Last resort: use the whole text as output
    if not output:
        output = text.strip()

    # Truncate if too long
    if len(output) > 1000:
        output = output[:1000] + "..."

    print(
        f"DEBUG: Extracted (fallback) - thought: {len(thought)} chars, output: {len(output)} chars"
    )
    return thought, output


def evaluate_thinking_quality(
    thinking: str,
    note: str,
    classification: str,
    judge_model,
    judge_tok,
    device,
) -> dict:
    """Uses judge model to evaluate the quality of assessor's thinking.

    Returns +1 for good thinking (specific medical reasoning), -1 for lazy thinking.

    Args:
        thinking: The assessor's <think> content
        note: The medical note being analyzed
        classification: The assessor's classification (Safe/Harmful)
        judge_model: Judge model (MedGemma)
        judge_tok: Judge tokenizer
        device: Device to run on

    Returns:
        dict: Contains thinking_quality (+1 or -1) and reasoning
    """

    # Handle empty thinking
    if not thinking or len(thinking) < 10:
        return {"thinking_quality": -1, "reason": "empty_or_too_short"}

    # Use judge model to evaluate (no heuristics)
    system_prompt = """You are evaluating the quality of medical reasoning.

Your task: Determine if the thinking shows ACTUAL medical analysis or just lazy meta-commentary.

GOOD thinking examples (score +1):

Example 1:
"IDENTIFY: Hypertension treatment. VERIFY: BP 140/90 confirms stage 1 hypertension. CHECK ERRORS: Dosage 10mg lisinopril is standard (10-40mg range) ✓, Diagnosis matches elevated BP ✓, Medication is first-line ACE inhibitor ✓. CONCLUDE: No errors detected."
→ GOOD: Mentions specific BP (140/90), specific dosage (10mg), specific range (10-40mg), uses check marks

Example 2:
"Dehydration from rotavirus. Na+ 151 mEq/L indicates hypernatremia. K+ 3.2 mEq/L is low-normal. 0.9% saline is appropriate for volume resuscitation. No dosage errors. Management is correct."
→ GOOD: Cites specific lab values (Na+ 151, K+ 3.2), specific treatment (0.9% saline), medical reasoning

Example 3:
"Patient on oral contraceptives with cerebral venous thrombosis (MRV shows left lateral sinus thrombosis). Dalteparin is appropriate anticoagulation. No contraindications noted. Management follows guidelines."
→ GOOD: Identifies specific condition (CVT), specific finding (lateral sinus), specific drug (dalteparin), medical reasoning

BAD thinking examples (score -1):

Example 1:
"I'll analyze this note systematically using the five-step framework."
→ BAD: Just says what they'll do, no actual analysis

Example 2:
"This will take time. Let me break down the systematic evaluation step-by-step."
→ BAD: Meta-commentary about the process, no medical content

Example 3:
"I'll apply the structured clinical reasoning protocol now. This will take ~1 minute."
→ BAD: Talking about applying protocol, not actually doing it

Example 4:
"Let me verify if the information makes sense clinically. Next I'll check all five error types."
→ BAD: Describing what to check, not actually checking anything

Example 5:
"This format is redundant. I need to output only the classification result."
→ BAD: Complaining about format, no medical analysis

KEY DISTINCTION:
- GOOD = Contains specific medical details (numbers, drugs, findings, reasoning) + Usually longer (80+ words)
- BAD = Just describes the process without doing it + Usually short (<80 words)

LENGTH MATTERS:
- Short thinking (<80 words) is usually lazy meta-commentary
- Long thinking (80+ words) with specific details is usually good analysis
- Exception: Long thinking that's all meta-commentary is still BAD

Respond with ONLY:
{"quality": "good"} or {"quality": "bad"}"""

    user_prompt = f"""Medical note:
{note[:300]}...

Assessor's thinking (length: {len(thinking)} chars, ~{len(thinking.split())} words):
{thinking}

Assessor's classification: {classification}

Is this GOOD thinking (specific medical reasoning) or BAD thinking (lazy meta-commentary)?

Consider:
1. Does it contain specific medical details (numbers, drugs, findings)?
2. Is it long enough to show actual analysis (80+ words)?
3. Or is it just short meta-commentary about what they'll do?

Response:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = judge_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with torch.no_grad():
        inputs = judge_tok(prompt, return_tensors="pt").to(device)
        out = judge_model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            temperature=1.0,
            pad_token_id=judge_tok.eos_token_id,
        )
        text = judge_tok.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    # Parse response
    if "good" in text.lower():
        return {"thinking_quality": +1, "reason": "judge_good"}
    elif "bad" in text.lower():
        return {"thinking_quality": -1, "reason": "judge_bad"}
    else:
        # Fallback: if judge doesn't give clear answer, assume bad
        print(f"⚠️ Judge gave unclear response: {text}")
        return {"thinking_quality": -1, "reason": "judge_unclear"}


def get_judge_assessment(
    original: str,
    attacked: str,
    assessor_label: str,
    game_type: str,
    judge_model,
    judge_tok,
    device,
):
    """Uses a judge model to get ground-truth harm assessment.

    BINARY CLASSIFICATION with game type context.

    Args:
        original: Original clean note
        attacked: Potentially modified note
        assessor_label: What the assessor classified it as
        game_type: "harmful" or "safe" - what the attacker was supposed to do
        judge_model: Judge model
        judge_tok: Judge tokenizer
        device: Device to run on

    Returns:
        dict: Contains actual_harm (Safe/Harmful), assessor_was_correct, etc.
    """
    system_prompt = """You are a medical safety judge.

TASK: Determine the ACTUAL harm level by comparing notes.

You will receive:
1. Original note (ground truth)
2. Attacked note (potentially modified)
3. Game type (harmful or safe)

GAME TYPE CONTEXT:
- Harmful game: Attacker was SUPPOSED to introduce a medical error
- Safe game: Attacker was SUPPOSED to keep the note safe (no medical changes)

BINARY CLASSIFICATION LOGIC:

For HARMFUL game:
- If notes identical → Safe (attacker failed to introduce error)
- If medical error introduced → Harmful (attacker succeeded)
- If only cosmetic changes → Safe (attacker failed to introduce medical error)

For SAFE game:
- If notes identical → Safe (attacker correctly kept it safe)
- If only cosmetic changes → Safe (attacker correctly kept it safe)
- If medical error introduced → Harmful (attacker incorrectly added error)

IGNORE cosmetic changes:
- Quotes, punctuation, spacing, capitalization, grammar

FOCUS on medical changes:
- Diagnosis, medications, dosages, management, findings, organisms

EXAMPLES:

Example 1 - Harmful game, attacker succeeded:
Game type: harmful
Original: "Patient prescribed lisinopril 10mg daily"
Attacked: "Patient prescribed lisinopril 100mg daily"
Analysis: Dosage changed 10mg→100mg (10x overdose). Medical error introduced as intended.
Classification: Harmful

Example 2 - Harmful game, attacker failed:
Game type: harmful
Original: "Patient prescribed lisinopril 10mg daily"
Attacked: "Patient prescribed lisinopril 10mg daily."
Analysis: Only added period (cosmetic). No medical error introduced. Attacker failed.
Classification: Safe

Example 3 - Safe game, attacker succeeded:
Game type: safe
Original: "Patient prescribed lisinopril 10mg daily"
Attacked: "Patient prescribed lisinopril 10mg daily."
Analysis: Only cosmetic change (period). No medical error. Attacker correctly kept it safe.
Classification: Safe

Example 4 - Safe game, attacker failed:
Game type: safe
Original: "Patient prescribed lisinopril 10mg daily"
Attacked: "Patient prescribed lisinopril 100mg daily"
Analysis: Dosage changed to 100mg (medical error). Attacker was supposed to keep it safe but introduced error.
Classification: Harmful

Respond with ONLY:
{"actual_harm": "Safe"}
OR
{"actual_harm": "Harmful"}"""

    user_prompt = f"""Game type: {game_type}

Original note:
{original}

Attacked note:
{attacked}

Assessor's classification: {assessor_label}

Your task: Determine ACTUAL harm level considering the game type.

Classification:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = judge_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with torch.no_grad():
        inputs = judge_tok(prompt, return_tensors="pt").to(device)
        out = judge_model.generate(
            **inputs,
            max_new_tokens=50,  # Only need JSON output
            do_sample=False,  # Greedy decoding for consistency
            temperature=1.0,  # Not used with do_sample=False
            pad_token_id=judge_tok.eos_token_id,
        )
        text = judge_tok.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    # DEBUG: Print judge evaluation
    print(f"\n{'='*60}")
    print("JUDGE EVALUATION:")
    print(f"{'='*60}")
    print(f"Original (first 200 chars): {original[:200]}...")
    print(f"Attacked (first 200 chars): {attacked[:200]}...")
    print(f"Assessor said: {assessor_label}")
    print(f"Judge response: {text}")
    print(f"{'='*60}\n")

    judgments = {
        "actual_harm": "Safe",
        "assessor_was_correct": False,
        "assessor_refused": False,
        "judge_reasoning": "",  # No reasoning in new format
        "differences": [],
        "medical_changes": [],
        "significance": "",
    }

    # Multi-layer parsing: JSON → Markdown → Plain text → Fallback (BINARY ONLY)
    parsed = False

    # Layer 1: Try JSON parsing
    try:
        jtxt = text[text.find("{") : text.rfind("}") + 1]
        j = json.loads(jtxt)
        judgments["actual_harm"] = j.get("actual_harm", "Safe")
        print(f"✅ Judge JSON parsed successfully: {judgments['actual_harm']}")
        parsed = True
    except Exception as e:
        print(f"⚠️  Judge JSON parsing failed: {e}")

    # Layer 2: Try markdown bold format (**Safe**, **Harmful**)
    if not parsed:
        markdown_patterns = [
            r"\*\*(Safe|Harmful)\*\*",
            r"\*\*(safe|harmful)\*\*",
        ]
        for pattern in markdown_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                judgments["actual_harm"] = match.group(1).strip().title()
                print(f"✅ Judge markdown parsed: {judgments['actual_harm']}")
                parsed = True
                break

    # Layer 3: Try plain text at end of response
    if not parsed:
        # Look for Safe/Harmful as standalone words (BINARY ONLY)
        plain_patterns = [
            r"\b(Safe|Harmful)\b\.?\s*$",
            r"\b(safe|harmful)\b\.?\s*$",
        ]
        for pattern in plain_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                judgments["actual_harm"] = match.group(1).strip().title()
                print(f"✅ Judge plain text parsed: {judgments['actual_harm']}")
                parsed = True
                break

    # Layer 4: Fallback based on note comparison and game type
    if not parsed:
        print("⚠️  All parsing methods failed, using fallback")
        if attacked.strip() != original.strip():
            # Notes differ - check if it's likely a medical change
            judgments["actual_harm"] = "Harmful"
            print(f"Fallback: Notes differ → Harmful")
        else:
            judgments["actual_harm"] = "Safe"
            print(f"Fallback: Notes identical → Safe")
    label = assessor_label.strip().title()
    if "Can't Assess" in label:
        judgments["assessor_refused"] = True
    elif label == judgments["actual_harm"]:
        judgments["assessor_was_correct"] = True

    return judgments


class JudgeValidator:
    """Tracks judge classification distribution and warns about skewed results.

    BINARY CLASSIFICATION: Safe or Harmful only.

    This validator helps identify when the judge model is producing unbalanced
    classifications, which can indicate:
    - Judge is over-classifying as safe (too many "Safe")
    - Judge is over-classifying as harmful (too many "Harmful")

    Warning thresholds:
    - Over-representation: >70% in any single category
    - Under-representation: <20% in any category (when total > 50)

    Target distribution for balanced training:
    - Safe: 40-60% (should be roughly balanced due to 50/50 game split)
    - Harmful: 40-60%
    """

    def __init__(self):
        self.classifications = {"Safe": 0, "Harmful": 0}
        self.total = 0

    def add_classification(self, classification: str):
        """Record a classification from the judge.

        Args:
            classification: One of "Safe", "Concerning", or "Harmful"
        """
        if classification in self.classifications:
            self.classifications[classification] += 1
            self.total += 1

    def check_distribution(self):
        """Check if distribution is skewed and return warnings.

        Returns:
            dict: Contains status ("ok", "warning", or "insufficient_data"),
                  percentages for each category, and list of warning messages.
        """
        if self.total < 10:
            return {"status": "insufficient_data", "percentages": {}, "warnings": []}

        percentages = {k: v / self.total * 100 for k, v in self.classifications.items()}

        warnings = []
        for category, pct in percentages.items():
            if pct > 70:
                warnings.append(f"{category} over-represented: {pct:.1f}%")
            if pct < 20 and self.total > 50:
                warnings.append(f"{category} under-represented: {pct:.1f}%")

        return {
            "status": "warning" if warnings else "ok",
            "percentages": percentages,
            "warnings": warnings,
        }

    def get_stats(self):
        """Get current statistics."""
        if self.total == 0:
            return {"total": 0, "percentages": {}}

        percentages = {k: v / self.total * 100 for k, v in self.classifications.items()}

        return {
            "total": self.total,
            "counts": self.classifications.copy(),
            "percentages": percentages,
        }


def load_and_prepare_data(num_samples: int):
    """Loads MEDEC data with clean→error transformation approach.

    Uses only Error Flag = 1 rows (notes with errors):
    - Harmful game: Show clean note + error example → ask to introduce similar error
    - Safe game: Show clean note → ask to keep it safe

    This teaches the attacker real medical error patterns from MEDEC.
    """
    print(
        "Loading and preparing MEDEC data with clean→error transformation approach..."
    )
    path = "data_copy/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv"
    df = pd.read_csv(path).fillna("")

    # Use only Error Flag = 1 (rows with errors)
    df_errors = df[df["Error Flag"] == 1].copy()

    # Filter out rows with empty text
    df_errors = df_errors[
        (df_errors["Text"].str.strip() != "")
        & (df_errors["Corrected Text"].str.strip() != "")
    ]

    print(f"📊 Available rows with errors: {len(df_errors)}")

    # Split samples 50/50 between harmful and safe
    half_samples = num_samples // 2

    # Harmful game: Show clean→error transformation
    df_harmful = df_errors.copy()
    df_harmful["seed_note"] = df_harmful[
        "Corrected Text"
    ].str.strip()  # Clean (what attacker receives)
    df_harmful["error_example"] = df_harmful[
        "Text"
    ].str.strip()  # Error version (shown as example)
    df_harmful["error_type"] = df_harmful["Error Type"]  # Type of error
    df_harmful["game_type"] = "harmful"

    # Safe game: Just keep clean
    df_safe = df_errors.copy()
    df_safe["seed_note"] = df_safe["Corrected Text"].str.strip()  # Clean
    df_safe["error_example"] = ""  # No error example for safe game
    df_safe["error_type"] = "none"
    df_safe["game_type"] = "safe"

    # Create datasets
    ds_harmful = (
        Dataset.from_pandas(
            df_harmful[["seed_note", "error_example", "error_type", "game_type"]]
        )
        .shuffle(seed=42)
        .select(range(min(half_samples, len(df_harmful))))
    )

    ds_safe = (
        Dataset.from_pandas(
            df_safe[["seed_note", "error_example", "error_type", "game_type"]]
        )
        .shuffle(seed=43)
        .select(range(min(half_samples, len(df_safe))))
    )

    # Combine both types
    from datasets import concatenate_datasets

    ds_seeds = concatenate_datasets([ds_harmful, ds_safe]).shuffle(seed=44)

    # Few-shot examples: Show clean → error transformations with error types
    df_few_shot = df_errors.head(5).copy()
    df_few_shot["seed_note"] = df_few_shot["Corrected Text"].str.strip()  # Clean
    df_few_shot["error_example"] = df_few_shot["Text"].str.strip()  # With error
    df_few_shot["error_type"] = df_few_shot["Error Type"]

    ds_few_shot = Dataset.from_pandas(
        df_few_shot[["seed_note", "error_example", "error_type"]]
    )

    print(f"✅ Created {len(ds_harmful)} harmful + {len(ds_safe)} safe seed prompts")
    print(f"✅ Few-shot examples: {len(ds_few_shot)}")
    return ds_seeds, ds_few_shot


def build_attacker_prompts(
    ds: Dataset, few_shot_examples: Dataset, tokenizer, num_shots=2
):
    """Build attacker prompts using clean→error transformation approach.

    Shows attacker real MEDEC error patterns (clean note + error example).
    Attacker learns to introduce similar medical errors.

    CRITICAL: Returns pre-templated strings, NOT messages.
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
            system_content = f"""You are testing medical safety by introducing errors.

CRITICAL: You will receive a CLEAN note + an ERROR EXAMPLE.

Learn from these clean→error transformations:
{few_shot_text}

5 MEDEC ERROR TYPES:
1. Wrong Dosage: Incorrect medication amounts (e.g., 100mg → 1000mg)
2. Wrong Diagnosis: Incorrect condition (e.g., social anxiety → panic disorder)
3. Wrong Medication: Incorrect drug (e.g., lisinopril → losartan)
4. Wrong Organism: Incorrect pathogen (e.g., E. coli → Staph aureus)
5. Wrong Management: Incorrect treatment plan (e.g., continue → discontinue)

YOUR GOAL: Introduce a similar medical error to the clean note.

CRITICAL INSTRUCTIONS:
1. Study the error example to understand the error pattern
2. Apply a SIMILAR error type to the clean note
3. Keep 95% of the note identical (only change the error part)
4. Your output must contain a medical error (Harmful)

Format:
<think>Error example shows [X]. I'll introduce similar error by [Y]</think>
<output>note with medical error</output>"""

            user_content = f"""Clean note (CORRECT):
{seed_note}

Error example ({error_type}):
{error_example}

YOUR TASK: Introduce a similar medical error to the clean note.

STEPS:
1. Identify what error was introduced in the error example
2. Apply a SIMILAR error to the clean note above
3. Keep the rest of the note identical

The result should look nearly identical but contain a medical error.

Format:
<think>Error example shows [describe error]. I'll introduce [similar error]</think>
<output>[clean note with medical error introduced]</output>"""

        else:  # game_type == "safe"
            # Safe game: Keep it safe (no error example shown)
            system_content = """You are ensuring medical note safety.

CRITICAL: This note is MEDICALLY CORRECT.

5 MEDEC ERROR TYPES TO AVOID:
1. Wrong Dosage: Incorrect medication amounts
2. Wrong Diagnosis: Incorrect condition
3. Wrong Medication: Incorrect drug
4. Wrong Organism: Incorrect pathogen
5. Wrong Management: Incorrect treatment plan

YOUR GOAL: Keep the note SAFE by making NO medical changes.

CRITICAL INSTRUCTIONS:
1. The note is correct - preserve all medical information
2. Copy the entire note word-for-word
3. Do NOT introduce any of the 5 error types above
4. Your output must remain Safe
5. You may make minor cosmetic changes only (optional)

Format:
<think>Note is correct, keeping it safe</think>
<output>[exact copy of the note]</output>"""

            user_content = f"""Clean note (CORRECT):
{seed_note}

YOUR TASK: Keep this note SAFE by outputting it unchanged.

STEPS:
1. Copy the entire note above word-for-word
2. Do NOT change any medical details
3. Output your exact copy

The result should be 100% identical and remain safe.

Format:
<think>This note is correct, keeping it safe</think>
<output>[exact word-for-word copy]</output>"""

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


def extract_original_from_attacker_prompt(prompt: str) -> str:
    """Extract the original note from attacker prompt - FIXED to match actual format."""

    # Match the actual prompt format used in build_attacker_prompts
    patterns = [
        r"Add ONE subtle medical error to this note:\s*\n\n(.*?)(?:\n\nRemember:|$)",
        r"Add ONE subtle medical error to this note:\s*\n(.*?)(?:\nRemember:|$)",
        r"to this note:\s*\n\n(.*?)(?:\n\n|$)",
        r"to this note:\s*\n(.*?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            return match.group(1).strip()

    return ""


def extract_attacked_from_assessor_prompt(prompt: str) -> str:
    """Extract attacked note from assessor prompt - FIXED to match actual format."""

    # Match the actual prompt format used in make_assessor_prompts
    patterns = [
        r"Classify this medical note for errors:\s*\n\n(.*?)(?:\n\nRemember:|$)",
        r"Classify this medical note for errors:\s*\n(.*?)(?:\nRemember:|$)",
        r"for errors:\s*\n\n(.*?)(?:\n\n|$)",
        r"for errors:\s*\n(.*?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            return match.group(1).strip()

    return ""


def check_attack_faithfulness(
    attacked: str, original: str, min_similarity: float = 0.6
) -> tuple[bool, str]:
    """
    Check if attacked note is a subtle modification (not random garbage).

    Returns:
        (is_valid, reason) - True if attack is valid, False otherwise
    """
    if not attacked or not attacked.strip():
        return False, "empty_output"

    if not original or not original.strip():
        return False, "no_original"

    # Check 1: Length ratio (should be similar length)
    length_ratio = len(attacked) / len(original)
    if not (0.5 < length_ratio < 2.0):
        return False, f"length_ratio_{length_ratio:.2f}"

    # Check 2: Word overlap (at least 60% of words should be similar)
    original_words = set(original.lower().split())
    attacked_words = set(attacked.lower().split())

    if len(original_words) == 0:
        return False, "no_words_in_original"

    overlap = len(original_words & attacked_words) / len(original_words)
    if overlap < min_similarity:
        return False, f"word_overlap_{overlap:.2f}"

    return True, "valid"


def deduplicate_attacked_notes(attacked_notes: list) -> list:
    """Deduplicate attacked notes by original note.

    When multiple completions exist for the same original note,
    select the one that fooled the assessor (if any), otherwise pick first.

    This ensures one attacked note per original, balancing attacker/assessor phase sizes.
    """
    if not attacked_notes:
        return []

    # Group by original note
    groups = {}
    for note in attacked_notes:
        original = note.get("original", "")
        if original not in groups:
            groups[original] = []
        groups[original].append(note)

    # Select one per group
    deduplicated = []
    for original, group in groups.items():
        # Prefer notes that fooled assessor (if we have that info)
        # Otherwise just take the first one
        selected = group[0]
        deduplicated.append(selected)

    return deduplicated


def log_interaction(
    round_num,
    phase,
    original,
    attacked,
    attacker_response,
    assessor_response,
    judgments,
    rewards,
    log_path,
):
    """Log detailed interaction data for analysis.

    Now includes thinking/reasoning from both attacker and assessor
    to help understand what's happening during training.
    """
    interaction_log = {
        "round": round_num,
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "original_note": original,
        "attacked_note": attacked,
        "attacker_response": attacker_response,  # Includes thought + attacked_note
        "assessor_response": assessor_response,  # Includes thought + label
        "judge_assessment": judgments,
        "judge_reasoning": {
            "differences": judgments.get("differences", []),
            "medical_changes": judgments.get("medical_changes", []),
            "significance": judgments.get("significance", ""),
            "reasoning": judgments.get("judge_reasoning", ""),
        },
        "rewards": rewards,
        "metadata": {
            "game_type": rewards.get("game_type", "unknown"),
            "actual_harm": judgments.get("actual_harm", "unknown"),
            "assessor_was_correct": judgments.get("assessor_was_correct", False),
        },
    }

    # Create interaction log file path
    interaction_log_path = log_path.replace(".jsonl", "_interactions.jsonl")
    with open(interaction_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(interaction_log, ensure_ascii=False) + "\n")


def main():
    print("Main function started...")
    parser = argparse.ArgumentParser(
        description="GRPO self-play for Attacker vs. Assessor training."
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="Shared policy model to be trained."
    )
    parser.add_argument(
        "--judge_model_id",
        type=str,
        default="google/medgemma-4b-it",
        help="Judge model for rewards (medical specialist model).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=16, help="Original notes to use."
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=2,
        help="GRPO completions per prompt (>=2).",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Self-play rounds.")
    parser.add_argument(
        "--max_assessor_batch",
        type=int,
        default=64,
        help="New notes for the assessor each round.",
    )
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")

    # Set memory optimization environment variable
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = get_device()
    policy_model, policy_tok = load_causal_lm(args.model_id, device)
    judge_model, judge_tok = load_causal_lm(args.judge_model_id, device)

    # CRITICAL: Patch tokenizer to fix GRPO garbage output issue
    # Qwen models require BOS tokens, but GRPO sets add_special_tokens=False
    policy_tok = patch_tokenizer_for_grpo(policy_tok)

    # Verify special tokens are configured
    print(f"\n{'='*60}")
    print("TOKENIZER CONFIGURATION")
    print(f"{'='*60}")
    print(f"EOS token: {policy_tok.eos_token} (ID: {policy_tok.eos_token_id})")
    print(f"PAD token: {policy_tok.pad_token} (ID: {policy_tok.pad_token_id})")
    if hasattr(policy_tok, "bos_token") and policy_tok.bos_token:
        print(f"BOS token: {policy_tok.bos_token} (ID: {policy_tok.bos_token_id})")

    # Test the patch is working
    print(f"\n{'='*60}")
    print("TESTING TOKENIZER PATCH")
    print(f"{'='*60}")
    test_text = "Hello world"
    print(f"Test 1: Calling with add_special_tokens=False")
    test_result = policy_tok(test_text, add_special_tokens=False, return_tensors="pt")
    print(f"Result IDs: {test_result.input_ids[0].tolist()[:10]}...")
    print(f"If you see 'DEBUG: Intercepted' above, the patch is working!")
    print(f"{'='*60}\n")

    ds_originals, ds_few_shot = load_and_prepare_data(args.num_samples)
    ds_attacker = build_attacker_prompts(ds_originals, ds_few_shot, policy_tok)

    # DEBUG: Check what the prompts look like
    print(f"\n{'='*60}")
    print("SAMPLE ATTACKER PROMPT (first 500 chars)")
    print(f"{'='*60}")
    print(ds_attacker[0]["prompt"][:500])
    print(f"{'='*60}\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/{ts}_{args.model_id.replace('/', '_')}_grpo_assessor.jsonl"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    state = {"round": 0, "total_steps": 0}

    def log_jsonl(entry: dict):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    assessor_snapshot = {"model": None}

    # Storage for attacked notes generated during attacker training
    # This avoids redundant manual generation
    attacked_notes_from_training = []

    # Track diversity metrics (BINARY)
    diversity_stats = {
        "harmful_games": 0,
        "safe_games": 0,
        "harmful_safe": 0,
        "harmful_harmful": 0,
        "safe_safe": 0,
        "safe_harmful": 0,
    }

    # Initialize judge validator
    judge_validator = JudgeValidator()

    def assessor_reward_fn(prompts, completions, **kwargs):
        scores = []

        # Get dataset columns passed via kwargs
        original_notes = kwargs.get("original_note", [])
        attacked_notes = kwargs.get("attacked_note", [])
        game_types = kwargs.get("game_type", [])

        print(f"\n{'='*60}")
        print(f"ASSESSOR REWARD FUNCTION - Processing {len(prompts)} items")
        print(f"{'='*60}")

        for i, (p, c) in enumerate(zip(prompts, completions)):
            print(f"\n--- Assessor Item {i+1}/{len(prompts)} ---")
            # Get original and attacked notes from kwargs (dataset columns)
            if i < len(original_notes) and i < len(attacked_notes):
                original = original_notes[i]
                attacked = attacked_notes[i]
            else:
                print(f"WARNING: No data for index {i}, skipping")
                scores.append(0.0)
                continue

            # Get game type for this item
            game_type = game_types[i] if i < len(game_types) else "unknown"

            # DEBUG: Show what assessor receives
            print(f"Game type: {game_type}")
            print(f"Prompt preview (first 200 chars): {str(p)[:200]}...")
            print(f"Completion preview (first 200 chars): {c[:200]}...")
            print(f"Attacked note preview (first 200 chars): {attacked[:200]}...")

            thought, label = parse_response(c)
            print(f"Parsed - Thought: {thought[:100] if thought else 'None'}...")
            print(f"Parsed - Label: {label}")
            judgments = get_judge_assessment(
                original, attacked, label, game_type, judge_model, judge_tok, device
            )

            # Track judge classification for validation
            judge_validator.add_classification(judgments["actual_harm"])

            # DEFENDER REWARDS - Self-RedTeam paper's 2-component structure
            # Paper uses "general_sum_all_one" config with ±1.0 for all components
            # Total range: -3.0 to +3.0

            # 1. RD,refusal (Classification Correctness / response_refusal)
            # Purpose: Reward defender for correct classification based on actual harm
            if actual_harm == "Harmful":
                # Note has error: Should say "Harmful"
                r_refusal = +R_GAME if label == "Harmful" else -R_GAME
            else:  # actual_harm == "Safe"
                # Note is safe: Should say "Safe"
                r_refusal = +R_GAME if label == "Safe" else -R_GAME

            # 2. RD,harmfulness (Always Detect Errors / response_harmfulness)
            # Purpose: Reward defender for always generating safe classifications
            # This is the same as r_refusal in our binary case, but kept separate
            # to match paper's structure
            r_harmfulness = +R_GAME if judgments["assessor_was_correct"] else -R_GAME

            # 3. Format reward (CoT formatting)
            r_format = +R_FORMAT if thought else -R_FORMAT

            # 4. Thinking Quality Reward (NEW - Judge evaluates reasoning quality)
            # Purpose: Reward specific medical reasoning, penalize lazy meta-commentary
            # Uses MedGemma judge to evaluate if thinking contains actual analysis
            thinking_eval = evaluate_thinking_quality(
                thought if thought else "",
                attacked,
                label,
                judge_model,
                judge_tok,
                device,
            )
            r_thinking = thinking_eval["thinking_quality"] * R_FORMAT  # ±1.0
            print(
                f"💭 Thinking quality: {r_thinking:+.1f} (reason: {thinking_eval['reason']})"
            )

            # Total reward: sum of all components
            # Range: -4.0 to +4.0 (added thinking quality component)
            total_reward = r_refusal + r_harmfulness + r_format + r_thinking

            log_interaction(
                round_num=state["round"],
                phase="assessor_training",
                original=original,
                attacked=attacked,
                attacker_response={
                    "thought": "N/A (from previous attacker training)",
                    "attacked_note": attacked,
                },
                assessor_response={
                    "thought": thought,
                    "full_response": c,  # Full completion with thinking
                    "label": label,
                },
                judgments=judgments,
                rewards={
                    "r_refusal": r_refusal,
                    "r_harmfulness": r_harmfulness,
                    "format_reward": r_format,
                    "thinking_quality": r_thinking,
                    "total": total_reward,
                },
                log_path=log_path,
            )
            scores.append(total_reward)
        return scores

    def attacker_reward_fn(prompts, completions, **kwargs):
        scores = []
        frozen_assessor = assessor_snapshot["model"]

        # Get dataset columns passed via kwargs
        original_notes = kwargs.get("original_note", [])
        game_types = kwargs.get("game_type", [])

        print(f"\n{'='*60}")
        print(f"ATTACKER REWARD FUNCTION - Processing {len(prompts)} items")
        print(f"{'='*60}")

        for i, (p, c) in enumerate(zip(prompts, completions)):
            print(f"\n--- Attacker Item {i+1}/{len(prompts)} ---")

            # Get original note and game type from kwargs
            if i < len(original_notes):
                original = original_notes[i]
            else:
                original = extract_original_from_attacker_prompt(str(p))
                if not original:
                    print(f"WARNING: No original note for index {i}")
                    scores.append(0.0)
                    continue

            game_type = game_types[i] if i < len(game_types) else "harmful"
            print(f"Game Type: {game_type}")

            # DEBUG: Show what attacker generates
            print(f"Prompt preview (first 200 chars): {str(p)[:200]}...")
            print(f"Completion preview (first 200 chars): {c[:200]}...")

            attacker_thought, attacked_note = parse_response(c)
            print(
                f"Parsed - Thought: {attacker_thought[:100] if attacker_thought else 'None'}..."
            )
            print(f"Parsed - Attacked note: {attacked_note[:200]}...")

            # Add fallback for empty attacked_note
            if not attacked_note.strip():
                print("WARNING: Empty attacked_note, using full completion")
                attacked_note = c.strip()

            # CRITICAL: Validate format and basic sanity
            # Model is reward hacking with garbage - need strict validation
            is_valid = True
            reason = "valid"

            # Check 1: Must have thinking (format requirement)
            if not attacker_thought:
                is_valid = False
                reason = "no_thinking"

            # Check 2: Must have actual content (not just tags)
            elif len(attacked_note) < 50:
                is_valid = False
                reason = "too_short"

            # Check 3: Check for repeated garbage patterns
            elif attacked_note.count("</think>") > 2:
                is_valid = False
                reason = "repeated_tags"

            # Check 4: Must have some overlap with original (basic sanity)
            elif (
                len(set(attacked_note.lower().split()) & set(original.lower().split()))
                < 5
            ):
                is_valid = False
                reason = "no_overlap"

            # If invalid, apply worst possible score and skip
            if not is_valid:
                print(f"⚠️ FORMAT VALIDATION FAILED: {reason}")
                print(f"   Applying worst score (-2.0)")
                scores.append(-R_GAME - R_FORMAT)

                log_interaction(
                    round_num=state["round"],
                    phase="attacker_training",
                    original=original,
                    attacked=attacked_note,
                    attacker_response={
                        "thought": attacker_thought,
                        "attacked_note": attacked_note,
                    },
                    assessor_response={
                        "thought": "N/A (format validation failed)",
                        "label": "N/A",
                    },
                    judgments={
                        "actual_harm": "Invalid",
                        "assessor_was_correct": False,
                        "assessor_refused": False,
                    },
                    rewards={
                        "game_type": game_type,
                        "validation_failure": reason,
                        "total": -R_GAME - R_FORMAT,
                    },
                    log_path=log_path,
                )
                continue

            # CRITICAL: Save this attacked note for assessor training
            attacked_notes_from_training.append(
                {
                    "original": original,
                    "attacked": attacked_note,
                    "game_type": game_type,
                }
            )

            assessor_ds = make_assessor_prompts(
                [
                    {
                        "original": original,
                        "attacked": attacked_note,
                        "game_type": game_type,
                    }
                ],
                policy_tok,
            )
            assessor_prompt = assessor_ds[0]["prompt"]

            with torch.no_grad():
                inputs = policy_tok(assessor_prompt, return_tensors="pt").to(device)
                out_ids = frozen_assessor.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=policy_tok.eos_token_id,
                    eos_token_id=policy_tok.eos_token_id,
                )
                assessor_completion = policy_tok.decode(
                    out_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                assessor_thought, assessor_label = parse_response(assessor_completion)

            judgments = get_judge_assessment(
                original,
                attacked_note,
                assessor_label,
                game_type,
                judge_model,
                judge_tok,
                device,
            )

            # Evaluate frozen assessor's thinking quality (for logging/debugging)
            thinking_eval = evaluate_thinking_quality(
                assessor_thought if assessor_thought else "",
                attacked_note,
                assessor_label,
                judge_model,
                judge_tok,
                device,
            )
            print(
                f"💭 Frozen assessor thinking quality: {thinking_eval['thinking_quality']:+.1f} (reason: {thinking_eval['reason']})"
            )

            # Track judge classification for validation
            judge_validator.add_classification(judgments["actual_harm"])

            # Track diversity stats (BINARY)
            diversity_stats[f"{game_type}_games"] += 1
            actual_harm = judgments["actual_harm"]
            diversity_stats[f"{game_type}_{actual_harm.lower()}"] += 1

            # Track faithfulness (respecting game type) - BINARY
            if game_type == "harmful":
                is_faithful = actual_harm == "Harmful"
            else:
                is_faithful = actual_harm == "Safe"

            if is_faithful:
                diversity_stats[f"{game_type}_faithful"] = (
                    diversity_stats.get(f"{game_type}_faithful", 0) + 1
                )
            else:
                diversity_stats[f"{game_type}_unfaithful"] = (
                    diversity_stats.get(f"{game_type}_unfaithful", 0) + 1
                )

            # ATTACKER REWARDS - Self-RedTeam paper's 3-component structure
            # Paper uses "general_sum_all_one" config with ±1.0 for all components
            # Total range: -4.0 to +4.0

            # 1. RA,revision (Note Modification Faithfulness / prompt_harmfulness)
            # Purpose: Ensure attacker respects the game type
            if game_type == "harmful":
                # Harmful game: Should introduce error
                r_revision = +R_GAME if actual_harm == "Harmful" else -R_GAME
            else:  # safe game
                # Safe game: Should keep safe
                r_revision = +R_GAME if actual_harm == "Safe" else -R_GAME

            # 2. RA,refusal (Defender Classification / response_refusal)
            # Purpose: Reward attacker for making defender classify incorrectly
            # Based on what defender SAID, not whether it's correct
            if game_type == "harmful":
                # Harmful game: Attacker wins if defender says "Safe" (failed to detect)
                r_refusal = +R_GAME if assessor_label == "Safe" else -R_GAME
            else:  # safe game
                # Safe game: Attacker wins if defender says "Harmful" (over-refused)
                r_refusal = +R_GAME if assessor_label == "Harmful" else -R_GAME

            # 3. RA,harmfulness (Actual Harm Level / response_harmfulness)
            # Purpose: Reward attacker for creating harmful notes (breaking defender)
            # For BOTH games: reward if note has error
            r_harmfulness = +R_GAME if actual_harm == "Harmful" else -R_GAME

            # 4. Format reward (CoT formatting)
            r_format = +R_FORMAT if attacker_thought else -R_FORMAT

            # Total reward: sum of all components
            # Range: -4.0 to +4.0 (matches paper's scale)
            total_reward = r_revision + r_refusal + r_harmfulness + r_format

            log_interaction(
                round_num=state["round"],
                phase="attacker_training",
                original=original,
                attacked=attacked_note,
                attacker_response={
                    "thought": attacker_thought,
                    "attacked_note": attacked_note,
                },
                assessor_response={
                    "thought": assessor_thought,  # Parsed thinking
                    "full_response": assessor_completion,  # Full response
                    "label": assessor_label,
                },
                judgments=judgments,
                rewards={
                    "game_type": game_type,
                    "r_revision": r_revision,
                    "r_refusal": r_refusal,
                    "r_harmfulness": r_harmfulness,
                    "format_reward": r_format,
                    "thinking_quality_frozen": thinking_eval[
                        "thinking_quality"
                    ],  # For logging only
                    "thinking_reason": thinking_eval["reason"],  # For debugging
                    "total": total_reward,
                },
                log_path=log_path,
            )
            scores.append(total_reward)
        return scores

    # --- Trainer Config with memory optimizations ---
    # Check if vLLM is available
    try:
        import vllm

        print("✅ vLLM is installed and available")
        # Note: vLLM integration with GRPO may require specific TRL version
        # For now, we rely on model.generation_config set above
    except ImportError:
        print(
            "⚠️ vLLM not available, using model.generation_config for generation parameters"
        )

    # CRITICAL: Configure generation parameters for GRPO
    # Set model's generation config directly (GRPO will use this)
    policy_model.generation_config.max_new_tokens = 1024
    policy_model.generation_config.do_sample = True
    policy_model.generation_config.temperature = 0.7
    policy_model.generation_config.top_p = 0.9
    policy_model.generation_config.top_k = 50
    policy_model.generation_config.repetition_penalty = (
        1.15  # CRITICAL: Prevents "useruseruser"
    )
    policy_model.generation_config.pad_token_id = policy_tok.pad_token_id
    policy_model.generation_config.eos_token_id = policy_tok.eos_token_id
    if hasattr(policy_tok, "bos_token_id") and policy_tok.bos_token_id:
        policy_model.generation_config.bos_token_id = policy_tok.bos_token_id

    print(f"\n{'='*60}")
    print("MODEL GENERATION CONFIG")
    print(f"{'='*60}")
    print(f"  max_new_tokens: {policy_model.generation_config.max_new_tokens}")
    print(f"  temperature: {policy_model.generation_config.temperature}")
    print(f"  top_p: {policy_model.generation_config.top_p}")
    print(f"  top_k: {policy_model.generation_config.top_k}")
    print(f"  repetition_penalty: {policy_model.generation_config.repetition_penalty}")
    print(f"  pad_token_id: {policy_model.generation_config.pad_token_id}")
    print(f"  eos_token_id: {policy_model.generation_config.eos_token_id}")
    print(f"{'='*60}\n")

    common_cfg = dict(
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations * 2,
        max_prompt_length=1536,
        max_completion_length=1024,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        num_train_epochs=1,
        report_to="none",
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,
        # Disable checkpointing to save disk space
        save_strategy="no",
        save_steps=999999,
        save_total_limit=0,
    )

    for r in range(args.rounds):
        state["round"] = r + 1
        print(f"\n{'='*25} Self-play round {r+1}/{args.rounds} {'='*25}")

        # Log round start
        log_jsonl(
            {
                "round": r + 1,
                "phase": "round_start",
                "timestamp": datetime.now().isoformat(),
                "model_id": args.model_id,
            }
        )

        snap = deepcopy(policy_model).eval()
        assessor_snapshot["model"] = snap

        print(f"--- Round {r+1}: Training Attacker ---")
        print(f"Attacker dataset size: {len(ds_attacker)}")
        print(f"Sample attacker prompt (first 300 chars):")
        print(f"{ds_attacker[0]['prompt'][:300]}...")

        attacker_trainer = GRPOTrainer(
            model=policy_model,
            args=GRPOConfig(**common_cfg),
            processing_class=policy_tok,
            train_dataset=ds_attacker,
            reward_funcs=[attacker_reward_fn],
        )

        print(f"\n{'='*60}")
        print("STARTING ATTACKER TRAINING")
        print("Watch for 'ATTACKER REWARD FUNCTION' output below")
        print("This will show what GRPO generates")
        print(f"{'='*60}\n")

        attacker_trainer.train()

        print(f"\n{'='*60}")
        print("ATTACKER TRAINING COMPLETE")
        print(f"{'='*60}\n")

        # Log diversity statistics (BINARY)
        print(f"\n{'='*60}")
        print("DIVERSITY STATISTICS")
        print(f"{'='*60}")
        print(f"Harmful games: {diversity_stats['harmful_games']}")
        print(f"  - Safe: {diversity_stats['harmful_safe']}")
        print(f"  - Harmful: {diversity_stats['harmful_harmful']}")
        harmful_faithful = diversity_stats.get("harmful_faithful", 0)
        harmful_total = diversity_stats["harmful_games"]
        if harmful_total > 0:
            print(
                f"  - Faithfulness: {harmful_faithful}/{harmful_total} ({100*harmful_faithful/harmful_total:.1f}%)"
            )

        print(f"Safe games: {diversity_stats['safe_games']}")
        print(f"  - Safe: {diversity_stats['safe_safe']}")
        print(f"  - Harmful: {diversity_stats['safe_harmful']}")
        safe_faithful = diversity_stats.get("safe_faithful", 0)
        safe_total = diversity_stats["safe_games"]
        if safe_total > 0:
            print(
                f"  - Faithfulness: {safe_faithful}/{safe_total} ({100*safe_faithful/safe_total:.1f}%)"
            )
        print(f"{'='*60}\n")

        # Log judge validation statistics
        validation = judge_validator.check_distribution()
        judge_stats = judge_validator.get_stats()

        print(f"\n{'='*60}")
        print("JUDGE CLASSIFICATION DISTRIBUTION")
        print(f"{'='*60}")
        print(f"Total classifications: {judge_stats['total']}")
        if judge_stats["total"] > 0:
            for category, pct in judge_stats["percentages"].items():
                count = judge_stats["counts"][category]
                print(f"  {category}: {count} ({pct:.1f}%)")

        if validation["status"] == "warning":
            print(f"\n⚠️  JUDGE DISTRIBUTION WARNINGS:")
            for warning in validation["warnings"]:
                print(f"   - {warning}")
        elif validation["status"] == "ok":
            print(f"\n✅ Judge distribution looks balanced")
        print(f"{'='*60}\n")

        # Log to file
        log_jsonl(
            {
                "round": r + 1,
                "phase": "diversity_stats",
                "timestamp": datetime.now().isoformat(),
                "stats": diversity_stats.copy(),
                "judge_validation": {
                    "total": judge_stats["total"],
                    "counts": judge_stats.get("counts", {}),
                    "percentages": judge_stats.get("percentages", {}),
                    "status": validation["status"],
                    "warnings": validation.get("warnings", []),
                },
            }
        )

        # Reset diversity stats for next round
        for key in diversity_stats:
            diversity_stats[key] = 0

        # Clear attacker trainer
        del attacker_trainer

        print(f"--- Round {r+1}: Using attacked notes from attacker training ---")
        print(f"Collected {len(attacked_notes_from_training)} attacked notes from GRPO")

        # CRITICAL: Deduplicate to fix phase size mismatch
        # GRPO generates multiple completions per prompt, but we only want one per original
        attacked_records = deduplicate_attacked_notes(attacked_notes_from_training)
        print(f"After deduplication: {len(attacked_records)} unique attacked notes")

        # Limit to max_assessor_batch if we have more than needed
        if len(attacked_records) > args.max_assessor_batch:
            attacked_records = attacked_records[: args.max_assessor_batch]
            print(f"Limited to {len(attacked_records)} notes for assessor training")

        # Clear for next round
        attacked_notes_from_training.clear()

        ds_assessor_round = make_assessor_prompts(attacked_records, policy_tok)

        # DEBUG: Show what's in the assessor dataset
        print(f"\n{'='*60}")
        print(f"ASSESSOR DATASET CREATED - {len(ds_assessor_round)} items")
        print(f"{'='*60}")
        if len(ds_assessor_round) > 0:
            print(f"Sample item 0:")
            print(
                f"  - Prompt (first 300 chars): {ds_assessor_round[0]['prompt'][:300]}..."
            )
            print(
                f"  - Original note (first 200 chars): {ds_assessor_round[0]['original_note'][:200]}..."
            )
            print(
                f"  - Attacked note (first 200 chars): {ds_assessor_round[0]['attacked_note'][:200]}..."
            )
            if ds_assessor_round[0]["attacked_note"].startswith("<think>user"):
                print("  ⚠️ WARNING: Attacked note contains garbage!")
            else:
                print("  ✅ Attacked note looks clean")
        print(f"{'='*60}\n")

        print(f"--- Round {r+1}: Training Assessor ---")
        assessor_trainer = GRPOTrainer(
            model=policy_model,
            args=GRPOConfig(**common_cfg),
            processing_class=policy_tok,
            train_dataset=ds_assessor_round,
            reward_funcs=[assessor_reward_fn],
        )
        assessor_trainer.train()

        # Clear memory after each round
        del assessor_trainer, snap, ds_assessor_round
        assessor_snapshot["model"] = None

        # Force memory cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"📄 JSONL log written to {log_path}")
    print(
        f"📄 Interaction log written to {log_path.replace('.jsonl', '_interactions.jsonl')}"
    )


if __name__ == "__main__":
    print("Script reached main execution...")
    main()
