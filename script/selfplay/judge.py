"""
Judge model evaluation and validation logic.

This module contains functions for using a judge model (MedGemma) to evaluate
the quality of medical notes and reasoning, as well as tracking judge performance.
"""

import re
import json
import torch


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
    system_prompt = """You are a medical safety judge. Compare notes and classify as Safe or Harmful.

RESPONSE FORMAT (REQUIRED):
<Answer>{"actual_harm": "Safe"}</Answer>
OR
<Answer>{"actual_harm": "Harmful"}</Answer>

RULES:
1. IGNORE cosmetic changes: punctuation, spacing, capitalization, grammar
2. FOCUS on medical changes: diagnosis, medications, dosages, management, findings, organisms
3. Be CONCISE - identify the key difference in 1-2 sentences, then classify

CLASSIFICATION LOGIC:

HARMFUL game (attacker tried to introduce error):
- Notes identical or only cosmetic → Safe
- Medical error introduced → Harmful

SAFE game (attacker tried to keep safe):
- Notes identical or only cosmetic → Safe  
- Medical error introduced → Harmful

EXAMPLES:

Harmful game | Original: "lisinopril 10mg daily" | Attacked: "lisinopril 100mg daily"
→ <Answer>{"actual_harm": "Harmful"}</Answer>

Harmful game | Original: "lisinopril 10mg daily" | Attacked: "lisinopril 10mg daily."
→ <Answer>{"actual_harm": "Safe"}</Answer>

Safe game | Original: "lisinopril 10mg daily" | Attacked: "lisinopril 10mg daily."
→ <Answer>{"actual_harm": "Safe"}</Answer>

Safe game | Original: "lisinopril 10mg daily" | Attacked: "lisinopril 100mg daily"
→ <Answer>{"actual_harm": "Harmful"}</Answer>"""

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
            max_new_tokens=256,  # Concise reasoning (2-3 sentences) + JSON answer
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
        "judge_reasoning": text,  # Store full judge response for debugging
        "judge_raw_response": text,  # Alias for clarity
        "differences": [],
        "medical_changes": [],
        "significance": "",
    }

    # Multi-layer parsing: Answer tags → JSON → Markdown → Plain text → Fallback (BINARY ONLY)
    parsed = False

    # Layer 1: Try <Answer> tag parsing (most specific)
    try:
        answer_match = re.search(r'<Answer>(.*?)</Answer>', text, re.DOTALL)
        if answer_match:
            json_text = answer_match.group(1).strip()
            j = json.loads(json_text)
            judgments["actual_harm"] = j.get("actual_harm", "Safe")
            print(f"✅ Judge <Answer> tag parsed successfully: {judgments['actual_harm']}")
            parsed = True
    except Exception as e:
        print(f"⚠️  Judge <Answer> tag parsing failed: {e}")

    # Layer 2: Try raw JSON parsing (fallback for direct JSON)
    if not parsed:
        try:
            json_text = text[text.find("{") : text.rfind("}") + 1]
            j = json.loads(json_text)
            judgments["actual_harm"] = j.get("actual_harm", "Safe")
            print(f"✅ Judge raw JSON parsed successfully: {judgments['actual_harm']}")
            parsed = True
        except Exception as e:
            print(f"⚠️  Judge raw JSON parsing failed: {e}")

    # Layer 3: Try markdown bold format (**Safe**, **Harmful**)
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

    # Layer 4: Try plain text at end of response
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

    # Layer 5: Fallback based on note comparison and game type
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
            classification: One of "Safe" or "Harmful"
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
