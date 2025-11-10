"""
Utility functions for parsing, validation, and logging.

This module contains helper functions used across the self-play training pipeline,
including response parsing, attack validation, and interaction logging.
"""

import os
import re
import json
from datetime import datetime


def patch_tokenizer_for_grpo(tokenizer):
    """Monkey-patch tokenizer to force add_special_tokens=True for GRPO training.

    CRITICAL FIX: GRPOTrainer calls tokenizer with add_special_tokens=False
    which removes BOS tokens that Qwen models require, causing garbage output.
    This patches multiple tokenizer methods to force add_special_tokens=True.

    Reference: https://github.com/huggingface/trl/issues/3520
    
    Args:
        tokenizer: Tokenizer to patch
        
    Returns:
        Patched tokenizer
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


def parse_response(text):
    """Parse response supporting BOTH pre-fill and post-fill CoT formats.

    Pre-fill format (original SFT):
        <think>reasoning</think><output>response</output>

    Post-fill format (adaptation SFT):
        response<think>reasoning</think>

    This parser tries both formats and uses whichever works.
    
    Args:
        text: Model output text (string or list of messages)
        
    Returns:
        (thought, output) tuple of strings
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


def extract_original_from_attacker_prompt(prompt: str) -> str:
    """Extract the original note from attacker prompt - FIXED to match actual format.
    
    Args:
        prompt: Attacker prompt string
        
    Returns:
        Original note text
    """

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
    """Extract attacked note from assessor prompt - FIXED to match actual format.
    
    Args:
        prompt: Assessor prompt string
        
    Returns:
        Attacked note text
    """

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
    
    Args:
        attacked: Attacked note text
        original: Original note text
        min_similarity: Minimum word overlap ratio (default 0.6)

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
    
    Args:
        attacked_notes: List of dicts with 'original', 'attacked', 'game_type'
        
    Returns:
        Deduplicated list of attacked notes
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
    
    Args:
        round_num: Current training round
        phase: Training phase (attacker_training or assessor_training)
        original: Original medical note
        attacked: Attacked/modified note
        attacker_response: Dict with attacker's thought and attacked_note
        assessor_response: Dict with assessor's thought and label
        judgments: Dict with judge's assessment
        rewards: Dict with reward components
        log_path: Path to main log file
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
            "raw_response": judgments.get("judge_raw_response", ""),
            "reasoning": judgments.get("judge_reasoning", ""),
            "differences": judgments.get("differences", []),
            "medical_changes": judgments.get("medical_changes", []),
            "significance": judgments.get("significance", ""),
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
