"""
Reward calculation functions for attacker and assessor roles.

This module contains reward functions compatible with GRPOTrainer,
implementing the Self-RedTeam paper's reward structure.
"""

import torch
from .utils import parse_response, log_interaction, extract_original_from_attacker_prompt
from .judge import get_judge_assessment, evaluate_thinking_quality
from .prompts import make_assessor_prompts


def create_assessor_reward_fn(
    judge_model,
    judge_tok,
    device,
    state: dict,
    log_path: str,
    judge_validator,
    R_GAME: float = 1.0,
    R_FORMAT: float = 1.0
):
    """Create assessor reward function with closure over dependencies.
    
    Returns a function compatible with GRPOTrainer that calculates rewards
    for the assessor (defender) role.
    
    Args:
        judge_model: Judge model for evaluation
        judge_tok: Judge tokenizer
        device: Device to run on
        state: Training state dict with 'round' key
        log_path: Path to log file
        judge_validator: JudgeValidator instance
        R_GAME: Game reward magnitude (default 1.0)
        R_FORMAT: Format reward magnitude (default 1.0)
        
    Returns:
        Reward function compatible with GRPOTrainer
    """
    
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

            # Extract actual_harm for use in reward calculations
            actual_harm = judgments["actual_harm"]

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
    
    return assessor_reward_fn


def create_attacker_reward_fn(
    policy_tok,
    judge_model,
    judge_tok,
    device,
    state: dict,
    log_path: str,
    attacked_notes_storage: list,
    diversity_stats: dict,
    judge_validator,
    assessor_snapshot: dict,
    R_GAME: float = 1.0,
    R_FORMAT: float = 1.0
):
    """Create attacker reward function with closure over dependencies.
    
    Returns a function compatible with GRPOTrainer that calculates rewards
    for the attacker role.
    
    Args:
        policy_tok: Policy tokenizer
        judge_model: Judge model for evaluation
        judge_tok: Judge tokenizer
        device: Device to run on
        state: Training state dict with 'round' key
        log_path: Path to log file
        attacked_notes_storage: List to store attacked notes for assessor training
        diversity_stats: Dict tracking diversity metrics
        judge_validator: JudgeValidator instance
        assessor_snapshot: Dict with frozen assessor model
        R_GAME: Game reward magnitude (default 1.0)
        R_FORMAT: Format reward magnitude (default 1.0)
        
    Returns:
        Reward function compatible with GRPOTrainer
    """
    
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
            attacked_notes_storage.append(
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
    
    return attacker_reward_fn
