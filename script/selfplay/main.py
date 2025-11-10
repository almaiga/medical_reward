"""
Main training orchestration for self-play GRPO training.

This module contains the main training loop and entry point for the
attacker vs. assessor self-play training system.
"""

print("Starting script imports...")
import os
import json
import argparse
import gc
from datetime import datetime
from copy import deepcopy

print("Basic imports successful...")
import torch

print("PyTorch imported...")
from transformers import AutoTokenizer, AutoModelForCausalLM
 
print("Transformers imported...")
from trl import GRPOConfig, GRPOTrainer

print("TRL imported...")

# Import from our modules
from .data import load_and_prepare_data
from .prompts import build_attacker_prompts, make_assessor_prompts
from .rewards import create_attacker_reward_fn, create_assessor_reward_fn
from .utils import patch_tokenizer_for_grpo, deduplicate_attacked_notes
from .judge import JudgeValidator

R_GAME = 1.0  # Game reward: +1 for win, -1 for loss
R_FORMAT = 1.0  # Format reward: +1 for correct CoT format, -1 for violation

print("Constants defined...")


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

        # Create attacker reward function
        attacker_reward_fn = create_attacker_reward_fn(
            policy_tok=policy_tok,
            judge_model=judge_model,
            judge_tok=judge_tok,
            device=device,
            state=state,
            log_path=log_path,
            attacked_notes_storage=attacked_notes_from_training,
            diversity_stats=diversity_stats,
            judge_validator=judge_validator,
            assessor_snapshot=assessor_snapshot,
            R_GAME=R_GAME,
            R_FORMAT=R_FORMAT,
        )

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
        
        # Create assessor reward function
        assessor_reward_fn = create_assessor_reward_fn(
            judge_model=judge_model,
            judge_tok=judge_tok,
            device=device,
            state=state,
            log_path=log_path,
            judge_validator=judge_validator,
            R_GAME=R_GAME,
            R_FORMAT=R_FORMAT,
        )
        
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
