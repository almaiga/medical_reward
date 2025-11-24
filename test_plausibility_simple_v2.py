#!/usr/bin/env python3
"""
Simple test of plausibility reward using selfplay module functions.

This script reuses the actual reward functions from the training pipeline
to test plausibility reward dynamics without training.
"""

import sys
sys.path.append('script')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from selfplay.data import load_and_prepare_data
from selfplay.prompts import build_attacker_prompts, make_assessor_prompts
from selfplay.rewards import create_attacker_reward_fn, create_assessor_reward_fn
from selfplay.judge import JudgeValidator
from selfplay.utils import parse_response
import json
from datetime import datetime


def load_models(
    policy_model_name="trainer_output/qwen3-4b-medical-selfplay-sft",
    judge_model_name="google/medgemma-4b-it"
):
    """Load models."""
    print(f"Loading policy model: {policy_model_name}")
    policy_tok = AutoTokenizer.from_pretrained(policy_model_name)
    if policy_tok.pad_token is None:
        policy_tok.pad_token = policy_tok.eos_token
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    policy_model.eval()
    
    print(f"Loading judge model: {judge_model_name}")
    judge_tok = AutoTokenizer.from_pretrained(judge_model_name)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    judge_model.eval()
    
    return policy_tok, policy_model, judge_tok, judge_model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--policy-model", type=str, default="trainer_output/qwen3-4b-medical-selfplay-sft")
    parser.add_argument("--judge-model", type=str, default="google/medgemma-4b-it")
    args = parser.parse_args()
    
    print("="*80)
    print("PLAUSIBILITY REWARD TEST (Using Selfplay Module)")
    print("="*80)
    print(f"Testing {args.num_samples} samples")
    print(f"Policy model: {args.policy_model}")
    print(f"Judge model: {args.judge_model}")
    print("="*80)
    
    # Load models
    policy_tok, policy_model, judge_tok, judge_model = load_models(
        args.policy_model, args.judge_model
    )
    device = next(policy_model.parameters()).device
    
    # Load data using selfplay module
    print("\nLoading data...")
    ds_seeds, ds_few_shot = load_and_prepare_data(args.num_samples)
    print(f"Loaded {len(ds_seeds)} samples and {len(ds_few_shot)} few-shot examples\n")
    
    # Build attacker prompts using selfplay module
    print("Building attacker prompts...")
    attacker_ds = build_attacker_prompts(ds_seeds, ds_few_shot, policy_tok, num_shots=2)
    
    # Storage for results
    attacked_notes_storage = []
    diversity_stats = {}
    judge_validator = JudgeValidator()
    state = {"round": 0}
    log_path = f"test_plausibility_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    # Create assessor snapshot (frozen copy)
    assessor_snapshot = {"model": policy_model}
    
    # Create reward functions using selfplay module
    print("Creating reward functions...")
    attacker_reward_fn = create_attacker_reward_fn(
        policy_tok=policy_tok,
        judge_model=judge_model,
        judge_tok=judge_tok,
        device=device,
        state=state,
        log_path=log_path,
        attacked_notes_storage=attacked_notes_storage,
        diversity_stats=diversity_stats,
        judge_validator=judge_validator,
        assessor_snapshot=assessor_snapshot,
        R_GAME=1.0,
        R_FORMAT=1.0
    )
    
    # Generate attacker completions
    print("\nGenerating attacker completions...")
    attacker_prompts = attacker_ds["prompt"]
    attacker_completions = []
    
    for i, prompt in enumerate(attacker_prompts):
        print(f"\nSample {i+1}/{len(attacker_prompts)}")
        inputs = policy_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = policy_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=policy_tok.eos_token_id,
            )
        completion = policy_tok.decode(
            outputs[0, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        attacker_completions.append(completion)
        
        # Show preview
        thought, attacked = parse_response(completion)
        print(f"  Thought: {thought[:100] if thought else 'None'}...")
        print(f"  Attacked: {attacked[:100] if attacked else 'None'}...")
    
    # Calculate rewards using the actual reward function
    print("\n" + "="*80)
    print("CALCULATING REWARDS")
    print("="*80)
    
    rewards = attacker_reward_fn(
        attacker_prompts,
        attacker_completions,
        original_note=attacker_ds["original_note"],
        game_category=attacker_ds["game_category"]
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total samples: {len(rewards)}")
    print(f"Average reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Min reward: {min(rewards):.2f}")
    print(f"Max reward: {max(rewards):.2f}")
    print(f"\nDiversity stats:")
    for key, value in sorted(diversity_stats.items()):
        print(f"  {key}: {value}")
    
    print(f"\nJudge validator stats:")
    stats = judge_validator.get_stats()
    print(f"  Total: {stats['total']}")
    for category, pct in stats.get('percentages', {}).items():
        print(f"  {category}: {pct:.1f}%")
    
    print(f"\nDetailed results saved to: {log_path}")
    print("\nTo analyze plausibility:")
    print(f"  grep 'is_plausible' {log_path} | grep -c 'true'")
    print(f"  grep 'is_plausible' {log_path} | grep -c 'false'")


if __name__ == "__main__":
    main()
