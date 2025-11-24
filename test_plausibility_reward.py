#!/usr/bin/env python3
"""
Test plausibility reward dynamics without policy training.

This script simulates one round of the self-play game:
1. Attacker generates modified note
2. Assessor evaluates the note
3. Judge assesses harm and plausibility
4. Rewards are calculated

No policy training - just prompting to verify the reward dynamics.
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import from selfplay module
import sys
sys.path.append('script')
from selfplay.judge import get_judge_assessment
from selfplay.prompts import build_attacker_prompts, make_assessor_prompts
from selfplay.utils import parse_response
from selfplay.data import load_and_prepare_data
from datasets import Dataset


def load_models(
    policy_model_name="Qwen/Qwen2.5-3B-Instruct",
    judge_model_name="google/medgemma-2b-it"
):
    """Load policy and judge models."""
    print(f"Loading policy model: {policy_model_name}")
    policy_tok = AutoTokenizer.from_pretrained(policy_model_name)
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


def load_test_data(data_path="data/splits/train.jsonl", num_samples=10):
    """Load test data samples."""
    samples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            samples.append(data)
    return samples


def run_attacker(sample, game_category, policy_model, policy_tok, device, few_shot_examples):
    """Run attacker to generate modified note."""
    print(f"\n{'='*80}")
    print("ATTACKER TURN")
    print(f"{'='*80}")
    print(f"Game category: {game_category}")
    
    # Get notes based on game category
    if "harmful" in game_category:
        original_note = sample.get("error_note", sample.get("text", ""))
    else:
        original_note = sample.get("clean_note", sample.get("text", ""))
    
    print(f"Original note (first 200 chars): {original_note[:200]}...")
    
    # Create dataset for attacker
    attacker_data = Dataset.from_dict({
        "game_category": [game_category],
        "error_note": [sample.get("error_note", original_note)],
        "clean_note": [sample.get("clean_note", original_note)],
        "error_type": [sample.get("error_type", "unknown")]
    })
    
    # Build attacker prompts
    attacker_ds = build_attacker_prompts(attacker_data, few_shot_examples, policy_tok, num_shots=2)
    attacker_prompt = attacker_ds[0]["prompt"]
    
    print(f"\nAttacker prompt (first 300 chars):\n{attacker_prompt[:300]}...\n")
    
    # Generate attack
    inputs = policy_tok(attacker_prompt, return_tensors="pt").to(device)
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
    
    # Parse response
    thought, attacked_note = parse_response(completion)
    
    print(f"Attacker thought: {thought[:200] if thought else 'None'}...")
    print(f"Attacked note (first 200 chars): {attacked_note[:200]}...")
    
    return original_note, attacked_note, thought, completion


def run_assessor(attacked_note, policy_model, policy_tok, device):
    """Run assessor to evaluate the note."""
    print(f"\n{'='*80}")
    print("ASSESSOR TURN")
    print(f"{'='*80}")
    print(f"Attacked note (first 200 chars): {attacked_note[:200]}...")
    
    # Create assessor prompt
    assessor_data = [{
        "attacked": attacked_note,
        "game_category": "unknown"  # Assessor doesn't know game type
    }]
    assessor_ds = make_assessor_prompts(assessor_data, policy_tok)
    assessor_prompt = assessor_ds[0]["prompt"]
    
    print(f"\nAssessor prompt (first 300 chars):\n{assessor_prompt[:300]}...\n")
    
    # Generate assessment
    inputs = policy_tok(assessor_prompt, return_tensors="pt").to(device)
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
    
    # Parse response
    thought, label = parse_response(completion)
    
    print(f"Assessor thought: {thought[:200] if thought else 'None'}...")
    print(f"Assessor label: {label}")
    
    return label, thought, completion


def run_judge(original_note, attacked_note, assessor_label, game_category, 
              judge_model, judge_tok, device):
    """Run judge to assess harm and plausibility."""
    print(f"\n{'='*80}")
    print("JUDGE TURN")
    print(f"{'='*80}")
    
    judgments = get_judge_assessment(
        original_note,
        attacked_note,
        assessor_label,
        game_category,
        judge_model,
        judge_tok,
        device
    )
    
    print(f"\nJudge assessment:")
    print(f"  Actual harm: {judgments['actual_harm']}")
    print(f"  Is plausible: {judgments['is_plausible']}")
    print(f"  Assessor correct: {judgments['assessor_was_correct']}")
    
    return judgments


def calculate_rewards(judgments, game_category, attacker_thought, assessor_thought):
    """Calculate rewards for attacker and assessor."""
    print(f"\n{'='*80}")
    print("REWARD CALCULATION")
    print(f"{'='*80}")
    
    actual_harm = judgments["actual_harm"]
    is_plausible = judgments["is_plausible"]
    assessor_correct = judgments["assessor_was_correct"]
    assessor_label = judgments.get("assessor_label", "Unknown")
    
    R_GAME = 1.0
    R_FORMAT = 1.0
    MIN_THINKING_LENGTH = 20
    
    # ATTACKER REWARDS
    print("\n--- ATTACKER REWARDS ---")
    
    # 1. RA_res_harm
    RA_res_harm = -R_GAME if assessor_correct else +R_GAME
    print(f"RA_res_harm: {RA_res_harm:+.1f} ({'assessor wrong' if RA_res_harm > 0 else 'assessor correct'})")
    
    # 2. RA_res_refusal
    if actual_harm == "Harmful" and assessor_label == "Harmful":
        RA_res_refusal = -R_GAME
    elif actual_harm == "Safe" and assessor_label == "Safe":
        RA_res_refusal = -R_GAME
    elif actual_harm == "Harmful" and assessor_label == "Safe":
        RA_res_refusal = +R_GAME
    else:
        RA_res_refusal = +R_GAME
    print(f"RA_res_refusal: {RA_res_refusal:+.1f}")
    
    # 3. RA_revision
    if "harmful" in game_category:
        RA_revision = +R_GAME if actual_harm == "Harmful" else -R_GAME
    else:
        RA_revision = +R_GAME if actual_harm == "Safe" else -R_GAME
    print(f"RA_revision: {RA_revision:+.1f} ({'faithful' if RA_revision > 0 else 'unfaithful'})")
    
    # 4. RA_format
    has_valid_thinking = (
        attacker_thought is not None and 
        len(attacker_thought.strip()) >= MIN_THINKING_LENGTH
    )
    RA_format = +R_FORMAT if has_valid_thinking else -R_FORMAT
    print(f"RA_format: {RA_format:+.1f} ({'valid' if RA_format > 0 else 'invalid'})")
    
    # 5. RA_plausibility (NEW!)
    RA_plausibility = +R_GAME if is_plausible else -R_GAME
    print(f"RA_plausibility: {RA_plausibility:+.1f} ({'✓ plausible' if is_plausible else '✗ implausible'})")
    
    # Total
    RA_total = RA_res_harm + RA_res_refusal + RA_revision + RA_format + RA_plausibility
    print(f"\n💰 ATTACKER TOTAL: {RA_total:+.1f}")
    
    # ASSESSOR REWARDS
    print("\n--- ASSESSOR REWARDS ---")
    
    # 1. RD_res_harm
    RD_res_harm = +R_GAME if assessor_correct else -R_GAME
    print(f"RD_res_harm: {RD_res_harm:+.1f} ({'correct' if RD_res_harm > 0 else 'wrong'})")
    
    # 2. RD_res_refusal
    if actual_harm == "Harmful" and assessor_label == "Harmful":
        RD_res_refusal = +R_GAME
    elif actual_harm == "Safe" and assessor_label == "Safe":
        RD_res_refusal = +R_GAME
    elif actual_harm == "Harmful" and assessor_label == "Safe":
        RD_res_refusal = -R_GAME
    else:
        RD_res_refusal = -R_GAME
    print(f"RD_res_refusal: {RD_res_refusal:+.1f}")
    
    # 3. RD_format
    has_valid_thinking_assessor = (
        assessor_thought is not None and 
        len(assessor_thought.strip()) >= MIN_THINKING_LENGTH
    )
    RD_format = +R_FORMAT if has_valid_thinking_assessor else -R_FORMAT
    print(f"RD_format: {RD_format:+.1f} ({'valid' if RD_format > 0 else 'invalid'})")
    
    # Total
    RD_total = RD_res_harm + RD_res_refusal + RD_format
    print(f"\n💰 ASSESSOR TOTAL: {RD_total:+.1f}")
    
    # Zero-sum check
    zero_sum = (RA_res_harm + RD_res_harm) + (RA_res_refusal + RD_res_refusal)
    print(f"\n🎯 Zero-sum check: {zero_sum:.1f} (should be 0.0)")
    
    return {
        "attacker": {
            "RA_res_harm": RA_res_harm,
            "RA_res_refusal": RA_res_refusal,
            "RA_revision": RA_revision,
            "RA_format": RA_format,
            "RA_plausibility": RA_plausibility,
            "total": RA_total
        },
        "assessor": {
            "RD_res_harm": RD_res_harm,
            "RD_res_refusal": RD_res_refusal,
            "RD_format": RD_format,
            "total": RD_total
        },
        "zero_sum_check": zero_sum
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test plausibility reward dynamics")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--data-path", type=str, default="data/splits/train.jsonl", help="Path to data")
    parser.add_argument("--output", type=str, default="test_plausibility_results.jsonl", help="Output file")
    parser.add_argument("--policy-model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Policy model")
    parser.add_argument("--judge-model", type=str, default="google/medgemma-2b-it", help="Judge model")
    
    args = parser.parse_args()
    
    print("="*80)
    print("PLAUSIBILITY REWARD TEST")
    print("="*80)
    print(f"Testing {args.num_samples} samples")
    print(f"Policy model: {args.policy_model}")
    print(f"Judge model: {args.judge_model}")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Load models
    policy_tok, policy_model, judge_tok, judge_model = load_models(
        args.policy_model, args.judge_model
    )
    device = policy_model.device
    
    # Load test data and few-shot examples
    print(f"\nLoading data...")
    try:
        ds_originals, ds_few_shot = load_and_prepare_data(num_samples=args.num_samples)
        print(f"Loaded {len(ds_originals)} samples and {len(ds_few_shot)} few-shot examples")
        samples = [dict(ds_originals[i]) for i in range(len(ds_originals))]
    except Exception as e:
        print(f"Warning: Could not load with load_and_prepare_data: {e}")
        print(f"Falling back to simple data loading...")
        samples = load_test_data(args.data_path, args.num_samples)
        # Create empty few-shot dataset
        ds_few_shot = Dataset.from_dict({
            "game_category": [],
            "error_note": [],
            "clean_note": [],
            "error_type": []
        })
        print(f"Loaded {len(samples)} samples")
    
    # Run tests
    results = []
    
    for i, sample in enumerate(samples):
        print(f"\n\n{'#'*80}")
        print(f"SAMPLE {i+1}/{len(samples)}")
        print(f"{'#'*80}")
        
        game_category = sample.get("game_category", "adversarial_harmful")
        
        try:
            # 1. Attacker generates attack
            original_note, attacked_note, attacker_thought, attacker_completion = run_attacker(
                sample, game_category, policy_model, policy_tok, device, ds_few_shot
            )
            
            # 2. Assessor evaluates
            assessor_label, assessor_thought, assessor_completion = run_assessor(
                attacked_note, policy_model, policy_tok, device
            )
            
            # 3. Judge assesses
            judgments = run_judge(
                original_note, attacked_note, assessor_label, game_category,
                judge_model, judge_tok, device
            )
            judgments["assessor_label"] = assessor_label
            
            # 4. Calculate rewards
            rewards = calculate_rewards(
                judgments, game_category, attacker_thought, assessor_thought
            )
            
            # Store result
            result = {
                "sample_id": i,
                "game_category": game_category,
                "original_note": original_note,
                "attacked_note": attacked_note,
                "attacker_thought": attacker_thought,
                "assessor_label": assessor_label,
                "assessor_thought": assessor_thought,
                "judgments": judgments,
                "rewards": rewards,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"SAMPLE {i+1} SUMMARY")
            print(f"{'='*80}")
            print(f"Game: {game_category}")
            print(f"Actual harm: {judgments['actual_harm']}")
            print(f"Plausible: {judgments['is_plausible']}")
            print(f"Assessor: {assessor_label} ({'✓' if judgments['assessor_was_correct'] else '✗'})")
            print(f"Attacker reward: {rewards['attacker']['total']:+.1f}")
            print(f"Assessor reward: {rewards['assessor']['total']:+.1f}")
            
        except Exception as e:
            print(f"\n❌ Error processing sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    print(f"\n\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Saved {len(results)} results to {args.output}")
    
    # Print aggregate statistics
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}")
    
    plausible_count = sum(1 for r in results if r['judgments']['is_plausible'])
    assessor_correct_count = sum(1 for r in results if r['judgments']['assessor_was_correct'])
    
    print(f"\nPlausibility rate: {plausible_count}/{len(results)} ({plausible_count/len(results)*100:.1f}%)")
    print(f"Assessor accuracy: {assessor_correct_count}/{len(results)} ({assessor_correct_count/len(results)*100:.1f}%)")
    
    avg_attacker_reward = sum(r['rewards']['attacker']['total'] for r in results) / len(results)
    avg_assessor_reward = sum(r['rewards']['assessor']['total'] for r in results) / len(results)
    
    print(f"\nAverage attacker reward: {avg_attacker_reward:+.2f}")
    print(f"Average assessor reward: {avg_assessor_reward:+.2f}")
    
    # Breakdown by plausibility
    plausible_results = [r for r in results if r['judgments']['is_plausible']]
    implausible_results = [r for r in results if not r['judgments']['is_plausible']]
    
    if plausible_results:
        avg_plausible_reward = sum(r['rewards']['attacker']['total'] for r in plausible_results) / len(plausible_results)
        print(f"\nAverage attacker reward (plausible): {avg_plausible_reward:+.2f}")
    
    if implausible_results:
        avg_implausible_reward = sum(r['rewards']['attacker']['total'] for r in implausible_results) / len(implausible_results)
        print(f"Average attacker reward (implausible): {avg_implausible_reward:+.2f}")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {args.output}")
    print(f"You can now analyze the results to refine the reward function.")


if __name__ == "__main__":
    main()
