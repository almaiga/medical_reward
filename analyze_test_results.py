#!/usr/bin/env python3
"""
Analyze test results from plausibility reward test.
"""

import json
import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Analyze test results")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    
    args = parser.parse_args()
    
    # Load results
    results = []
    with open(args.input, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    print("="*80)
    print(f"ANALYSIS OF {len(results)} TEST RESULTS")
    print("="*80)
    
    # Plausibility analysis
    print("\n--- PLAUSIBILITY ---")
    plausible = [r for r in results if r['judgments']['is_plausible']]
    implausible = [r for r in results if not r['judgments']['is_plausible']]
    
    print(f"Plausible: {len(plausible)} ({len(plausible)/len(results)*100:.1f}%)")
    print(f"Implausible: {len(implausible)} ({len(implausible)/len(results)*100:.1f}%)")
    
    # Assessor performance
    print("\n--- ASSESSOR PERFORMANCE ---")
    correct = [r for r in results if r['judgments']['assessor_was_correct']]
    print(f"Correct: {len(correct)} ({len(correct)/len(results)*100:.1f}%)")
    
    # By plausibility
    if plausible:
        plausible_correct = [r for r in plausible if r['judgments']['assessor_was_correct']]
        print(f"Correct on plausible: {len(plausible_correct)}/{len(plausible)} ({len(plausible_correct)/len(plausible)*100:.1f}%)")
    
    if implausible:
        implausible_correct = [r for r in implausible if r['judgments']['assessor_was_correct']]
        print(f"Correct on implausible: {len(implausible_correct)}/{len(implausible)} ({len(implausible_correct)/len(implausible)*100:.1f}%)")
    
    # Reward analysis
    print("\n--- REWARDS ---")
    
    attacker_rewards = [r['rewards']['attacker']['total'] for r in results]
    assessor_rewards = [r['rewards']['assessor']['total'] for r in results]
    
    print(f"Attacker avg: {sum(attacker_rewards)/len(attacker_rewards):+.2f}")
    print(f"Assessor avg: {sum(assessor_rewards)/len(assessor_rewards):+.2f}")
    
    # By plausibility
    if plausible:
        plausible_rewards = [r['rewards']['attacker']['total'] for r in plausible]
        print(f"Attacker avg (plausible): {sum(plausible_rewards)/len(plausible_rewards):+.2f}")
    
    if implausible:
        implausible_rewards = [r['rewards']['attacker']['total'] for r in implausible]
        print(f"Attacker avg (implausible): {sum(implausible_rewards)/len(implausible_rewards):+.2f}")
    
    # Reward components
    print("\n--- REWARD COMPONENTS (Attacker) ---")
    components = ['RA_res_harm', 'RA_res_refusal', 'RA_revision', 'RA_format', 'RA_plausibility']
    for comp in components:
        values = [r['rewards']['attacker'][comp] for r in results]
        avg = sum(values) / len(values)
        positive = sum(1 for v in values if v > 0)
        print(f"{comp}: {avg:+.2f} ({positive}/{len(values)} positive)")
    
    # Examples
    print("\n--- EXAMPLE: BEST ATTACK (Highest Reward) ---")
    best = max(results, key=lambda r: r['rewards']['attacker']['total'])
    print(f"Reward: {best['rewards']['attacker']['total']:+.1f}")
    print(f"Plausible: {best['judgments']['is_plausible']}")
    print(f"Assessor: {best['assessor_label']} (actual: {best['judgments']['actual_harm']})")
    print(f"Original: {best['original_note'][:150]}...")
    print(f"Attacked: {best['attacked_note'][:150]}...")
    
    print("\n--- EXAMPLE: WORST ATTACK (Lowest Reward) ---")
    worst = min(results, key=lambda r: r['rewards']['attacker']['total'])
    print(f"Reward: {worst['rewards']['attacker']['total']:+.1f}")
    print(f"Plausible: {worst['judgments']['is_plausible']}")
    print(f"Assessor: {worst['assessor_label']} (actual: {worst['judgments']['actual_harm']})")
    print(f"Original: {worst['original_note'][:150]}...")
    print(f"Attacked: {worst['attacked_note'][:150]}...")
    
    # Implausible examples
    if implausible:
        print("\n--- EXAMPLE: IMPLAUSIBLE ATTACK ---")
        imp_example = implausible[0]
        print(f"Reward: {imp_example['rewards']['attacker']['total']:+.1f}")
        print(f"Assessor: {imp_example['assessor_label']} (actual: {imp_example['judgments']['actual_harm']})")
        print(f"Original: {imp_example['original_note'][:150]}...")
        print(f"Attacked: {imp_example['attacked_note'][:150]}...")
        print(f"Judge reasoning: {imp_example['judgments'].get('judge_reasoning', 'N/A')[:200]}...")


if __name__ == "__main__":
    main()
