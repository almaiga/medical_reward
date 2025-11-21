#!/usr/bin/env python3
"""
Filter selfplay interactions by game category.
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter interactions by game category")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--game-category", type=str, 
                        choices=['vanilla_harmful', 'vanilla_benign', 'adversarial_harmful', 'adversarial_benign'], 
                        default='adversarial_harmful', help="Game category to filter")
    parser.add_argument("--phase", type=str, choices=['attacker_training', 'assessor_training'],
                        default='attacker_training', help="Phase to filter")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to output")
    
    args = parser.parse_args()
    
    print(f"Filtering {args.input}...")
    print(f"  Game category: {args.game_category}")
    print(f"  Phase: {args.phase}")
    
    filtered = []
    total = 0
    
    with open(args.input, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                total += 1
                
                # Check phase
                if data.get('phase') != args.phase:
                    continue
                
                # Check game category (in rewards)
                rewards = data.get('rewards', {})
                game_category = rewards.get('game_category', 'unknown')
                
                if game_category == args.game_category:
                    filtered.append(data)
                    
                    if args.max_samples and len(filtered) >= args.max_samples:
                        break
                        
            except json.JSONDecodeError:
                continue
    
    print(f"\nTotal interactions: {total}")
    print(f"Filtered to {len(filtered)} {args.game_category} {args.phase} interactions")
    
    # Write output
    with open(args.output, 'w') as f:
        for item in filtered:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nSaved to {args.output}")
    
    # Print some stats
    if filtered:
        print("\nSample statistics:")
        actual_harms = [item['judge_assessment']['actual_harm'] for item in filtered 
                       if 'judge_assessment' in item]
        from collections import Counter
        harm_dist = Counter(actual_harms)
        print(f"  Actual harm distribution: {dict(harm_dist)}")
        
        assessor_correct = [item['judge_assessment']['assessor_was_correct'] 
                           for item in filtered if 'judge_assessment' in item]
        if assessor_correct:
            acc = sum(assessor_correct) / len(assessor_correct) * 100
            print(f"  Assessor accuracy: {acc:.1f}%")


if __name__ == "__main__":
    main()
