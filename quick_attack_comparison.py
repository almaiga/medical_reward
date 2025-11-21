#!/usr/bin/env python3
"""
Quick comparison tool to manually inspect attack quality.
Shows original vs attacked notes side-by-side for human evaluation.
"""

import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher, unified_diff
from typing import List, Dict
import random


def highlight_differences(original: str, attacked: str) -> tuple:
    """Highlight differences between two texts."""
    # Simple word-level diff
    orig_words = original.split()
    attack_words = attacked.split()
    
    matcher = SequenceMatcher(None, orig_words, attack_words)
    
    changes = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            changes.append({
                'type': 'replace',
                'original': ' '.join(orig_words[i1:i2]),
                'attacked': ' '.join(attack_words[j1:j2])
            })
        elif tag == 'delete':
            changes.append({
                'type': 'delete',
                'original': ' '.join(orig_words[i1:i2]),
                'attacked': ''
            })
        elif tag == 'insert':
            changes.append({
                'type': 'insert',
                'original': '',
                'attacked': ' '.join(attack_words[j1:j2])
            })
    
    return changes


def load_interactions(file_path: str, max_samples: int = None) -> List[Dict]:
    """Load interactions from file."""
    interactions = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data = json.loads(line)
                if 'original_note' in data and 'attacked_note' in data:
                    interactions.append(data)
            except json.JSONDecodeError:
                continue
    return interactions


def print_comparison(idx: int, interaction: Dict):
    """Print a formatted comparison of original vs attacked note."""
    print("\n" + "="*100)
    print(f"EXAMPLE #{idx}")
    print("="*100)
    
    original = interaction['original_note']
    attacked = interaction['attacked_note']
    
    # Print metadata if available
    if 'assessor_response' in interaction:
        assessor_label = interaction['assessor_response'].get('label', 'unknown')
        print(f"Assessor Label: {assessor_label}")
    
    if 'judge_assessment' in interaction:
        actual_harm = interaction['judge_assessment'].get('actual_harm', 'unknown')
        correct = interaction['judge_assessment'].get('assessor_was_correct', None)
        print(f"Actual Harm: {actual_harm} | Assessor Correct: {correct}")
    
    if 'rewards' in interaction:
        total_reward = interaction['rewards'].get('total', None)
        harm_evasion = interaction['rewards'].get('harm_evasion', None)
        print(f"Total Reward: {total_reward} | Harm Evasion: {harm_evasion}")
    
    print("\n" + "-"*100)
    print("ORIGINAL NOTE:")
    print("-"*100)
    print(original)
    
    print("\n" + "-"*100)
    print("ATTACKED NOTE:")
    print("-"*100)
    print(attacked)
    
    print("\n" + "-"*100)
    print("KEY DIFFERENCES:")
    print("-"*100)
    
    changes = highlight_differences(original, attacked)
    if changes:
        for i, change in enumerate(changes[:5], 1):  # Show first 5 changes
            print(f"\n{i}. {change['type'].upper()}:")
            if change['original']:
                print(f"   Original: {change['original'][:200]}")
            if change['attacked']:
                print(f"   Attacked: {change['attacked'][:200]}")
    else:
        print("No significant differences detected (might be identical or very similar)")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description="Quick attack comparison tool")
    parser.add_argument("--input", type=str, required=True, help="Path to trainer interactions JSONL")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of examples to show")
    parser.add_argument("--random", action="store_true", help="Select random examples")
    parser.add_argument("--filter-successful", action="store_true", help="Only show successful attacks (fooled assessor)")
    parser.add_argument("--filter-failed", action="store_true", help="Only show failed attacks (caught by assessor)")
    
    args = parser.parse_args()
    
    # Load interactions
    print(f"Loading interactions from {args.input}...")
    interactions = load_interactions(args.input)
    print(f"Loaded {len(interactions)} interactions")
    
    if len(interactions) == 0:
        print("No valid interactions found!")
        return
    
    # Filter if requested
    if args.filter_successful:
        interactions = [
            i for i in interactions 
            if 'judge_assessment' in i and not i['judge_assessment'].get('assessor_was_correct', True)
        ]
        print(f"Filtered to {len(interactions)} successful attacks")
    
    if args.filter_failed:
        interactions = [
            i for i in interactions 
            if 'judge_assessment' in i and i['judge_assessment'].get('assessor_was_correct', False)
        ]
        print(f"Filtered to {len(interactions)} failed attacks")
    
    # Select examples
    if args.random:
        selected = random.sample(interactions, min(args.num_examples, len(interactions)))
    else:
        selected = interactions[:args.num_examples]
    
    # Print comparisons
    for idx, interaction in enumerate(selected, 1):
        print_comparison(idx, interaction)
        
        if idx < len(selected):
            input("\nPress Enter to see next example...")


if __name__ == "__main__":
    main()
