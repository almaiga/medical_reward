#!/usr/bin/env python3
"""
Analyze attack difficulty and assessor performance.
Helps understand why self-play training might not be improving.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
from collections import Counter
import re


def load_interactions(file_path: str) -> List[Dict]:
    """Load all interactions from file."""
    interactions = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'original_note' in data and 'attacked_note' in data:
                    interactions.append(data)
            except json.JSONDecodeError:
                continue
    return interactions


def calculate_edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def analyze_attack_characteristics(original: str, attacked: str) -> Dict:
    """Analyze characteristics of an attack."""
    # Basic metrics
    orig_len = len(original)
    attack_len = len(attacked)
    length_diff = abs(attack_len - orig_len)
    length_ratio = attack_len / orig_len if orig_len > 0 else 0
    
    # Edit distance
    edit_dist = calculate_edit_distance(original, attacked)
    edit_ratio = edit_dist / max(orig_len, attack_len) if max(orig_len, attack_len) > 0 else 0
    
    # Word-level changes
    orig_words = set(original.lower().split())
    attack_words = set(attacked.lower().split())
    words_added = len(attack_words - orig_words)
    words_removed = len(orig_words - attack_words)
    
    # Check for common patterns that might indicate low-quality attacks
    has_think_tag = '<think>' in attacked.lower()
    is_very_short = len(attacked) < 50
    is_identical = original == attacked
    
    # Check for medical terms (simple heuristic)
    medical_terms = ['diagnosis', 'treatment', 'medication', 'symptom', 'disease', 
                     'patient', 'clinical', 'therapy', 'syndrome', 'infection']
    medical_term_count = sum(1 for term in medical_terms if term in attacked.lower())
    
    return {
        'original_length': orig_len,
        'attacked_length': attack_len,
        'length_diff': length_diff,
        'length_ratio': length_ratio,
        'edit_distance': edit_dist,
        'edit_ratio': edit_ratio,
        'words_added': words_added,
        'words_removed': words_removed,
        'has_think_tag': has_think_tag,
        'is_very_short': is_very_short,
        'is_identical': is_identical,
        'medical_term_count': medical_term_count
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze attack difficulty")
    parser.add_argument("--input", type=str, required=True, help="Path to trainer interactions JSONL")
    parser.add_argument("--output", type=str, help="Output CSV file")
    
    args = parser.parse_args()
    
    # Load interactions
    print(f"Loading interactions from {args.input}...")
    interactions = load_interactions(args.input)
    print(f"Loaded {len(interactions)} interactions")
    
    if len(interactions) == 0:
        print("No valid interactions found!")
        return
    
    # Analyze each interaction
    results = []
    for interaction in interactions:
        original = interaction['original_note']
        attacked = interaction['attacked_note']
        
        # Get attack characteristics
        characteristics = analyze_attack_characteristics(original, attacked)
        
        # Get metadata
        result = {**characteristics}
        
        if 'assessor_response' in interaction:
            result['assessor_label'] = interaction['assessor_response'].get('label', 'unknown')
        
        if 'judge_assessment' in interaction:
            result['actual_harm'] = interaction['judge_assessment'].get('actual_harm', 'unknown')
            result['assessor_correct'] = interaction['judge_assessment'].get('assessor_was_correct', None)
        
        if 'rewards' in interaction:
            result['total_reward'] = interaction['rewards'].get('total', None)
            result['harm_evasion_reward'] = interaction['rewards'].get('harm_evasion', None)
        
        if 'round' in interaction:
            result['round'] = interaction['round']
        
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    
    # Print analysis
    print("\n" + "="*80)
    print("ATTACK DIFFICULTY ANALYSIS")
    print("="*80)
    
    print(f"\nTotal interactions: {len(df)}")
    
    # Quality issues
    print("\n--- QUALITY ISSUES ---")
    print(f"Attacks with <think> tags: {df['has_think_tag'].sum()} ({df['has_think_tag'].mean()*100:.1f}%)")
    print(f"Very short attacks (<50 chars): {df['is_very_short'].sum()} ({df['is_very_short'].mean()*100:.1f}%)")
    print(f"Identical to original: {df['is_identical'].sum()} ({df['is_identical'].mean()*100:.1f}%)")
    
    # Attack characteristics
    print("\n--- ATTACK CHARACTERISTICS ---")
    print(f"Average edit distance: {df['edit_distance'].mean():.1f}")
    print(f"Average edit ratio: {df['edit_ratio'].mean():.3f}")
    print(f"Average length ratio: {df['length_ratio'].mean():.3f}")
    print(f"Average words added: {df['words_added'].mean():.1f}")
    print(f"Average words removed: {df['words_removed'].mean():.1f}")
    print(f"Average medical terms: {df['medical_term_count'].mean():.1f}")
    
    # Assessor performance
    if 'assessor_correct' in df.columns:
        print("\n--- ASSESSOR PERFORMANCE ---")
        correct_rate = df['assessor_correct'].mean()
        print(f"Overall accuracy: {correct_rate*100:.1f}%")
        
        if 'assessor_label' in df.columns:
            print("\nLabel distribution:")
            print(df['assessor_label'].value_counts())
        
        # Performance by attack characteristics
        print("\n--- PERFORMANCE BY ATTACK TYPE ---")
        
        # By edit ratio (quartiles)
        df['edit_ratio_quartile'] = pd.qcut(df['edit_ratio'], q=4, labels=['Q1 (small)', 'Q2', 'Q3', 'Q4 (large)'], duplicates='drop')
        if df['edit_ratio_quartile'].nunique() > 1:
            print("\nAccuracy by edit ratio:")
            print(df.groupby('edit_ratio_quartile')['assessor_correct'].agg(['mean', 'count']))
        
        # By length change
        df['length_change'] = df['length_ratio'].apply(
            lambda x: 'much_shorter' if x < 0.8 else ('shorter' if x < 0.95 else ('similar' if x < 1.05 else ('longer' if x < 1.2 else 'much_longer')))
        )
        print("\nAccuracy by length change:")
        print(df.groupby('length_change')['assessor_correct'].agg(['mean', 'count']))
        
        # Quality issues vs performance
        print("\n--- QUALITY ISSUES VS PERFORMANCE ---")
        if df['has_think_tag'].sum() > 0:
            print(f"Accuracy on attacks with <think> tags: {df[df['has_think_tag']]['assessor_correct'].mean()*100:.1f}%")
        if df['is_very_short'].sum() > 0:
            print(f"Accuracy on very short attacks: {df[df['is_very_short']]['assessor_correct'].mean()*100:.1f}%")
    
    # Reward analysis
    if 'total_reward' in df.columns:
        print("\n--- REWARD ANALYSIS ---")
        print(f"Average total reward: {df['total_reward'].mean():.2f}")
        print(f"Average harm evasion reward: {df['harm_evasion_reward'].mean():.2f}")
        
        # High reward attacks
        high_reward = df[df['total_reward'] > df['total_reward'].quantile(0.75)]
        print(f"\nHigh reward attacks (top 25%):")
        print(f"  Average edit ratio: {high_reward['edit_ratio'].mean():.3f}")
        print(f"  Assessor accuracy: {high_reward['assessor_correct'].mean()*100:.1f}%")
        
        # Low reward attacks
        low_reward = df[df['total_reward'] < df['total_reward'].quantile(0.25)]
        print(f"\nLow reward attacks (bottom 25%):")
        print(f"  Average edit ratio: {low_reward['edit_ratio'].mean():.3f}")
        print(f"  Assessor accuracy: {low_reward['assessor_correct'].mean()*100:.1f}%")
    
    # Round progression
    if 'round' in df.columns and df['round'].nunique() > 1:
        print("\n--- PROGRESSION OVER ROUNDS ---")
        round_stats = df.groupby('round').agg({
            'assessor_correct': 'mean',
            'edit_ratio': 'mean',
            'total_reward': 'mean'
        })
        print(round_stats)
    
    print("\n" + "="*80)
    
    # Recommendations
    print("\n--- RECOMMENDATIONS ---")
    
    if df['has_think_tag'].mean() > 0.1:
        print("⚠️  Many attacks contain <think> tags - attacker is not generating proper attacks")
    
    if df['is_very_short'].mean() > 0.1:
        print("⚠️  Many attacks are very short - might be degenerate outputs")
    
    if df['is_identical'].mean() > 0.05:
        print("⚠️  Some attacks are identical to original - attacker is not making changes")
    
    if 'assessor_correct' in df.columns:
        if df['assessor_correct'].mean() > 0.8:
            print("⚠️  Assessor accuracy is very high - attacks might be too easy")
        elif df['assessor_correct'].mean() < 0.4:
            print("⚠️  Assessor accuracy is very low - attacks might be too hard or implausible")
        
        if df['edit_ratio'].mean() > 0.5:
            print("⚠️  Large edit ratios - attacks might be changing too much")
        elif df['edit_ratio'].mean() < 0.05:
            print("⚠️  Small edit ratios - attacks might be too subtle or minimal")


if __name__ == "__main__":
    main()
