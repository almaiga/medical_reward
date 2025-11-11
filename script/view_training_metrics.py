#!/usr/bin/env python3
"""
Simple script to view training metrics from GRPO self-play training.

Usage:
    python script/view_training_metrics.py results/20251111_123456_model_grpo_metrics_summary.json
    python script/view_training_metrics.py results/20251111_123456_model_grpo_metrics.jsonl
"""

import json
import sys
from pathlib import Path


def view_summary(summary_path: str):
    """View metrics summary file."""
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"TRAINING METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Start: {summary['start_time']}")
    if 'end_time' in summary:
        print(f"End: {summary['end_time']}")
    print(f"{'='*60}\n")
    
    # Round-by-round metrics
    for round_data in summary['rounds']:
        round_num = round_data['round']
        print(f"Round {round_num}:")
        
        if 'attacker' in round_data:
            att = round_data['attacker']
            print(f"  Attacker:")
            if att.get('loss') is not None:
                print(f"    Loss: {att['loss']:.4f}")
            if att.get('reward') is not None:
                print(f"    Reward: {att['reward']:.4f} ± {att.get('reward_std', 0):.4f}")
            if att.get('entropy') is not None:
                print(f"    Entropy: {att['entropy']:.4f}")
        
        if 'assessor' in round_data:
            ass = round_data['assessor']
            print(f"  Assessor:")
            if ass.get('loss') is not None:
                print(f"    Loss: {ass['loss']:.4f}")
            if ass.get('reward') is not None:
                print(f"    Reward: {ass['reward']:.4f} ± {ass.get('reward_std', 0):.4f}")
            if ass.get('entropy') is not None:
                print(f"    Entropy: {ass['entropy']:.4f}")
        
        print()
    
    # Overall statistics
    if 'overall' in summary:
        print(f"{'='*60}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*60}")
        overall = summary['overall']
        print(f"Total Rounds: {overall['total_rounds']}")
        
        if overall['attacker']['avg_loss'] is not None:
            print(f"\nAttacker:")
            print(f"  Avg Loss: {overall['attacker']['avg_loss']:.4f}")
            if overall['attacker']['avg_reward'] is not None:
                print(f"  Avg Reward: {overall['attacker']['avg_reward']:.4f}")
        
        if overall['assessor']['avg_loss'] is not None:
            print(f"\nAssessor:")
            print(f"  Avg Loss: {overall['assessor']['avg_loss']:.4f}")
            if overall['assessor']['avg_reward'] is not None:
                print(f"  Avg Reward: {overall['assessor']['avg_reward']:.4f}")
        
        print(f"{'='*60}\n")


def view_detailed(jsonl_path: str, last_n: int = 10):
    """View detailed metrics from JSONL file."""
    metrics = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    
    print(f"\n{'='*60}")
    print(f"DETAILED TRAINING METRICS")
    print(f"{'='*60}")
    print(f"Total entries: {len(metrics)}")
    print(f"Showing last {min(last_n, len(metrics))} entries")
    print(f"{'='*60}\n")
    
    for entry in metrics[-last_n:]:
        print(f"Round {entry['round']} - {entry['phase']}:")
        
        if entry['phase'] in ['attacker', 'assessor']:
            # Training metrics
            if 'loss' in entry:
                print(f"  Loss: {entry['loss']:.4f}")
            if 'reward' in entry:
                print(f"  Reward: {entry['reward']:.4f} ± {entry.get('reward_std', 0):.4f}")
            if 'grad_norm' in entry:
                print(f"  Grad Norm: {entry['grad_norm']:.4f}")
            if 'learning_rate' in entry:
                print(f"  LR: {entry['learning_rate']:.2e}")
            if 'entropy' in entry:
                print(f"  Entropy: {entry['entropy']:.4f}")
            if 'completions/mean_length' in entry:
                print(f"  Mean Length: {entry['completions/mean_length']:.1f} tokens")
        
        elif entry['phase'] == 'round_summary':
            # Round summary
            if 'diversity' in entry:
                div = entry['diversity']
                print(f"  Diversity:")
                print(f"    Harmful games: {div.get('harmful_games', 0)}")
                print(f"    Safe games: {div.get('safe_games', 0)}")
            
            if 'judge' in entry:
                judge = entry['judge']
                print(f"  Judge:")
                print(f"    Total: {judge.get('total', 0)}")
                if 'percentages' in judge:
                    for cat, pct in judge['percentages'].items():
                        print(f"    {cat}: {pct:.1f}%")
        
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python script/view_training_metrics.py <metrics_file>")
        print("\nExamples:")
        print("  python script/view_training_metrics.py results/experiment_metrics_summary.json")
        print("  python script/view_training_metrics.py results/experiment_metrics.jsonl")
        sys.exit(1)
    
    metrics_file = sys.argv[1]
    
    if not Path(metrics_file).exists():
        print(f"Error: File not found: {metrics_file}")
        sys.exit(1)
    
    if metrics_file.endswith('_summary.json'):
        view_summary(metrics_file)
    elif metrics_file.endswith('.jsonl'):
        last_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        view_detailed(metrics_file, last_n)
    else:
        print(f"Error: Unknown file type. Expected .json or .jsonl")
        sys.exit(1)


if __name__ == "__main__":
    main()
