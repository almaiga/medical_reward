#!/usr/bin/env python3
"""Diagnose class imbalance in selfplay training."""

import json
from collections import defaultdict
import sys

# Read the file path from command line or use default
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = "/Users/josmaiga/Downloads/Trainer Output Qwen3-4B Medical Selfplay Interactions (12).jsonl"

print(f"Analyzing: {file_path}\n")

with open(file_path, 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]

# Filter for actual game interactions (not metadata)
games = [d for d in data if 'rewards' in d and 'game_category' in d.get('rewards', {})]

print(f"Total games: {len(games)}")

# Check structure of first game
if games:
    print("\nFirst game structure:")
    print(f"  Keys: {list(games[0].keys())}")
    if 'judge_assessment' in games[0]:
        print(f"  judge_assessment keys: {list(games[0]['judge_assessment'].keys())}")
print()

# Analyze by game category
by_category = defaultdict(lambda: {'total': 0, 'safe': 0, 'harmful': 0, 'attacker_copied': 0})

for game in games:
    cat = game['rewards']['game_category']
    
    # Handle different data structures
    if 'judgments' in game:
        actual = game['judgments']['actual_harm']
    elif 'judge_assessment' in game:
        actual = game['judge_assessment'].get('actual_harm', 'Unknown')
    else:
        print(f"Warning: No judgment data in game, skipping")
        continue
    
    # Check if attacker just copied
    original = game.get('original_note', '')
    attacked = game.get('attacked_note', '')
    is_copy = (original.strip() == attacked.strip())
    
    by_category[cat]['total'] += 1
    if actual == 'Safe':
        by_category[cat]['safe'] += 1
    else:
        by_category[cat]['harmful'] += 1
    
    if is_copy:
        by_category[cat]['attacker_copied'] += 1

print("DISTRIBUTION BY GAME CATEGORY:")
print("="*60)
for cat in ['vanilla_harmful', 'adversarial_harmful', 'vanilla_benign', 'adversarial_benign']:
    stats = by_category[cat]
    if stats['total'] > 0:
        print(f"\n{cat}: {stats['total']} games")
        print(f"  Safe: {stats['safe']} ({100*stats['safe']/stats['total']:.1f}%)")
        print(f"  Harmful: {stats['harmful']} ({100*stats['harmful']/stats['total']:.1f}%)")
        print(f"  Attacker copied: {stats['attacker_copied']} ({100*stats['attacker_copied']/stats['total']:.1f}%)")
        
        # Expected distribution
        if 'harmful' in cat:
            print(f"  Expected: ~100% Harmful")
            if stats['safe'] > stats['total'] * 0.2:
                print(f"  ⚠️ WARNING: Too many Safe classifications!")
                print(f"     Judge may be ignoring original errors in harmful games")
        else:  # benign
            if 'vanilla' in cat:
                print(f"  Expected: ~100% Safe (vanilla_benign should copy clean notes)")
                if stats['harmful'] > stats['total'] * 0.2:
                    print(f"  ⚠️ WARNING: vanilla_benign should be mostly Safe!")
                    print(f"     Attacker may be injecting errors when it should just copy")
            else:  # adversarial_benign
                print(f"  Expected: Mixed (attacker tries to inject errors)")

print("\n" + "="*60)
print("OVERALL:")
total_safe = sum(s['safe'] for s in by_category.values())
total_harmful = sum(s['harmful'] for s in by_category.values())
total = total_safe + total_harmful
print(f"Safe: {total_safe}/{total} ({100*total_safe/total:.1f}%)")
print(f"Harmful: {total_harmful}/{total} ({100*total_harmful/total:.1f}%)")
print(f"\nExpected: ~50/50 for balanced training")

# Analyze by round
print("\n" + "="*60)
print("DISTRIBUTION BY ROUND:")
print("="*60)

by_round = defaultdict(lambda: {'safe': 0, 'harmful': 0})
for game in games:
    round_num = game.get('round', 0)
    
    # Handle different data structures
    if 'judgments' in game:
        actual = game['judgments']['actual_harm']
    elif 'judge_assessment' in game:
        actual = game['judge_assessment'].get('actual_harm', 'Unknown')
    else:
        continue
    
    if actual == 'Safe':
        by_round[round_num]['safe'] += 1
    else:
        by_round[round_num]['harmful'] += 1

for round_num in sorted(by_round.keys()):
    stats = by_round[round_num]
    total = stats['safe'] + stats['harmful']
    print(f"\nRound {round_num}: {total} games")
    print(f"  Safe: {stats['safe']} ({100*stats['safe']/total:.1f}%)")
    print(f"  Harmful: {stats['harmful']} ({100*stats['harmful']/total:.1f}%)")

# Sample some games to see what's happening
print("\n" + "="*60)
print("SAMPLE GAMES (first 3 of each category):")
print("="*60)

for cat in ['vanilla_harmful', 'adversarial_harmful', 'vanilla_benign', 'adversarial_benign']:
    cat_games = [g for g in games if g['rewards']['game_category'] == cat][:3]
    
    if cat_games:
        print(f"\n{cat}:")
        for i, game in enumerate(cat_games, 1):
            original = game.get('original_note', '')[:100]
            attacked = game.get('attacked_note', '')[:100]
            
            # Handle different data structures
            if 'judgments' in game:
                actual = game['judgments']['actual_harm']
            elif 'judge_assessment' in game:
                actual = game['judge_assessment'].get('actual_harm', 'Unknown')
            else:
                actual = 'Unknown'
            
            is_copy = (game.get('original_note', '').strip() == game.get('attacked_note', '').strip())
            
            print(f"\n  Sample {i}:")
            print(f"    Original: {original}...")
            print(f"    Attacked: {attacked}...")
            print(f"    Is copy: {is_copy}")
            print(f"    Judge said: {actual}")
