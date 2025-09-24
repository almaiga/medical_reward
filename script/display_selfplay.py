#!/usr/bin/env python3

import json
import argparse
import sys
from collections import defaultdict
from datetime import datetime

def load_jsonl(file_path):
    """Load JSONL file and return list of parsed JSON objects"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        sys.exit(1)

def parse_training_data(data):
    """Parse the JSONL training data into rounds"""
    rounds = defaultdict(lambda: {'attacker_training': [], 'assessor_training': []})
    
    for entry in data:
        if 'round' in entry and 'phase' in entry:
            round_num = entry['round']
            phase = entry['phase']
            if phase in ['attacker_training', 'assessor_training']:
                rounds[round_num][phase].append(entry)
    
    return dict(rounds)

def display_header(file_path, total_rounds, display_rounds):
    """Display the header information"""
    print("🎮 MEDICAL REWARD SELF-PLAY GAME ANALYSIS")
    print("=" * 60)
    print(f"📄 File: {file_path}")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Total Rounds: {total_rounds}")
    print(f"👁️  Displaying: {display_rounds} rounds")
    print()

def display_game_summary(rounds):
    """Display overall game statistics"""
    total_attacks = sum(len(round_data['attacker_training']) for round_data in rounds.values())
    total_assessments = sum(len(round_data['assessor_training']) for round_data in rounds.values())
    
    print("📊 OVERALL STATISTICS")
    print("-" * 30)
    print(f"Total Attack Games: {total_attacks}")
    print(f"Total Assessment Games: {total_assessments}")
    print()

def display_round_details(round_num, round_data):
    """Display detailed information for a specific round"""
    print(f"🎯 ROUND {round_num}")
    print("-" * 40)
    
    # Attacker phase
    attacker_games = round_data['attacker_training']
    if attacker_games:
        print(f"📈 ATTACKER PHASE ({len(attacker_games)} games)")
        
        successful_deceptions = 0
        total_rewards = []
        harm_levels_created = defaultdict(int)
        
        for game in attacker_games:
            # Count successful deceptions (when assessor was wrong)
            if not game['judge_assessment']['assessor_was_correct']:
                successful_deceptions += 1
            
            total_rewards.append(game['rewards']['total'])
            harm_levels_created[game['judge_assessment']['actual_harm']] += 1
        
        deception_rate = (successful_deceptions / len(attacker_games)) * 100
        avg_reward = sum(total_rewards) / len(total_rewards)
        
        print(f"   • Successful Deceptions: {successful_deceptions}/{len(attacker_games)} ({deception_rate:.1f}%)")
        print(f"   • Average Reward: {avg_reward:.2f}")
        print(f"   • Harm Created: Low={harm_levels_created['Low']}, Med={harm_levels_created['Medium']}, High={harm_levels_created['High']}")
        
        # Show sample interaction
        if attacker_games:
            sample = attacker_games[0]
            print(f"\n   📋 Sample Attack:")
            original_preview = sample['original_note'][:80] + "..." if len(sample['original_note']) > 80 else sample['original_note']
            attacked_preview = sample['attacked_note'][:80] + "..." if len(sample['attacked_note']) > 80 else sample['attacked_note']
            print(f"      Original: {original_preview}")
            print(f"      Attacked: {attacked_preview}")
            print(f"      Assessor Said: {sample['assessor_response']['label']}")
            print(f"      True Harm: {sample['judge_assessment']['actual_harm']}")
    
    # Assessor phase
    assessor_games = round_data['assessor_training']
    if assessor_games:
        print(f"\n🔍 ASSESSOR PHASE ({len(assessor_games)} games)")
        
        correct_assessments = 0
        harm_distribution = defaultdict(int)
        accuracy_by_harm = defaultdict(list)
        
        for game in assessor_games:
            if game['judge_assessment']['assessor_was_correct']:
                correct_assessments += 1
            
            harm = game['judge_assessment']['actual_harm']
            correct = game['judge_assessment']['assessor_was_correct']
            
            harm_distribution[harm] += 1
            accuracy_by_harm[harm].append(correct)
        
        overall_accuracy = (correct_assessments / len(assessor_games)) * 100
        print(f"   • Overall Accuracy: {correct_assessments}/{len(assessor_games)} ({overall_accuracy:.1f}%)")
        
        print(f"   • Performance by Harm Level:")
        for harm in ['Low', 'Medium', 'High']:
            if harm in accuracy_by_harm:
                count = harm_distribution[harm]
                accuracy = (sum(accuracy_by_harm[harm]) / len(accuracy_by_harm[harm])) * 100
                print(f"      {harm}: {count} cases, {accuracy:.1f}% accuracy")
    
    print()

def display_learning_progression(rounds, max_rounds):
    """Show learning progression across rounds"""
    print("📈 LEARNING PROGRESSION")
    print("-" * 30)
    
    for round_num in sorted(rounds.keys()):
        if round_num > max_rounds:
            break
            
        round_data = rounds[round_num]
        
        # Attacker performance
        attacker_games = round_data['attacker_training']
        if attacker_games:
            successful = sum(1 for game in attacker_games 
                           if not game['judge_assessment']['assessor_was_correct'])
            deception_rate = (successful / len(attacker_games)) * 100
            avg_reward = sum(game['rewards']['total'] for game in attacker_games) / len(attacker_games)
            attacker_info = f"Deception: {deception_rate:.1f}%, Reward: {avg_reward:.2f}"
        else:
            attacker_info = "No data"
        
        # Assessor performance
        assessor_games = round_data['assessor_training']
        if assessor_games:
            correct = sum(1 for game in assessor_games 
                         if game['judge_assessment']['assessor_was_correct'])
            accuracy = (correct / len(assessor_games)) * 100
            assessor_info = f"Accuracy: {accuracy:.1f}%"
        else:
            assessor_info = "No data"
        
        print(f"Round {round_num}: Attacker[{attacker_info}] | Assessor[{assessor_info}]")

def main():
    parser = argparse.ArgumentParser(description='Display self-play training results')
    parser.add_argument('file', help='Path to the JSONL interaction file')
    parser.add_argument('rounds', type=int, help='Number of rounds to display')
    parser.add_argument('--summary-only', action='store_true', 
                       help='Show only summary statistics')
    
    args = parser.parse_args()
    
    # Load and parse data
    data = load_jsonl(args.file)
    rounds = parse_training_data(data)
    
    if not rounds:
        print("❌ No training rounds found in the file")
        sys.exit(1)
    
    total_rounds = max(rounds.keys())
    display_rounds = min(args.rounds, total_rounds)
    
    # Display results
    display_header(args.file, total_rounds, display_rounds)
    display_game_summary(rounds)
    display_learning_progression(rounds, display_rounds)
    
    if not args.summary_only:
        print("\n" + "="*60)
        print("DETAILED ROUND ANALYSIS")
        print("="*60)
        
        for round_num in sorted(rounds.keys()):
            if round_num <= display_rounds:
                display_round_details(round_num, rounds[round_num])

if __name__ == "__main__":
    main()