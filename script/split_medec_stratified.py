#!/usr/bin/env python3
"""
Stratified MEDEC Data Splitting Script

This script:
1. Loads the full MEDEC training set
2. Identifies already processed note IDs from existing data
3. Stratifies remaining notes by error type
4. Splits into 75% educational / 25% adaptation
5. Saves split assignments to JSON files

Maintains exact error type proportions from MEDEC in both splits.
"""

import os
import csv
import json
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict


def load_medec_data(csv_path):
    """Load MEDEC training set and filter for error cases."""
    print(f"📂 Loading MEDEC data from {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row['Error Flag'] == '1']
    
    print(f"   Found {len(rows)} error cases")
    return rows


def load_existing_note_ids(educational_path, adaptation_path):
    """Load already processed note IDs from existing data files."""
    edu_ids = set()
    adapt_ids = set()
    
    # Load educational IDs
    if os.path.exists(educational_path):
        print(f"📂 Loading existing educational data from {educational_path}")
        with open(educational_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'metadata' in item and 'original_id' in item['metadata']:
                        edu_ids.add(item['metadata']['original_id'])
                except json.JSONDecodeError:
                    continue
        print(f"   Found {len(edu_ids)} unique note IDs")
    else:
        print(f"⚠️  Educational data file not found: {educational_path}")
    
    # Load adaptation IDs
    if os.path.exists(adaptation_path):
        print(f"📂 Loading existing adaptation data from {adaptation_path}")
        with open(adaptation_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'metadata' in item and 'original_id' in item['metadata']:
                        adapt_ids.add(item['metadata']['original_id'])
                except json.JSONDecodeError:
                    continue
        print(f"   Found {len(adapt_ids)} unique note IDs")
    else:
        print(f"⚠️  Adaptation data file not found: {adaptation_path}")
    
    return edu_ids, adapt_ids


def stratify_by_error_type(rows, edu_ratio=0.75, adapt_ratio=0.25, 
                           existing_edu_ids=None, existing_adapt_ids=None,
                           random_seed=42):
    """
    Stratify notes by error type into educational and adaptation splits.
    
    Maintains exact error type proportions from MEDEC in both splits.
    Excludes already processed note IDs.
    """
    random.seed(random_seed)
    
    existing_edu_ids = existing_edu_ids or set()
    existing_adapt_ids = existing_adapt_ids or set()
    all_existing_ids = existing_edu_ids | existing_adapt_ids
    
    # Group notes by error type
    notes_by_type = defaultdict(list)
    for row in rows:
        note_id = row['Text ID']
        error_type = row['Error Type']
        
        # Skip if already processed
        if note_id not in all_existing_ids:
            notes_by_type[error_type].append(row)
    
    # Calculate total remaining notes
    total_remaining = sum(len(notes) for notes in notes_by_type.values())
    
    print(f"\n📊 Remaining Notes by Error Type:")
    for error_type in sorted(notes_by_type.keys()):
        count = len(notes_by_type[error_type])
        print(f"   {error_type:20s}: {count:4d} notes")
    print(f"   {'TOTAL':20s}: {total_remaining:4d} notes")
    
    # Stratified split: take edu_ratio from each error type
    edu_notes = []
    adapt_notes = []
    
    print(f"\n🔀 Stratified Split ({edu_ratio:.0%} edu / {adapt_ratio:.0%} adapt):")
    
    for error_type in sorted(notes_by_type.keys()):
        notes = notes_by_type[error_type]
        
        # Shuffle for randomness
        random.shuffle(notes)
        
        # Calculate split point
        edu_count = int(len(notes) * edu_ratio)
        
        # Split
        edu_split = notes[:edu_count]
        adapt_split = notes[edu_count:]
        
        edu_notes.extend(edu_split)
        adapt_notes.extend(adapt_split)
        
        print(f"   {error_type:20s}: {len(edu_split):4d} edu + {len(adapt_split):4d} adapt = {len(notes):4d} total")
    
    print(f"   {'TOTAL':20s}: {len(edu_notes):4d} edu + {len(adapt_notes):4d} adapt = {total_remaining:4d} total")
    
    return edu_notes, adapt_notes


def verify_stratification(medec_rows, edu_notes, adapt_notes):
    """Verify that stratification maintains MEDEC proportions."""
    
    # Get MEDEC distribution
    medec_dist = Counter(row['Error Type'] for row in medec_rows)
    total_medec = len(medec_rows)
    
    # Get split distributions
    edu_dist = Counter(row['Error Type'] for row in edu_notes)
    adapt_dist = Counter(row['Error Type'] for row in adapt_notes)
    
    total_edu = len(edu_notes)
    total_adapt = len(adapt_notes)
    
    print(f"\n✅ Stratification Verification:")
    print(f"\n{'Error Type':20s} | {'MEDEC %':>8s} | {'Edu %':>8s} | {'Δ':>6s} | {'Adapt %':>8s} | {'Δ':>6s}")
    print("-" * 80)
    
    max_deviation = 0.0
    
    for error_type in sorted(medec_dist.keys()):
        medec_pct = (medec_dist[error_type] / total_medec) * 100
        edu_pct = (edu_dist[error_type] / total_edu) * 100 if total_edu > 0 else 0
        adapt_pct = (adapt_dist[error_type] / total_adapt) * 100 if total_adapt > 0 else 0
        
        edu_delta = abs(edu_pct - medec_pct)
        adapt_delta = abs(adapt_pct - medec_pct)
        
        max_deviation = max(max_deviation, edu_delta, adapt_delta)
        
        print(f"{error_type:20s} | {medec_pct:7.2f}% | {edu_pct:7.2f}% | {edu_delta:5.2f}% | {adapt_pct:7.2f}% | {adapt_delta:5.2f}%")
    
    print("-" * 80)
    print(f"Maximum deviation: {max_deviation:.2f}%")
    
    if max_deviation < 0.5:
        print("✅ Excellent stratification! (<0.5% deviation)")
    elif max_deviation < 1.0:
        print("✅ Good stratification! (<1.0% deviation)")
    else:
        print("⚠️  Stratification deviation >1.0%")
    
    return max_deviation


def save_split_data(edu_notes, adapt_notes, output_dir, existing_edu_ids, existing_adapt_ids):
    """Save split assignments to JSON files."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare educational split data
    edu_data = {
        'note_ids': [row['Text ID'] for row in edu_notes],
        'error_types': {row['Text ID']: row['Error Type'] for row in edu_notes},
        'count': len(edu_notes),
        'existing_count': len(existing_edu_ids),
        'total_count': len(edu_notes) + len(existing_edu_ids)
    }
    
    # Prepare adaptation split data
    adapt_data = {
        'note_ids': [row['Text ID'] for row in adapt_notes],
        'error_types': {row['Text ID']: row['Error Type'] for row in adapt_notes},
        'count': len(adapt_notes),
        'existing_count': len(existing_adapt_ids),
        'total_count': len(adapt_notes) + len(existing_adapt_ids)
    }
    
    # Save educational split
    edu_path = output_dir / 'educational_remaining.json'
    with open(edu_path, 'w', encoding='utf-8') as f:
        json.dump(edu_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved educational split: {edu_path}")
    print(f"   New notes: {edu_data['count']}")
    print(f"   Existing notes: {edu_data['existing_count']}")
    print(f"   Total target: {edu_data['total_count']}")
    
    # Save adaptation split
    adapt_path = output_dir / 'adaptation_remaining.json'
    with open(adapt_path, 'w', encoding='utf-8') as f:
        json.dump(adapt_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved adaptation split: {adapt_path}")
    print(f"   New notes: {adapt_data['count']}")
    print(f"   Existing notes: {adapt_data['existing_count']}")
    print(f"   Total target: {adapt_data['total_count']}")
    
    # Save summary
    summary = {
        'educational': {
            'new_notes': edu_data['count'],
            'existing_notes': edu_data['existing_count'],
            'total_notes': edu_data['total_count'],
            'examples_per_note': 4,
            'new_examples': edu_data['count'] * 4,
            'total_examples': edu_data['total_count'] * 4
        },
        'adaptation': {
            'new_notes': adapt_data['count'],
            'existing_notes': adapt_data['existing_count'],
            'total_notes': adapt_data['total_count'],
            'examples_per_note': 4,
            'new_examples': adapt_data['count'] * 4,
            'total_examples': adapt_data['total_count'] * 4
        },
        'total': {
            'new_notes': edu_data['count'] + adapt_data['count'],
            'new_examples': (edu_data['count'] + adapt_data['count']) * 4,
            'total_notes': edu_data['total_count'] + adapt_data['total_count'],
            'total_examples': (edu_data['total_count'] + adapt_data['total_count']) * 4
        }
    }
    
    summary_path = output_dir / 'split_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved summary: {summary_path}")
    
    return edu_path, adapt_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Stratified MEDEC data splitting (75% edu / 25% adapt)"
    )
    parser.add_argument(
        '--medec_path',
        type=str,
        default='data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv',
        help='Path to MEDEC training CSV'
    )
    parser.add_argument(
        '--educational_data',
        type=str,
        default='data/sft_training/20251017_161801_sft_merged.jsonl',
        help='Path to existing educational SFT data'
    )
    parser.add_argument(
        '--adaptation_data',
        type=str,
        default='data/adaptation/game_format_adaptation.jsonl',
        help='Path to existing adaptation data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/splits',
        help='Output directory for split files'
    )
    parser.add_argument(
        '--edu_ratio',
        type=float,
        default=0.75,
        help='Ratio for educational split (default: 0.75)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STRATIFIED MEDEC DATA SPLITTING")
    print("=" * 80)
    print(f"\nGoal: Split remaining MEDEC notes into {args.edu_ratio:.0%} edu / {1-args.edu_ratio:.0%} adapt")
    print(f"Strategy: Stratified by error type (maintains MEDEC proportions)")
    print(f"Random seed: {args.random_seed}")
    
    # Load MEDEC data
    medec_rows = load_medec_data(args.medec_path)
    
    # Load existing processed IDs
    print(f"\n📊 Checking Existing Data:")
    existing_edu_ids, existing_adapt_ids = load_existing_note_ids(
        args.educational_data,
        args.adaptation_data
    )
    
    overlap = existing_edu_ids & existing_adapt_ids
    if overlap:
        print(f"⚠️  Warning: {len(overlap)} notes appear in both splits!")
        print(f"   This should not happen. Check your data.")
    else:
        print(f"✅ No overlap between existing splits")
    
    print(f"\n📈 Current Progress:")
    print(f"   Educational: {len(existing_edu_ids)} notes ({len(existing_edu_ids) * 4} examples)")
    print(f"   Adaptation: {len(existing_adapt_ids)} notes ({len(existing_adapt_ids) * 4} examples)")
    print(f"   Total processed: {len(existing_edu_ids | existing_adapt_ids)} notes")
    print(f"   Total available: {len(medec_rows)} notes")
    print(f"   Remaining: {len(medec_rows) - len(existing_edu_ids | existing_adapt_ids)} notes")
    
    # Stratified split
    adapt_ratio = 1.0 - args.edu_ratio
    edu_notes, adapt_notes = stratify_by_error_type(
        medec_rows,
        edu_ratio=args.edu_ratio,
        adapt_ratio=adapt_ratio,
        existing_edu_ids=existing_edu_ids,
        existing_adapt_ids=existing_adapt_ids,
        random_seed=args.random_seed
    )
    
    # Verify stratification
    max_deviation = verify_stratification(medec_rows, edu_notes, adapt_notes)
    
    # Save split data
    edu_path, adapt_path, summary_path = save_split_data(
        edu_notes,
        adapt_notes,
        args.output_dir,
        existing_edu_ids,
        existing_adapt_ids
    )
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("✅ STRATIFICATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n📋 Summary:")
    print(f"   Educational: {len(edu_notes)} new notes + {len(existing_edu_ids)} existing = {len(edu_notes) + len(existing_edu_ids)} total")
    print(f"   Adaptation: {len(adapt_notes)} new notes + {len(existing_adapt_ids)} existing = {len(adapt_notes) + len(existing_adapt_ids)} total")
    print(f"   Total: {len(edu_notes) + len(adapt_notes)} new notes to generate")
    print(f"   Examples: {(len(edu_notes) + len(adapt_notes)) * 4} new examples")
    print(f"   Max deviation: {max_deviation:.2f}%")
    
    print(f"\n📁 Output Files:")
    print(f"   {edu_path}")
    print(f"   {adapt_path}")
    print(f"   {summary_path}")
    
    print(f"\n🚀 Next Steps:")
    print(f"\n1. Generate educational SFT data:")
    print(f"   python3 script/generate_sft_data.py \\")
    print(f"     --medec_path {args.medec_path} \\")
    print(f"     --note_ids_file {edu_path} \\")
    print(f"     --api_provider openai \\")
    print(f"     --model gpt-5 \\")
    print(f"     --output_dir data/sft_training")
    
    print(f"\n2. Generate adaptation data:")
    print(f"   python3 script/generate_game_format_adaptation.py \\")
    print(f"     --medec_path {args.medec_path} \\")
    print(f"     --note_ids_file {adapt_path} \\")
    print(f"     --output_path data/adaptation/game_adaptation_expansion.jsonl")
    
    print(f"\n" + "=" * 80)


if __name__ == '__main__':
    main()
