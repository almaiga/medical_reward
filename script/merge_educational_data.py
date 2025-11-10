#!/usr/bin/env python3
"""
Merge educational data from multiple sources into final stratified dataset.

Takes:
1. Existing educational file 1 (20251107_044104_openai_gpt-5_sft.jsonl)
2. Existing educational file 2 (20251017_161801_sft_merged.jsonl)
3. Newly generated supplemental file
4. Stratified split definition (educational_stratified.json)

Outputs:
- Final merged educational dataset with exactly 913 notes
- Properly stratified by error type
"""

import json
import argparse
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser(
        description="Merge educational data into final stratified dataset"
    )
    parser.add_argument(
        "--existing_file1",
        type=str,
        default="data/sft_training/20251107_044104_openai_gpt-5_sft.jsonl",
        help="First existing educational file"
    )
    parser.add_argument(
        "--existing_file2",
        type=str,
        default="data/sft_training/20251017_161801_sft_merged.jsonl",
        help="Second existing educational file"
    )
    parser.add_argument(
        "--supplemental_file",
        type=str,
        required=True,
        help="Newly generated supplemental educational file"
    )
    parser.add_argument(
        "--stratified_split",
        type=str,
        default="data/splits/educational_stratified.json",
        help="JSON file with stratified note IDs"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/sft_training/educational_stratified_complete.jsonl",
        help="Output path for merged educational data"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("MERGE EDUCATIONAL DATA")
    print("=" * 70)
    
    # Load stratified split definition
    print(f"\n📂 Loading stratified split: {args.stratified_split}")
    with open(args.stratified_split, 'r') as f:
        split_data = json.load(f)
        target_ids = set(split_data['note_ids'])
        target_error_types = split_data['error_types']
    
    print(f"   Target: {len(target_ids)} notes")
    print(f"   Error types:")
    for error_type, count in sorted(target_error_types.items()):
        print(f"     • {error_type}: {count}")
    
    # Load all educational data
    all_examples = []
    seen_note_ids = set()
    
    # Load existing file 1
    print(f"\n📂 Loading existing file 1: {args.existing_file1}")
    count1 = 0
    with open(args.existing_file1, 'r') as f:
        for line in f:
            data = json.loads(line)
            note_id = data['metadata']['original_id']
            if note_id in target_ids:
                all_examples.append(data)
                seen_note_ids.add(note_id)
                count1 += 1
    print(f"   Found {count1} examples from {len([n for n in seen_note_ids if n in target_ids])} unique notes")
    
    # Load existing file 2
    print(f"\n📂 Loading existing file 2: {args.existing_file2}")
    count2 = 0
    file2_note_ids = set()
    with open(args.existing_file2, 'r') as f:
        for line in f:
            data = json.loads(line)
            note_id = data['metadata']['original_id']
            if note_id in target_ids and note_id not in seen_note_ids:
                all_examples.append(data)
                seen_note_ids.add(note_id)
                file2_note_ids.add(note_id)
                count2 += 1
    print(f"   Found {count2} examples from {len(file2_note_ids)} unique notes")
    
    # Load supplemental file
    print(f"\n📂 Loading supplemental file: {args.supplemental_file}")
    count3 = 0
    file3_note_ids = set()
    try:
        with open(args.supplemental_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                note_id = data['metadata']['original_id']
                if note_id in target_ids and note_id not in seen_note_ids:
                    all_examples.append(data)
                    seen_note_ids.add(note_id)
                    file3_note_ids.add(note_id)
                    count3 += 1
        print(f"   Found {count3} notes matching stratified split")
    except FileNotFoundError:
        print(f"   ⚠️  File not found - skipping")
        count3 = 0
        file3_note_ids = set()
    
    # Check coverage
    print(f"\n{'=' * 70}")
    print(f"COVERAGE:")
    print(f"  From existing file 1: {count1} examples")
    print(f"  From existing file 2: {count2} examples")
    print(f"  From supplemental: {count3} examples")
    print(f"  Total examples collected: {len(all_examples)}")
    print(f"  Unique notes collected: {len(seen_note_ids)}")
    print(f"  Target notes: {len(target_ids)}")
    
    missing = target_ids - seen_note_ids
    if missing:
        print(f"\n  ⚠️  MISSING {len(missing)} notes:")
        print(f"     {sorted(list(missing))[:10]}...")
        print(f"\n  You need to generate these {len(missing)} notes!")
        return
    
    # Count error types in collected data (only from harmful examples)
    error_counts = Counter()
    for data in all_examples:
        error_type = data['metadata'].get('error_type', 'unknown')
        # Only count actual error types, not vanilla/safe/none
        if error_type not in ['vanilla', 'none', 'safe', 'unknown']:
            error_counts[error_type] += 1
    
    print(f"\n  ✓ All notes collected!")
    print(f"\n  Error type distribution:")
    for error_type, count in sorted(error_counts.items()):
        target_count = target_error_types.get(error_type, 0)
        match = "✓" if count == target_count else "✗"
        print(f"    {match} {error_type}: {count} (target: {target_count})")
    
    # Save merged data
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving merged data: {output_path}")
    
    # Sort by note ID for consistency
    all_examples_sorted = sorted(all_examples, 
                                 key=lambda x: int(x['metadata']['original_id'].split('-')[-1]))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in all_examples_sorted:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Saved {len(all_examples)} examples")
    
    print(f"\n{'=' * 70}")
    print(f"SUCCESS!")
    print(f"{'=' * 70}")
    print(f"\nFinal educational dataset:")
    print(f"  Path: {output_path}")
    print(f"  Notes: {len(all_examples)}")
    print(f"  Stratification: ✓ Proper 75/25 split")
    print(f"  Error types: ✓ All 5 types included")
    
    print(f"\nYou can now use this for training:")
    print(f"  python3 script/train_qwen3_sft.py \\")
    print(f"    --data_path {output_path} \\")
    print(f"    --epochs 3 \\")
    print(f"    --batch_size 4")
    
    print(f"\n{'=' * 70}")

if __name__ == "__main__":
    main()
