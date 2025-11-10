#!/usr/bin/env python3
"""
Merge and Validate Training Data

This script:
1. Merges existing + new educational SFT data
2. Merges existing + new adaptation data
3. Validates stratification is maintained
4. Validates format consistency
5. Checks for duplicates
6. Generates final statistics
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def load_jsonl(file_path):
    """Load JSONL file."""
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return []
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing line {line_num} in {file_path}: {e}")
    
    return data


def validate_format(examples):
    """Validate CoT format in examples."""
    valid_count = 0
    invalid_examples = []
    
    for i, example in enumerate(examples):
        if 'messages' not in example:
            invalid_examples.append((i, "Missing 'messages' field"))
            continue
        
        # Check assistant message
        assistant_msg = None
        for msg in example['messages']:
            if msg.get('role') == 'assistant':
                assistant_msg = msg.get('content', '')
                break
        
        if not assistant_msg:
            invalid_examples.append((i, "No assistant message"))
            continue
        
        # Check for CoT format
        has_think = '<think>' in assistant_msg and '</think>' in assistant_msg
        has_output = '<output>' in assistant_msg and '</output>' in assistant_msg
        
        if has_think and has_output:
            valid_count += 1
        else:
            invalid_examples.append((i, f"Invalid format (think:{has_think}, output:{has_output})"))
    
    return valid_count, invalid_examples


def check_duplicates(examples):
    """Check for duplicate note IDs."""
    note_ids = []
    for example in examples:
        if 'metadata' in example and 'original_id' in example['metadata']:
            note_ids.append(example['metadata']['original_id'])
    
    id_counts = Counter(note_ids)
    duplicates = {note_id: count for note_id, count in id_counts.items() if count > 4}
    
    return duplicates


def analyze_distribution(examples):
    """Analyze error type distribution."""
    error_types = []
    roles = []
    
    for example in examples:
        if 'metadata' in example:
            if 'error_type' in example['metadata']:
                error_type = example['metadata']['error_type']
                if error_type not in ['vanilla', 'safe', 'none']:
                    error_types.append(error_type)
            
            if 'role' in example['metadata']:
                roles.append(example['metadata']['role'])
    
    return Counter(error_types), Counter(roles)


def merge_and_save(existing_data, new_data, output_path):
    """Merge existing and new data, save to output."""
    merged = existing_data + new_data
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge and validate training data"
    )
    parser.add_argument(
        '--existing_educational',
        type=str,
        default='data/sft_training/20251017_161801_sft_merged.jsonl',
        help='Existing educational SFT data'
    )
    parser.add_argument(
        '--new_educational',
        type=str,
        required=True,
        help='New educational SFT data to merge'
    )
    parser.add_argument(
        '--existing_adaptation',
        type=str,
        default='data/adaptation/game_format_adaptation.jsonl',
        help='Existing adaptation data'
    )
    parser.add_argument(
        '--new_adaptation',
        type=str,
        required=True,
        help='New adaptation data to merge'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for merged files'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MERGE AND VALIDATE TRAINING DATA")
    print("=" * 80)
    
    # Load educational data
    print("\n📂 Loading Educational SFT Data:")
    existing_edu = load_jsonl(args.existing_educational)
    new_edu = load_jsonl(args.new_educational)
    print(f"   Existing: {len(existing_edu)} examples")
    print(f"   New: {len(new_edu)} examples")
    
    # Load adaptation data
    print("\n📂 Loading Adaptation Data:")
    existing_adapt = load_jsonl(args.existing_adaptation)
    new_adapt = load_jsonl(args.new_adaptation)
    print(f"   Existing: {len(existing_adapt)} examples")
    print(f"   New: {len(new_adapt)} examples")
    
    # Merge educational
    print("\n🔀 Merging Educational Data...")
    edu_output = f"{args.output_dir}/sft_training/educational_sft_complete.jsonl"
    merged_edu = merge_and_save(existing_edu, new_edu, edu_output)
    print(f"   ✅ Merged {len(merged_edu)} examples → {edu_output}")
    
    # Merge adaptation
    print("\n🔀 Merging Adaptation Data...")
    adapt_output = f"{args.output_dir}/adaptation/game_adaptation_complete.jsonl"
    merged_adapt = merge_and_save(existing_adapt, new_adapt, adapt_output)
    print(f"   ✅ Merged {len(merged_adapt)} examples → {adapt_output}")
    
    # Validate educational format
    print("\n✅ Validating Educational Format...")
    edu_valid, edu_invalid = validate_format(merged_edu)
    edu_valid_pct = (edu_valid / len(merged_edu)) * 100 if merged_edu else 0
    print(f"   Valid: {edu_valid}/{len(merged_edu)} ({edu_valid_pct:.1f}%)")
    if edu_invalid:
        print(f"   ⚠️  {len(edu_invalid)} invalid examples")
        for idx, reason in edu_invalid[:5]:
            print(f"      Example {idx}: {reason}")
    
    # Validate adaptation format
    print("\n✅ Validating Adaptation Format...")
    adapt_valid, adapt_invalid = validate_format(merged_adapt)
    adapt_valid_pct = (adapt_valid / len(merged_adapt)) * 100 if merged_adapt else 0
    print(f"   Valid: {adapt_valid}/{len(merged_adapt)} ({adapt_valid_pct:.1f}%)")
    if adapt_invalid:
        print(f"   ⚠️  {len(adapt_invalid)} invalid examples")
        for idx, reason in adapt_invalid[:5]:
            print(f"      Example {idx}: {reason}")
    
    # Check duplicates
    print("\n🔍 Checking for Duplicates...")
    edu_dupes = check_duplicates(merged_edu)
    adapt_dupes = check_duplicates(merged_adapt)
    
    if edu_dupes:
        print(f"   ⚠️  Educational: {len(edu_dupes)} notes with >4 examples")
        for note_id, count in list(edu_dupes.items())[:5]:
            print(f"      {note_id}: {count} examples")
    else:
        print(f"   ✅ Educational: No duplicates (4 examples per note)")
    
    if adapt_dupes:
        print(f"   ⚠️  Adaptation: {len(adapt_dupes)} notes with >4 examples")
        for note_id, count in list(adapt_dupes.items())[:5]:
            print(f"      {note_id}: {count} examples")
    else:
        print(f"   ✅ Adaptation: No duplicates (4 examples per note)")
    
    # Analyze distributions
    print("\n📊 Educational Distribution:")
    edu_error_types, edu_roles = analyze_distribution(merged_edu)
    print(f"   Total examples: {len(merged_edu)}")
    print(f"   Unique notes: {len(merged_edu) // 4}")
    print(f"\n   Roles:")
    for role, count in edu_roles.items():
        print(f"      {role}: {count} ({count/len(merged_edu)*100:.1f}%)")
    print(f"\n   Error types:")
    for error_type, count in edu_error_types.most_common():
        print(f"      {error_type}: {count} ({count/sum(edu_error_types.values())*100:.1f}%)")
    
    print("\n📊 Adaptation Distribution:")
    adapt_error_types, adapt_roles = analyze_distribution(merged_adapt)
    print(f"   Total examples: {len(merged_adapt)}")
    print(f"   Unique notes: {len(merged_adapt) // 4}")
    print(f"\n   Roles:")
    for role, count in adapt_roles.items():
        print(f"      {role}: {count} ({count/len(merged_adapt)*100:.1f}%)")
    print(f"\n   Error types:")
    for error_type, count in adapt_error_types.most_common():
        print(f"      {error_type}: {count} ({count/sum(adapt_error_types.values())*100:.1f}%)")
    
    # Generate validation report
    report = {
        'educational': {
            'total_examples': len(merged_edu),
            'unique_notes': len(merged_edu) // 4,
            'format_valid': edu_valid,
            'format_valid_pct': edu_valid_pct,
            'duplicates': len(edu_dupes),
            'error_types': dict(edu_error_types),
            'roles': dict(edu_roles)
        },
        'adaptation': {
            'total_examples': len(merged_adapt),
            'unique_notes': len(merged_adapt) // 4,
            'format_valid': adapt_valid,
            'format_valid_pct': adapt_valid_pct,
            'duplicates': len(adapt_dupes),
            'error_types': dict(adapt_error_types),
            'roles': dict(adapt_roles)
        },
        'total': {
            'examples': len(merged_edu) + len(merged_adapt),
            'notes': (len(merged_edu) + len(merged_adapt)) // 4
        }
    }
    
    report_path = f"{args.output_dir}/validation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Validation report saved: {report_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ MERGE AND VALIDATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n📁 Output Files:")
    print(f"   Educational: {edu_output}")
    print(f"   Adaptation: {adapt_output}")
    print(f"   Report: {report_path}")
    
    print(f"\n📊 Final Statistics:")
    print(f"   Total examples: {len(merged_edu) + len(merged_adapt)}")
    print(f"   Educational: {len(merged_edu)} ({len(merged_edu)/(len(merged_edu)+len(merged_adapt))*100:.1f}%)")
    print(f"   Adaptation: {len(merged_adapt)} ({len(merged_adapt)/(len(merged_edu)+len(merged_adapt))*100:.1f}%)")
    print(f"   Format valid: {edu_valid + adapt_valid}/{len(merged_edu) + len(merged_adapt)} ({(edu_valid + adapt_valid)/(len(merged_edu) + len(merged_adapt))*100:.1f}%)")
    
    if edu_valid_pct > 95 and adapt_valid_pct > 95 and not edu_dupes and not adapt_dupes:
        print("\n🎉 All validation checks passed! Ready for training.")
    else:
        print("\n⚠️  Some validation issues found. Review the report.")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
