#!/usr/bin/env python3
"""
Create proper stratified 75/25 split for Educational vs Adaptation data.

Goal:
- Educational: 75% of each error type
- Adaptation: 25% of each error type
- No overlap
- All 5 error types in both splits
"""

import csv
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

def main():
    random.seed(42)  # Reproducible
    
    print("=" * 70)
    print("PROPER STRATIFIED 75/25 SPLIT")
    print("=" * 70)
    
    # Load MEDEC
    medec_path = "data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
    with open(medec_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_error_notes = [
            r for r in reader 
            if r['Error Flag'] == '1' 
            and r['Text'].strip() 
            and r['Corrected Text'].strip()
        ]
    
    print(f"\n📊 Total MEDEC error notes: {len(all_error_notes)}")
    
    # Group by error type
    notes_by_type = defaultdict(list)
    for note in all_error_notes:
        notes_by_type[note['Error Type']].append(note)
    
    # Stratified split
    educational_notes = []
    adaptation_notes = []
    
    print(f"\n📋 Stratified split by error type:")
    
    for error_type in sorted(notes_by_type.keys()):
        notes = notes_by_type[error_type]
        random.shuffle(notes)  # Randomize within type
        
        # 75/25 split
        split_idx = int(len(notes) * 0.75)
        edu_notes = notes[:split_idx]
        adapt_notes = notes[split_idx:]
        
        educational_notes.extend(edu_notes)
        adaptation_notes.extend(adapt_notes)
        
        print(f"\n  {error_type}: {len(notes)} total")
        print(f"    → Educational: {len(edu_notes)} ({len(edu_notes)/len(notes)*100:.1f}%)")
        print(f"    → Adaptation: {len(adapt_notes)} ({len(adapt_notes)/len(notes)*100:.1f}%)")
    
    # Extract IDs
    educational_ids = [n['Text ID'] for n in educational_notes]
    adaptation_ids = [n['Text ID'] for n in adaptation_notes]
    
    print(f"\n{'=' * 70}")
    print(f"RESULTS:")
    print(f"  Educational: {len(educational_ids)} notes ({len(educational_ids)/len(all_error_notes)*100:.1f}%)")
    print(f"  Adaptation: {len(adaptation_ids)} notes ({len(adaptation_ids)/len(all_error_notes)*100:.1f}%)")
    print(f"  Adaptation examples: {len(adaptation_ids)} × 4 = {len(adaptation_ids) * 4}")
    
    # Verify no overlap
    overlap = set(educational_ids) & set(adaptation_ids)
    print(f"\n  Overlap: {len(overlap)} notes ✓")
    
    # Count error types in each split
    edu_errors = Counter([n['Error Type'] for n in educational_notes])
    adapt_errors = Counter([n['Error Type'] for n in adaptation_notes])
    
    print(f"\n  Educational error types:")
    for error_type, count in sorted(edu_errors.items()):
        print(f"    • {error_type}: {count}")
    
    print(f"\n  Adaptation error types:")
    for error_type, count in sorted(adapt_errors.items()):
        print(f"    • {error_type}: {count}")
    
    # Save splits
    output_dir = Path("data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save educational split
    edu_output = output_dir / "educational_stratified.json"
    with open(edu_output, 'w') as f:
        json.dump({
            "note_ids": educational_ids,
            "count": len(educational_ids),
            "error_types": dict(edu_errors),
            "split": "75% stratified"
        }, f, indent=2)
    print(f"\n💾 Saved educational split: {edu_output}")
    
    # Save adaptation split
    adapt_output = output_dir / "adaptation_stratified.json"
    with open(adapt_output, 'w') as f:
        json.dump({
            "note_ids": adaptation_ids,
            "count": len(adaptation_ids),
            "error_types": dict(adapt_errors),
            "split": "25% stratified"
        }, f, indent=2)
    print(f"💾 Saved adaptation split: {adapt_output}")
    
    # Compare with existing educational data
    print(f"\n{'=' * 70}")
    print(f"COMPARISON WITH EXISTING DATA:")
    
    existing_edu_files = [
        'data/sft_training/20251107_044104_openai_gpt-5_sft.jsonl',
        'data/sft_training/20251017_161801_sft_merged.jsonl'
    ]
    
    existing_edu_ids = set()
    for filepath in existing_edu_files:
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    existing_edu_ids.add(data['metadata']['original_id'])
        except FileNotFoundError:
            pass
    
    if existing_edu_ids:
        print(f"\n  Existing educational: {len(existing_edu_ids)} notes")
        print(f"  New educational: {len(educational_ids)} notes")
        
        # How many can we reuse?
        reusable = existing_edu_ids & set(educational_ids)
        need_to_generate = set(educational_ids) - existing_edu_ids
        
        print(f"\n  Can reuse: {len(reusable)} notes ({len(reusable)/len(educational_ids)*100:.1f}%)")
        print(f"  Need to generate: {len(need_to_generate)} notes ({len(need_to_generate)/len(educational_ids)*100:.1f}%)")
        
        if len(need_to_generate) > 0:
            # Save notes that need generation
            need_gen_output = output_dir / "educational_need_generation.json"
            with open(need_gen_output, 'w') as f:
                json.dump({
                    "note_ids": sorted(list(need_to_generate)),
                    "count": len(need_to_generate),
                    "purpose": "Educational notes missing from existing data"
                }, f, indent=2)
            print(f"\n💾 Saved notes needing generation: {need_gen_output}")
    
    print(f"\n{'=' * 70}")
    print(f"NEXT STEPS:")
    print(f"\n1. Generate adaptation data ({len(adaptation_ids)} notes → {len(adaptation_ids) * 4} examples):")
    print(f"   python3 script/generate_game_format_adaptation.py \\")
    print(f"     --note_ids_file data/splits/adaptation_stratified.json \\")
    print(f"     --use_gpt \\")
    print(f"     --output_path data/adaptation/game_format_adaptation_stratified.jsonl")
    
    if len(need_to_generate) > 0:
        print(f"\n2. Generate missing educational data ({len(need_to_generate)} notes):")
        print(f"   [Use your educational SFT generation script]")
        print(f"   --note_ids_file data/splits/educational_need_generation.json")
    else:
        print(f"\n2. Educational data is complete! ✓")
    
    print(f"\n{'=' * 70}")

if __name__ == "__main__":
    main()
