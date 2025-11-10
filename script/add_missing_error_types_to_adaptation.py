#!/usr/bin/env python3
"""
Add missing error types (causalOrganism, diagnosis) to adaptation data.

Current situation:
- Educational data (920 notes total from 2 files):
  * 20251017_161801_sft_merged.jsonl: causalOrganism (63), diagnosis (331)
  * 20251107_044104_openai_gpt-5_sft.jsonl: management (345), pharma (90), treatment (87)
- Adaptation data (299 notes): only management, pharmacotherapy, treatment
- Missing from adaptation: causalOrganism and diagnosis
- Problem: ALL causalOrganism and diagnosis notes are in educational data

Solution:
- Select ~20 causalOrganism notes from educational data for adaptation
- Select ~100 diagnosis notes from educational data for adaptation
- Generate game format adaptation data for these
- Keep using the full educational data for SFT (no need to remove them)
- This creates some overlap, which is fine - educational teaches basics,
  adaptation teaches game format
"""

import csv
import json
import random
from pathlib import Path
from collections import Counter

def main():
    # Load MEDEC data
    medec_path = "data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
    print("=" * 70)
    print("ADD MISSING ERROR TYPES TO ADAPTATION DATA")
    print("=" * 70)
    
    # Load all educational IDs from both files
    educational_files = [
        'data/sft_training/20251107_044104_openai_gpt-5_sft.jsonl',
        'data/sft_training/20251017_161801_sft_merged.jsonl'
    ]
    
    all_educational_ids = set()
    for filepath in educational_files:
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                all_educational_ids.add(data['metadata']['original_id'])
    
    print(f"\n📚 Educational data (combined): {len(all_educational_ids)} notes")
    
    with open(medec_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Get causalOrganism and diagnosis notes FROM educational data
    # (since ALL of them are in educational data)
    causal_notes = [
        r for r in rows
        if r['Error Flag'] == '1'
        and r['Error Type'] == 'causalOrganism'
        and r['Text'].strip()
        and r['Corrected Text'].strip()
        and r['Text ID'] in all_educational_ids
    ]
    
    diagnosis_notes = [
        r for r in rows
        if r['Error Flag'] == '1'
        and r['Error Type'] == 'diagnosis'
        and r['Text'].strip()
        and r['Corrected Text'].strip()
        and r['Text ID'] in all_educational_ids
    ]
    
    print(f"\n📊 Available notes (from educational data):")
    print(f"  causalOrganism: {len(causal_notes)} available")
    print(f"  diagnosis: {len(diagnosis_notes)} available")
    print(f"\n  Note: These notes are in educational data but will also be")
    print(f"  used for adaptation (different format/purpose)")
    
    # Select notes for adaptation
    # Use ~30% for adaptation to keep balance
    random.seed(42)  # Reproducible
    
    num_causal = min(20, len(causal_notes))
    num_diagnosis = min(100, len(diagnosis_notes))
    
    selected_causal = random.sample(causal_notes, num_causal)
    selected_diagnosis = random.sample(diagnosis_notes, num_diagnosis)
    
    selected_notes = selected_causal + selected_diagnosis
    selected_ids = [r['Text ID'] for r in selected_notes]
    
    print(f"\n✓ Selected for adaptation:")
    print(f"  causalOrganism: {len(selected_causal)} notes")
    print(f"  diagnosis: {len(selected_diagnosis)} notes")
    print(f"  Total: {len(selected_notes)} notes")
    
    # Show ID ranges
    if selected_causal:
        causal_ids = [int(r['Text ID'].split('-')[-1]) for r in selected_causal]
        print(f"\n📍 ID ranges:")
        print(f"  causalOrganism: ms-train-{min(causal_ids)} to "
              f"ms-train-{max(causal_ids)}")
    
    if selected_diagnosis:
        diagnosis_ids = [int(r['Text ID'].split('-')[-1]) for r in selected_diagnosis]
        if not selected_causal:
            print(f"\n📍 ID ranges:")
        print(f"  diagnosis: ms-train-{min(diagnosis_ids)} to "
              f"ms-train-{max(diagnosis_ids)}")
    
    # Load existing adaptation IDs
    existing_adaptation_path = "data/splits/adaptation_correct.json"
    with open(existing_adaptation_path, 'r') as f:
        existing_data = json.load(f)
        existing_ids = existing_data['note_ids']
    
    print(f"\n📂 Existing adaptation data: {len(existing_ids)} notes")
    
    # Combine with new IDs
    all_adaptation_ids = existing_ids + selected_ids
    
    # Count error types in combined set
    all_adaptation_rows = [
        r for r in rows
        if r['Text ID'] in all_adaptation_ids
        and r['Error Flag'] == '1'
    ]
    error_counts = Counter([r['Error Type'] for r in all_adaptation_rows])
    
    print(f"\n✓ Combined adaptation data: {len(all_adaptation_ids)} notes")
    print(f"\n  Error type distribution:")
    for error_type, count in sorted(error_counts.items()):
        pct = count / len(all_adaptation_rows) * 100
        print(f"    • {error_type}: {count} ({pct:.1f}%)")
    
    # Save supplemental adaptation IDs
    output_path = "data/splits/adaptation_supplemental.json"
    output_data = {
        "note_ids": selected_ids,
        "count": len(selected_ids),
        "error_types": {
            "causalOrganism": len(selected_causal),
            "diagnosis": len(selected_diagnosis)
        },
        "purpose": "Add missing error types to adaptation data"
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved supplemental IDs to: {output_path}")
    
    # Save combined adaptation IDs
    combined_output_path = "data/splits/adaptation_complete.json"
    combined_data = {
        "note_ids": all_adaptation_ids,
        "count": len(all_adaptation_ids),
        "error_types": dict(error_counts),
        "sources": {
            "original": len(existing_ids),
            "supplemental": len(selected_ids)
        }
    }
    
    with open(combined_output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"💾 Saved combined IDs to: {combined_output_path}")
    
    print(f"\n🎯 Next Steps:")
    print(f"\n1. Generate supplemental adaptation data (~{len(selected_ids) * 4} "
          f"examples):")
    print(f"   python3 script/generate_game_format_adaptation.py \\")
    print(f"     --note_ids_file {output_path} \\")
    print(f"     --use_gpt \\")
    print(f"     --output_path data/adaptation/game_format_adaptation_supplemental.jsonl")
    
    print(f"\n2. Merge with existing adaptation data:")
    print(f"   cat data/adaptation/game_format_adaptation.jsonl \\")
    print(f"       data/adaptation/game_format_adaptation_supplemental.jsonl \\")
    print(f"       > data/adaptation/game_format_adaptation_complete.jsonl")
    
    print(f"\n3. Use complete adaptation data for training")
    
    print(f"\n" + "=" * 70)

if __name__ == "__main__":
    main()
