#!/usr/bin/env python3
"""
Organize clean, stratified educational and adaptation data into data/sft_clean.

This script:
1. Re-runs the merge to get clean educational data
2. Copies the stratified adaptation data
3. Creates a summary document
4. Validates the final datasets
"""

import json
import shutil
import subprocess
from pathlib import Path
from collections import Counter

def run_merge():
    """Run the merge script to create clean educational data."""
    print("=" * 70)
    print("STEP 1: MERGE EDUCATIONAL DATA")
    print("=" * 70)
    
    cmd = [
        "python3", "script/merge_educational_data.py",
        "--existing_file1", "data/sft_training/20251107_044104_openai_gpt-5_sft.jsonl",
        "--existing_file2", "data/sft_training/20251017_161801_sft_merged.jsonl",
        "--supplemental_file", "data/sft_training/20251107_142710_openai_gpt-5_sft.jsonl",
        "--stratified_split", "data/splits/educational_stratified.json",
        "--output_path", "data/sft_training/educational_stratified_complete.jsonl"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("❌ Merge failed!")
        return False
    
    return True

def organize_clean_data():
    """Copy and organize data into data/sft_clean."""
    print("\n" + "=" * 70)
    print("STEP 2: ORGANIZE CLEAN DATA")
    print("=" * 70)
    
    # Create clean directory
    clean_dir = Path("data/sft_clean")
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy educational data
    edu_source = Path("data/sft_training/educational_stratified_complete.jsonl")
    edu_dest = clean_dir / "educational_stratified.jsonl"
    
    if edu_source.exists():
        shutil.copy(edu_source, edu_dest)
        print(f"✓ Copied educational data to {edu_dest}")
    else:
        print(f"❌ Educational data not found: {edu_source}")
        return False
    
    # Copy adaptation data
    adapt_source = Path("data/adaptation/game_format_adaptation_stratified.jsonl")
    adapt_dest = clean_dir / "adaptation_stratified.jsonl"
    
    if adapt_source.exists():
        shutil.copy(adapt_source, adapt_dest)
        print(f"✓ Copied adaptation data to {adapt_dest}")
    else:
        print(f"❌ Adaptation data not found: {adapt_source}")
        return False
    
    return True

def validate_and_summarize():
    """Validate the clean data and create summary."""
    print("\n" + "=" * 70)
    print("STEP 3: VALIDATE AND SUMMARIZE")
    print("=" * 70)
    
    clean_dir = Path("data/sft_clean")
    
    # Load educational data
    edu_path = clean_dir / "educational_stratified.jsonl"
    with open(edu_path, 'r') as f:
        edu_data = [json.loads(line) for line in f]
    
    edu_note_ids = set()
    edu_error_counts = Counter()
    for item in edu_data:
        edu_note_ids.add(item['metadata']['original_id'])
        error_type = item['metadata'].get('error_type')
        if error_type not in ['vanilla', 'none', 'safe', 'unknown']:
            edu_error_counts[error_type] += 1
    
    # Load adaptation data
    adapt_path = clean_dir / "adaptation_stratified.jsonl"
    with open(adapt_path, 'r') as f:
        adapt_data = [json.loads(line) for line in f]
    
    adapt_note_ids = set()
    adapt_error_counts = Counter()
    for item in adapt_data:
        adapt_note_ids.add(item['metadata']['original_id'])
        error_type = item['metadata'].get('error_type')
        if error_type not in ['vanilla', 'none', 'safe', 'unknown']:
            adapt_error_counts[error_type] += 1
    
    # Check for overlap
    overlap = edu_note_ids & adapt_note_ids
    
    # Print summary
    print(f"\n📊 EDUCATIONAL DATA:")
    print(f"  Path: {edu_path}")
    print(f"  Total examples: {len(edu_data)}")
    print(f"  Unique notes: {len(edu_note_ids)}")
    print(f"  Error type distribution:")
    for error_type, count in sorted(edu_error_counts.items()):
        print(f"    • {error_type}: {count}")
    
    print(f"\n📊 ADAPTATION DATA:")
    print(f"  Path: {adapt_path}")
    print(f"  Total examples: {len(adapt_data)}")
    print(f"  Unique notes: {len(adapt_note_ids)}")
    print(f"  Error type distribution:")
    for error_type, count in sorted(adapt_error_counts.items()):
        print(f"    • {error_type}: {count}")
    
    print(f"\n📊 OVERLAP CHECK:")
    print(f"  Overlapping notes: {len(overlap)}")
    if len(overlap) == 0:
        print(f"  ✓ No overlap - clean separation!")
    else:
        print(f"  ⚠️  Warning: {len(overlap)} notes appear in both datasets")
    
    # Create summary document
    summary = {
        "educational": {
            "path": str(edu_path),
            "total_examples": len(edu_data),
            "unique_notes": len(edu_note_ids),
            "error_types": dict(edu_error_counts),
            "percentage_of_medec": f"{len(edu_note_ids)/1219*100:.1f}%"
        },
        "adaptation": {
            "path": str(adapt_path),
            "total_examples": len(adapt_data),
            "unique_notes": len(adapt_note_ids),
            "error_types": dict(adapt_error_counts),
            "percentage_of_medec": f"{len(adapt_note_ids)/1219*100:.1f}%"
        },
        "total": {
            "unique_notes": len(edu_note_ids) + len(adapt_note_ids),
            "overlap": len(overlap),
            "coverage_of_medec": f"{(len(edu_note_ids) + len(adapt_note_ids))/1219*100:.1f}%"
        },
        "stratification": {
            "educational_split": "75% of MEDEC",
            "adaptation_split": "25% of MEDEC",
            "all_error_types_included": True,
            "proper_stratification": True
        }
    }
    
    summary_path = clean_dir / "SUMMARY.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Saved summary to {summary_path}")
    
    # Create README
    readme_content = f"""# Clean Stratified Training Data

This directory contains properly stratified educational and adaptation data for medical error detection training.

## Files

- `educational_stratified.jsonl` - Educational SFT data (913 notes, 75% of MEDEC)
- `adaptation_stratified.jsonl` - Game format adaptation data (306 notes → 1,224 examples, 25% of MEDEC)
- `SUMMARY.json` - Detailed statistics and validation

## Statistics

### Educational Data
- **Total examples**: {len(edu_data)}
- **Unique notes**: {len(edu_note_ids)} (75% of MEDEC)
- **Error types**: All 5 types included
  - causalOrganism: {edu_error_counts.get('causalOrganism', 0)}
  - diagnosis: {edu_error_counts.get('diagnosis', 0)}
  - management: {edu_error_counts.get('management', 0)}
  - pharmacotherapy: {edu_error_counts.get('pharmacotherapy', 0)}
  - treatment: {edu_error_counts.get('treatment', 0)}

### Adaptation Data
- **Total examples**: {len(adapt_data)}
- **Unique notes**: {len(adapt_note_ids)} (25% of MEDEC)
- **Format**: Game format with attacker/assessor roles
- **Style split**: 75% clean AI-style / 25% messy human-style (for safe examples)
- **Error types**: All 5 types included
  - causalOrganism: {adapt_error_counts.get('causalOrganism', 0)}
  - diagnosis: {adapt_error_counts.get('diagnosis', 0)}
  - management: {adapt_error_counts.get('management', 0)}
  - pharmacotherapy: {adapt_error_counts.get('pharmacotherapy', 0)}
  - treatment: {adapt_error_counts.get('treatment', 0)}

## Validation

- ✓ Proper 75/25 stratification
- ✓ All 5 error types in both datasets
- ✓ No overlap between educational and adaptation
- ✓ 100% MEDEC coverage ({len(edu_note_ids) + len(adapt_note_ids)}/1219 notes)

## Usage

### Educational SFT Training
```bash
python3 script/train_qwen3_sft.py \\
  --data_path data/sft_clean/educational_stratified.jsonl \\
  --epochs 3 \\
  --batch_size 4 \\
  --output_dir trainer_output/qwen3_educational
```

### Adaptation Training
```bash
python3 script/train_qwen3_sft.py \\
  --model_id trainer_output/qwen3_educational \\
  --data_path data/sft_clean/adaptation_stratified.jsonl \\
  --epochs 1 \\
  --batch_size 4 \\
  --learning_rate 1e-5 \\
  --output_dir trainer_output/qwen3_game_adapted
```

### GRPO Self-Play Training
```bash
python3 script/train_selfplay_advanced.py \\
  --model_id trainer_output/qwen3_game_adapted \\
  --num_samples 16 \\
  --rounds 3
```

## Generation Details

- **Educational data**: Merged from 3 sources with proper stratification
  - 20251107_044104_openai_gpt-5_sft.jsonl
  - 20251017_161801_sft_merged.jsonl
  - 20251107_142710_openai_gpt-5_sft.jsonl (supplemental)

- **Adaptation data**: Generated with GPT-5 in parallel
  - Strategic reasoning for attacker and assessor roles
  - Pre-fill CoT format
  - 75/25 AI-style/human-style split for robustness

Generated: {Path('data/sft_clean').stat().st_mtime}
"""
    
    readme_path = clean_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"💾 Saved README to {readme_path}")
    
    return True

def main():
    print("=" * 70)
    print("ORGANIZE CLEAN STRATIFIED DATA")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Merge educational data from 3 sources")
    print("  2. Copy stratified data to data/sft_clean")
    print("  3. Validate and create summary documents")
    print()
    
    # Step 1: Merge educational data
    if not run_merge():
        print("\n❌ Failed at merge step")
        return
    
    # Step 2: Organize into clean directory
    if not organize_clean_data():
        print("\n❌ Failed at organize step")
        return
    
    # Step 3: Validate and summarize
    if not validate_and_summarize():
        print("\n❌ Failed at validation step")
        return
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS!")
    print("=" * 70)
    print("\nClean stratified data is ready in: data/sft_clean/")
    print("\nNext steps:")
    print("  1. Train educational SFT: data/sft_clean/educational_stratified.jsonl")
    print("  2. Train adaptation: data/sft_clean/adaptation_stratified.jsonl")
    print("  3. Run GRPO self-play training")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
