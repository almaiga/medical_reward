#!/usr/bin/env python3
"""
Merge educational and adaptation data into a single file.

Note: It's generally better to train in two stages (educational → adaptation)
rather than merging, but this script is provided for convenience.
"""

import json
from pathlib import Path

def main():
    print("=" * 70)
    print("MERGE ALL SFT DATA")
    print("=" * 70)
    
    # Input files
    educational_path = Path("data/sft_clean/educational_stratified.jsonl")
    adaptation_path = Path("data/sft_clean/adaptation_stratified.jsonl")
    output_path = Path("data/sft_clean/merged_all.jsonl")
    
    # Check files exist
    if not educational_path.exists():
        print(f"❌ Educational data not found: {educational_path}")
        return
    
    if not adaptation_path.exists():
        print(f"❌ Adaptation data not found: {adaptation_path}")
        return
    
    # Load data
    print(f"\n📂 Loading educational data...")
    with open(educational_path, 'r') as f:
        edu_data = [json.loads(line) for line in f]
    print(f"   Loaded {len(edu_data)} examples")
    
    print(f"\n📂 Loading adaptation data...")
    with open(adaptation_path, 'r') as f:
        adapt_data = [json.loads(line) for line in f]
    print(f"   Loaded {len(adapt_data)} examples")
    
    # Merge
    merged_data = edu_data + adapt_data
    
    print(f"\n💾 Saving merged data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Saved {len(merged_data)} examples")
    
    print(f"\n✅ SUCCESS!")
    print(f"\nMerged file: {output_path}")
    print(f"Total examples: {len(merged_data)}")
    print(f"  - Educational: {len(edu_data)}")
    print(f"  - Adaptation: {len(adapt_data)}")
    
    print(f"\n⚠️  NOTE: It's recommended to train in two stages:")
    print(f"  1. Educational SFT (3 epochs)")
    print(f"  2. Adaptation SFT (1 epoch)")
    print(f"\nRather than training on merged data.")
    print("=" * 70)

if __name__ == "__main__":
    main()
