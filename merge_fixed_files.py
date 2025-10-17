#!/usr/bin/env python3
"""
Merge all *_fixed.jsonl files in data/sft_training directory.
Deduplicates based on 'original_id' field.
Each original_id has 4 lines (entries) associated with it.
"""
import json
from pathlib import Path
from datetime import datetime

def merge_fixed_files():
    # Find all _fixed files
    sft_dir = Path("data/sft_training")
    fixed_files = sorted(sft_dir.glob("*_fixed.jsonl"))
    
    print(f"Found {len(fixed_files)} files with _fixed suffix:")
    for f in fixed_files:
        print(f"  - {f.name}")
    
    # Track unique entries by original_id (each ID has 4 lines)
    seen_ids = set()
    merged_data = []
    duplicate_groups = 0
    lines_per_id = 4
    
    # Process each file
    for file_path in fixed_files:
        print(f"\nProcessing: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process in groups of 4 lines
        for i in range(0, len(lines), lines_per_id):
            group = lines[i:i+lines_per_id]
            
            if len(group) != lines_per_id:
                print(f"  Warning: Incomplete group at line {i+1} (only {len(group)} lines)")
                continue
            
            try:
                # Parse all 4 entries in the group
                entries = []
                original_id = None
                
                for j, line in enumerate(group):
                    entry = json.loads(line.strip())
                    entries.append(entry)
                    
                    # Get original_id from metadata or root level
                    entry_id = entry.get('original_id')
                    if entry_id is None and 'metadata' in entry:
                        entry_id = entry['metadata'].get('original_id')
                    
                    if entry_id:
                        original_id = entry_id
                
                if original_id is None:
                    print(f"  Warning: Group at line {i+1} has no 'original_id', skipping")
                    continue
                
                # Check if we've seen this ID before
                if original_id in seen_ids:
                    duplicate_groups += 1
                    print(f"  Duplicate group: original_id={original_id} (lines {i+1}-{i+lines_per_id})")
                else:
                    seen_ids.add(original_id)
                    merged_data.extend(entries)
                    
            except json.JSONDecodeError as e:
                print(f"  Error parsing group at line {i+1}: {e}")
    
    # Create output filename with current date/time
    now = datetime.now()
    output_filename = now.strftime("%Y%m%d_%H%M%S") + "_sft_merged.jsonl"
    output_path = sft_dir / output_filename
    
    # Write merged data
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in merged_data:
            f.write(json.dumps(entry) + '\n')
    
    unique_ids = len(seen_ids)
    total_lines = len(merged_data)
    
    print(f"\n{'='*60}")
    print(f"Merge complete!")
    print(f"Unique original_ids: {unique_ids}")
    print(f"Total lines written: {total_lines} ({unique_ids} × 4)")
    print(f"Duplicate groups skipped: {duplicate_groups}")
    print(f"Output file: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    merge_fixed_files()
