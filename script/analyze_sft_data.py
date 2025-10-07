#!/usr/bin/env python3
"""
Analyze and preview generated SFT training data.
"""

import json
import argparse
import os
from collections import Counter
from pathlib import Path

def load_jsonl(file_path: str):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def analyze_raw_data(data):
    """Analyze raw training data."""
    print("=== Raw Data Analysis ===")
    print(f"Total examples: {len(data)}")
    
    # Role distribution
    roles = Counter(item['role'] for item in data)
    print(f"Role distribution: {dict(roles)}")
    
    # Error type distribution
    error_types = Counter(item['error_type'] for item in data)
    print(f"Error types: {dict(error_types)}")
    
    # Show sample examples
    print("\n=== Sample Examples ===")
    
    # Show one attacker example
    attacker_examples = [item for item in data if item['role'] == 'attacker']
    if attacker_examples:
        print("\n--- Attacker Example ---")
        ex = attacker_examples[0]
        print(f"ID: {ex['id']}")
        print(f"Error Type: {ex['error_type']}")
        print(f"Original Text: {ex['original_text'][:1000]}...")
        print(f"Target Output: {ex['target_output'][:1000]}...")
        print(f"Generated Response: {ex['response'][:1000]}...")
    
    # Show one assessor example
    assessor_examples = [item for item in data if item['role'] == 'assessor']
    if assessor_examples:
        print("\n--- Assessor Example ---")
        ex = assessor_examples[0]
        print(f"ID: {ex['id']}")
        print(f"Error Type: {ex['error_type']}")
        print(f"Attacked Text: {ex['attacked_text'][:1000]}...")
        print(f"Target Classification: {ex['target_classification']}")
        print(f"Generated Response: {ex['response'][:1000]}...")

def analyze_sft_data(data):
    """Analyze SFT format data."""
    print("\n=== SFT Format Analysis ===")
    print(f"Total training pairs: {len(data)}")
    
    # Role distribution from metadata
    roles = Counter(item['metadata']['role'] for item in data)
    print(f"Role distribution: {dict(roles)}")
    
    # Average lengths
    prompt_lengths = [len(item['messages'][0]['content']) for item in data]
    response_lengths = [len(item['messages'][1]['content']) for item in data]
    
    print(f"Average prompt length: {sum(prompt_lengths) / len(prompt_lengths):.0f} chars")
    print(f"Average response length: {sum(response_lengths) / len(response_lengths):.0f} chars")
    
    # Show sample SFT pair
    print("\n=== Sample SFT Pair ===")
    if data:
        ex = data[1]
        print(f"Role: {ex['metadata']['role']}")
        print(f"Error Type: {ex['metadata']['error_type']}")
        print(f"Prompt: {ex['messages'][0]['content'][:1000]}...")
        print(f"Response: {ex['messages'][1]['content'][:1000]}...")

def validate_format_compliance(data):
    """Check if responses follow the required <think>/<output> format."""
    print("\n=== Format Compliance Check ===")
    
    compliant_count = 0
    total_count = len(data)
    
    for item in data:
        if 'response' in item:
            response = item['response']
        else:
            response = item['messages'][1]['content']
        
        has_think = '<think>' in response and '</think>' in response
        has_output = '<output>' in response and '</output>' in response
        
        if has_think and has_output:
            compliant_count += 1
    
    compliance_rate = (compliant_count / total_count) * 100 if total_count > 0 else 0
    print(f"Format compliance: {compliant_count}/{total_count} ({compliance_rate:.1f}%)")
    
    if compliance_rate < 90:
        print("⚠️  Low compliance rate! Consider adjusting prompts or API parameters.")
    else:
        print("✅ Good format compliance!")

def main():
    parser = argparse.ArgumentParser(description="Analyze SFT training data")
    parser.add_argument("file_path", help="Path to JSONL file to analyze")
    parser.add_argument("--format", choices=['raw', 'sft'], default='auto',
                       help="Data format (auto-detect by default)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"❌ File not found: {args.file_path}")
        return
    
    print(f"📊 Analyzing: {args.file_path}")
    
    # Load data
    data = load_jsonl(args.file_path)
    
    # Auto-detect format
    if args.format == 'auto':
        if data and 'messages' in data[0]:
            format_type = 'sft'
        else:
            format_type = 'raw'
    else:
        format_type = args.format
    
    print(f"Detected format: {format_type}")
    
    # Analyze based on format
    if format_type == 'raw':
        analyze_raw_data(data)
        validate_format_compliance(data)
    else:
        analyze_sft_data(data)
        validate_format_compliance(data)

if __name__ == "__main__":
    main()