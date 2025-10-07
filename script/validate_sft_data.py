#!/usr/bin/env python3
"""
Validate SFT training data for format compliance and quality.
"""

import json
import argparse
import re
from collections import Counter, defaultdict

def load_and_validate_data(data_path: str):
    """Load and validate SFT data."""
    print(f"Loading data from {data_path}")
    
    data = []
    errors = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON decode error - {e}")
    
    print(f"Loaded {len(data)} examples")
    if errors:
        print(f"Found {len(errors)} JSON errors:")
        for error in errors[:5]:  # Show first 5
            print(f"  {error}")
    
    return data

def analyze_data_structure(data):
    """Analyze the structure and content of the data."""
    print("\n=== Data Structure Analysis ===")
    
    # Check required fields
    required_fields = ['messages', 'metadata']
    missing_fields = defaultdict(int)
    
    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                missing_fields[field] += 1
    
    if missing_fields:
        print("Missing required fields:")
        for field, count in missing_fields.items():
            print(f"  {field}: {count} examples")
    else:
        print("✅ All examples have required fields")
    
    # Analyze metadata
    roles = Counter()
    error_types = Counter()
    
    for item in data:
        if 'metadata' in item:
            meta = item['metadata']
            roles[meta.get('role', 'unknown')] += 1
            error_types[meta.get('error_type', 'unknown')] += 1
    
    print(f"\nRole distribution:")
    for role, count in roles.items():
        print(f"  {role}: {count}")
    
    print(f"\nError type distribution:")
    for error_type, count in error_types.most_common():
        print(f"  {error_type}: {count}")

def check_format_compliance(data):
    """Check if responses follow the required <think><output> format."""
    print("\n=== Format Compliance Check ===")
    
    format_stats = {
        'has_think': 0,
        'has_output': 0,
        'proper_format': 0,
        'total': 0
    }
    
    format_issues = []
    
    for i, item in enumerate(data):
        if 'messages' not in item:
            continue
            
        messages = item['messages']
        if len(messages) < 2:
            continue
            
        # Get assistant response
        assistant_msg = None
        for msg in messages:
            if msg.get('role') == 'assistant':
                assistant_msg = msg.get('content', '')
                break
        
        if not assistant_msg:
            continue
            
        format_stats['total'] += 1
        
        # Check for required tags
        has_think = '<think>' in assistant_msg and '</think>' in assistant_msg
        has_output = '<output>' in assistant_msg
        
        if has_think:
            format_stats['has_think'] += 1
        if has_output:
            format_stats['has_output'] += 1
        if has_think and has_output:
            format_stats['proper_format'] += 1
        else:
            # Record format issues
            role = item.get('metadata', {}).get('role', 'unknown')
            format_issues.append({
                'index': i,
                'role': role,
                'has_think': has_think,
                'has_output': has_output,
                'content_preview': assistant_msg[:100] + '...' if len(assistant_msg) > 100 else assistant_msg
            })
    
    # Print statistics
    total = format_stats['total']
    if total > 0:
        print(f"Format compliance (out of {total} examples):")
        print(f"  Has <think> tags: {format_stats['has_think']} ({format_stats['has_think']/total*100:.1f}%)")
        print(f"  Has <output> tags: {format_stats['has_output']} ({format_stats['has_output']/total*100:.1f}%)")
        print(f"  Proper format: {format_stats['proper_format']} ({format_stats['proper_format']/total*100:.1f}%)")
    
    # Show format issues
    if format_issues:
        print(f"\nFormat issues found in {len(format_issues)} examples:")
        for issue in format_issues[:5]:  # Show first 5
            print(f"  Example {issue['index']} ({issue['role']}): Think={issue['has_think']}, Output={issue['has_output']}")
            print(f"    Preview: {issue['content_preview']}")
    else:
        print("✅ All examples have proper format!")

def analyze_content_quality(data):
    """Analyze content quality and patterns."""
    print("\n=== Content Quality Analysis ===")
    
    # Length statistics
    user_lengths = []
    assistant_lengths = []
    
    for item in data:
        if 'messages' not in item:
            continue
            
        for msg in item['messages']:
            content = msg.get('content', '')
            if msg.get('role') == 'user':
                user_lengths.append(len(content))
            elif msg.get('role') == 'assistant':
                assistant_lengths.append(len(content))
    
    if user_lengths:
        print(f"User message lengths:")
        print(f"  Average: {sum(user_lengths)/len(user_lengths):.0f} chars")
        print(f"  Min: {min(user_lengths)}, Max: {max(user_lengths)}")
    
    if assistant_lengths:
        print(f"Assistant message lengths:")
        print(f"  Average: {sum(assistant_lengths)/len(assistant_lengths):.0f} chars")
        print(f"  Min: {min(assistant_lengths)}, Max: {max(assistant_lengths)}")
    
    # Check for common patterns
    classification_patterns = ['Safe', 'Concerning', 'Harmful']
    classification_counts = Counter()
    
    for item in data:
        if item.get('metadata', {}).get('role') == 'assessor':
            messages = item.get('messages', [])
            for msg in messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    for pattern in classification_patterns:
                        if pattern in content:
                            classification_counts[pattern] += 1
    
    if classification_counts:
        print(f"\nAssessor classification distribution:")
        for classification, count in classification_counts.items():
            print(f"  {classification}: {count}")

def check_data_balance(data):
    """Check if data is balanced across roles and error types."""
    print("\n=== Data Balance Analysis ===")
    
    # Role balance
    role_counts = Counter()
    role_error_types = defaultdict(Counter)
    
    for item in data:
        meta = item.get('metadata', {})
        role = meta.get('role', 'unknown')
        error_type = meta.get('error_type', 'unknown')
        
        role_counts[role] += 1
        role_error_types[role][error_type] += 1
    
    print("Role balance:")
    total = sum(role_counts.values())
    for role, count in role_counts.items():
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {role}: {count} ({percentage:.1f}%)")
    
    print("\nError type distribution by role:")
    for role, error_counts in role_error_types.items():
        print(f"  {role}:")
        for error_type, count in error_counts.most_common():
            print(f"    {error_type}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Validate SFT training data")
    parser.add_argument("data_path", help="Path to SFT data file (JSONL)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    # Load and validate data
    data = load_and_validate_data(args.data_path)
    
    if not data:
        print("No valid data found!")
        return
    
    # Run analyses
    analyze_data_structure(data)
    check_format_compliance(data)
    
    if args.detailed:
        analyze_content_quality(data)
        check_data_balance(data)
    
    print(f"\n=== Summary ===")
    print(f"Total examples: {len(data)}")
    print(f"Data appears ready for SFT training!")

if __name__ == "__main__":
    main()