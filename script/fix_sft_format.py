#!/usr/bin/env python3
"""
Fix format issues in SFT training data.
Adds missing <think> tags and ensures proper format compliance.
"""

import json
import argparse
import re
from pathlib import Path

def fix_missing_think_tags(content: str, role: str) -> str:
    """Add missing <think> tags to responses that only have <output>."""
    
    # If already has both tags, return as-is
    if '<think>' in content and '<output>' in content:
        return content
    
    # If only has <output>, add a minimal <think> tag
    if '<output>' in content and '<think>' not in content:
        # Extract the output content
        output_match = re.search(r'<output>(.*?)(?:</output>|$)', content, re.DOTALL)
        if output_match:
            output_content = output_match.group(1).strip()
            
            # Create appropriate thinking based on role
            if role == 'attacker':
                if 'vanilla' in content.lower() or len(output_content) > 500:
                    think_content = "I'll make minor wording changes while preserving medical accuracy."
                else:
                    think_content = "I'll introduce a subtle medical error for educational purposes."
            else:  # assessor
                # Try to infer classification from output
                if any(cls in output_content for cls in ['Safe', 'Concerning', 'Harmful']):
                    if 'Safe' in output_content:
                        think_content = "This note appears medically appropriate with no safety concerns."
                    elif 'Concerning' in output_content:
                        think_content = "This note has some issues that could affect patient care."
                    else:  # Harmful
                        think_content = "This note contains errors that could lead to patient harm."
                else:
                    think_content = "Let me analyze this medical note for safety issues."
            
            # Reconstruct with both tags
            fixed_content = f"<think>\n{think_content}\n</think>\n<output>\n{output_content}\n</output>"
            return fixed_content
    
    # If has neither tag, wrap entire content in output with minimal think
    if '<think>' not in content and '<output>' not in content:
        if role == 'attacker':
            think_content = "I'll modify this medical note as requested."
        else:
            think_content = "Let me assess this medical note."
        
        fixed_content = f"<think>\n{think_content}\n</think>\n<output>\n{content.strip()}\n</output>"
        return fixed_content
    
    return content

def fix_sft_data(input_path: str, output_path: str, fix_format: bool = True):
    """Fix format issues in SFT data."""
    print(f"Loading data from {input_path}")
    
    data = []
    fixed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                
                if fix_format and 'messages' in item and 'metadata' in item:
                    role = item['metadata']['role']
                    
                    # Fix assistant messages
                    for message in item['messages']:
                        if message['role'] == 'assistant':
                            original_content = message['content']
                            
                            # Check if needs fixing
                            has_think = '<think>' in original_content
                            has_output = '<output>' in original_content
                            
                            if not has_think or not has_output:
                                fixed_content = fix_missing_think_tags(original_content, role)
                                message['content'] = fixed_content
                                fixed_count += 1
                                
                                print(f"Fixed line {line_num} ({role}): Added missing tags")
                
                data.append(item)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
    
    print(f"Loaded {len(data)} examples, fixed {fixed_count} format issues")
    
    # Save fixed data
    print(f"Saving fixed data to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(data), fixed_count

def validate_fixed_data(data_path: str):
    """Validate that the fixed data has proper format compliance."""
    print(f"\nValidating fixed data: {data_path}")
    
    format_stats = {
        'has_think': 0,
        'has_output': 0,
        'proper_format': 0,
        'total': 0
    }
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            
            if 'messages' in item:
                for message in item['messages']:
                    if message['role'] == 'assistant':
                        content = message['content']
                        format_stats['total'] += 1
                        
                        has_think = '<think>' in content and '</think>' in content
                        has_output = '<output>' in content
                        
                        if has_think:
                            format_stats['has_think'] += 1
                        if has_output:
                            format_stats['has_output'] += 1
                        if has_think and has_output:
                            format_stats['proper_format'] += 1
    
    total = format_stats['total']
    if total > 0:
        print(f"Format compliance after fixing:")
        print(f"  Has <think> tags: {format_stats['has_think']}/{total} ({format_stats['has_think']/total*100:.1f}%)")
        print(f"  Has <output> tags: {format_stats['has_output']}/{total} ({format_stats['has_output']/total*100:.1f}%)")
        print(f"  Proper format: {format_stats['proper_format']}/{total} ({format_stats['proper_format']/total*100:.1f}%)")
        
        if format_stats['proper_format'] == total:
            print("✅ All examples now have proper format!")
        else:
            print(f"⚠️  Still have {total - format_stats['proper_format']} examples with format issues")

def main():
    parser = argparse.ArgumentParser(description="Fix format issues in SFT training data")
    parser.add_argument("input_path", help="Path to input SFT data file")
    parser.add_argument("--output_path", help="Path to output fixed data file (default: input_path with _fixed suffix)")
    parser.add_argument("--no_fix", action="store_true", help="Only validate, don't fix")
    parser.add_argument("--validate_only", action="store_true", help="Only run validation on existing file")
    
    args = parser.parse_args()
    
    if args.validate_only:
        validate_fixed_data(args.input_path)
        return
    
    # Set output path
    if not args.output_path:
        input_path = Path(args.input_path)
        args.output_path = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")
    
    print("=== SFT Data Format Fixer ===")
    
    if args.no_fix:
        print("Running validation only (no fixes will be applied)")
        validate_fixed_data(args.input_path)
    else:
        # Fix the data
        total_examples, fixed_count = fix_sft_data(args.input_path, args.output_path, fix_format=True)
        
        # Validate the fixed data
        validate_fixed_data(args.output_path)
        
        print(f"\n=== Summary ===")
        print(f"Input file: {args.input_path}")
        print(f"Output file: {args.output_path}")
        print(f"Total examples: {total_examples}")
        print(f"Fixed examples: {fixed_count}")
        
        if fixed_count > 0:
            print(f"✅ Fixed {fixed_count} format issues!")
            print(f"Use the fixed file for training: {args.output_path}")
        else:
            print(f"✅ No format issues found!")

if __name__ == "__main__":
    main()