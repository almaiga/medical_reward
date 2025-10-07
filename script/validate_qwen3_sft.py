#!/usr/bin/env python3
"""
Validate SFT data quality for Qwen3-8B fine-tuning.
"""

import json
import argparse
from collections import Counter

def analyze_sft_quality(file_path):
    """Analyze SFT data quality for Qwen3-8B training."""
    print(f"🔍 Analyzing SFT Data Quality for Qwen3-8B: {file_path}")
    print("=" * 60)
    
    with open(file_path, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    # Basic stats
    total_examples = len(examples)
    roles = Counter(ex['metadata']['role'] for ex in examples)
    error_types = Counter(ex['metadata']['error_type'] for ex in examples)
    
    print(f"📊 Dataset Overview:")
    print(f"  Total examples: {total_examples}")
    print(f"  Roles: {dict(roles)}")
    print(f"  Error types: {dict(error_types)}")
    print()
    
    # Format compliance
    format_issues = []
    reasoning_quality = []
    
    for i, example in enumerate(examples):
        response = example['messages'][1]['content']
        role = example['metadata']['role']
        
        # Check format
        has_think = '<think>' in response and '</think>' in response
        has_output = '<output>' in response and '</output>' in response
        
        if not (has_think and has_output):
            format_issues.append(f"Example {i+1} ({role}): Missing format")
        
        if has_think:
            think_content = response.split('<think>')[1].split('</think>')[0].strip()
            
            # Quality checks for Qwen3-8B training
            quality_score = 0
            
            # Check for educational reasoning
            if 'analyze' in think_content.lower() or 'review' in think_content.lower():
                quality_score += 1
            
            # Check for clinical reasoning
            if 'clinical' in think_content.lower() or 'medical' in think_content.lower():
                quality_score += 1
                
            # Check for specific reasoning
            if 'because' in think_content.lower() or 'due to' in think_content.lower():
                quality_score += 1
                
            # Check for risk assessment
            if 'risk' in think_content.lower() or 'danger' in think_content.lower():
                quality_score += 1
            
            reasoning_quality.append(quality_score)
    
    # Report format compliance
    compliance_rate = (total_examples - len(format_issues)) / total_examples * 100
    print(f"✅ Format Compliance: {compliance_rate:.1f}%")
    if format_issues:
        print(f"⚠️  Format Issues: {len(format_issues)}")
        for issue in format_issues[:3]:  # Show first 3
            print(f"    - {issue}")
    print()
    
    # Report reasoning quality
    avg_quality = sum(reasoning_quality) / len(reasoning_quality) if reasoning_quality else 0
    print(f"🧠 Reasoning Quality Score: {avg_quality:.1f}/4.0")
    print(f"   (Based on: analysis, clinical content, causal reasoning, risk assessment)")
    print()
    
    # Qwen3-8B specific checks
    print(f"🎯 Qwen3-8B Training Readiness:")
    
    # Check for balanced roles
    role_balance = min(roles.values()) / max(roles.values()) if roles else 0
    print(f"  ✅ Role Balance: {role_balance:.2f} (target: >0.8)")
    
    # Check average lengths
    think_lengths = []
    output_lengths = []
    
    for example in examples:
        response = example['messages'][1]['content']
        if '<think>' in response:
            think_content = response.split('<think>')[1].split('</think>')[0].strip()
            think_lengths.append(len(think_content.split()))
        if '<output>' in response:
            output_content = response.split('<output>')[1].split('</output>')[0].strip()
            output_lengths.append(len(output_content.split()))
    
    avg_think_len = sum(think_lengths) / len(think_lengths) if think_lengths else 0
    avg_output_len = sum(output_lengths) / len(output_lengths) if output_lengths else 0
    
    print(f"  📝 Avg Think Length: {avg_think_len:.0f} words (target: 50-150)")
    print(f"  📝 Avg Output Length: {avg_output_len:.0f} words")
    print()
    
    # Show sample for manual review
    print(f"📄 Sample Training Example:")
    if examples:
        sample = examples[0]
        role = sample['metadata']['role']
        response = sample['messages'][1]['content']
        
        print(f"Role: {role.upper()}")
        print(f"Response Preview:")
        print(response[:400] + "..." if len(response) > 400 else response)
    
    print()
    print(f"🎉 Summary for Qwen3-8B Fine-tuning:")
    if compliance_rate > 95 and avg_quality > 2.5 and role_balance > 0.8:
        print(f"✅ HIGH QUALITY - Ready for Qwen3-8B fine-tuning!")
    elif compliance_rate > 90 and avg_quality > 2.0:
        print(f"⚠️  GOOD QUALITY - Minor improvements recommended")
    else:
        print(f"❌ NEEDS IMPROVEMENT - Address format/quality issues")

def main():
    parser = argparse.ArgumentParser(description="Validate SFT data for Qwen3-8B")
    parser.add_argument("file_path", help="Path to SFT JSONL file")
    
    args = parser.parse_args()
    analyze_sft_quality(args.file_path)

if __name__ == "__main__":
    main()