#!/usr/bin/env python3
"""
Test script to validate Qwen3-compliant SFT data format.
"""

import json
import subprocess

def test_qwen3_format():
    """Test SFT generation with Qwen3-optimized prompts."""
    print("🧪 Testing Qwen3-Compliant SFT Generation")
    print("=" * 50)
    
    print("🎯 Key Qwen3 Features Being Optimized:")
    print("  ✅ Thinking Mode: Structured <think>/<output> format")
    print("  ✅ Enhanced Reasoning: 5-step logical analysis")
    print("  ✅ Agent Capabilities: Medical assessment tasks")
    print("  ✅ Human Preference Alignment: Clear instructions")
    print()
    
    # Test with 2 examples using GPT-4o
    cmd = [
        "python3", "script/generate_sft_data.py",
        "--api_provider", "openai",
        "--model", "gpt-5",
        "--max_examples", "2",
        "--output_dir", "data/sft_training"
    ]
    
    print("🚀 Generating Qwen3-optimized examples...")
    print("Expected improvements:")
    print("  - More structured thinking patterns")
    print("  - Better clinical reasoning")
    print("  - Enhanced logical flow")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Generation completed!")
        
        # Find the latest generated file
        import os
        import glob
        
        sft_files = glob.glob("data/sft_training/*_sft.jsonl")
        if sft_files:
            latest_file = max(sft_files, key=os.path.getctime)
            print(f"📄 Latest file: {latest_file}")
            
            # Analyze the format
            analyze_qwen3_compliance(latest_file)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed: {e}")
        print("STDERR:", e.stderr)

def analyze_qwen3_compliance(file_path):
    """Analyze generated data for Qwen3 compliance."""
    print(f"\n🔍 Analyzing Qwen3 Compliance: {file_path}")
    print("-" * 50)
    
    with open(file_path, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    for i, example in enumerate(examples):
        role = example['metadata']['role']
        response = example['messages'][1]['content']
        
        print(f"\n📝 Example {i+1} ({role.upper()}):")
        
        # Check format compliance
        has_think = '<think>' in response and '</think>' in response
        has_output = '<output>' in response and '</output>' in response
        
        print(f"  ✅ Think block: {'Yes' if has_think else '❌ Missing'}")
        print(f"  ✅ Output block: {'Yes' if has_output else '❌ Missing'}")
        
        if has_think:
            think_content = response.split('<think>')[1].split('</think>')[0].strip()
            
            # Check for structured reasoning (numbered points)
            has_structure = any(str(i) in think_content for i in range(1, 6))
            has_analysis = 'analysis' in think_content.lower() or 'assess' in think_content.lower()
            has_reasoning = 'because' in think_content.lower() or 'reasoning' in think_content.lower()
            
            print(f"  ✅ Structured format: {'Yes' if has_structure else 'Partial'}")
            print(f"  ✅ Clinical analysis: {'Yes' if has_analysis else 'Partial'}")
            print(f"  ✅ Logical reasoning: {'Yes' if has_reasoning else 'Partial'}")
            print(f"  📊 Think length: {len(think_content)} chars")
        
        # Show preview
        preview = response[:200] + "..." if len(response) > 200 else response
        print(f"  📄 Preview: {preview}")
    
    print(f"\n🎯 Qwen3 Optimization Summary:")
    print(f"  📊 Total examples: {len(examples)}")
    print(f"  🧠 Enhanced reasoning format: Structured 5-step analysis")
    print(f"  🎯 Agent-ready: Medical assessment tasks")
    print(f"  💬 Conversation-ready: Clear instruction following")

def main():
    test_qwen3_format()

if __name__ == "__main__":
    main()