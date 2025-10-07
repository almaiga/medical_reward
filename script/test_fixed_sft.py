#!/usr/bin/env python3
"""
Test the fixed SFT generation (without max_output_tokens).
"""

import subprocess

def main():
    print("🧪 Testing FIXED SFT Generation with GPT-5 (2 examples)")
    print("=" * 60)
    print("Key fix: Removed max_output_tokens parameter")
    print()
    
    cmd = [
        "python", "script/generate_sft_data.py",
        "--api_provider", "openai",
        "--model", "gpt-5",
        "--max_examples", "2",
        "--output_dir", "data/sft_training"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print("✅ Generation completed!")
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

if __name__ == "__main__":
    main()