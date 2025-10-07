#!/usr/bin/env python3
"""
Test script to run SFT generation with just 1 example.
"""

import subprocess
import sys

def main():
    print("🧪 Testing SFT generation with 1 example...")
    
    cmd = [
        "python3", "script/generate_sft_data.py",
        "--api_provider", "openai",
        "--model", "gpt-5", 
        "--max_examples", "1",
        "--output_dir", "data/sft_training"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run with timeout to prevent hanging
        result = subprocess.run(
            cmd, 
            timeout=60,  # 60 second timeout
            capture_output=True, 
            text=True
        )
        
        print("✅ Command completed!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 60 seconds!")
        print("This confirms the hanging issue.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()