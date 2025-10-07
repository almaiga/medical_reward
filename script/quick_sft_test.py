#!/usr/bin/env python3
"""
Quick SFT test with time estimation.
"""

import subprocess
import sys

def main():
    print("⚡ Quick SFT Generation Test")
    print("=" * 40)
    
    # Test with 3 examples
    examples = 3
    estimated_time = examples * 25  # ~25 seconds per example
    
    print(f"📊 Testing with {examples} examples")
    print(f"⏱️  Estimated time: ~{estimated_time} seconds ({estimated_time/60:.1f} minutes)")
    print(f"🔄 Each example = 2 API calls + processing")
    print()
    
    cmd = [
        "python3", "script/generate_sft_data.py",
        "--api_provider", "openai",
        "--model", "gpt-4o-mini", 
        "--max_examples", str(examples),
        "--output_dir", "data/sft_training"
    ]
    
    print("🚀 Starting generation...")
    print("(You should see a progress bar)")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Generation completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Generation failed: {e}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Generation interrupted by user")

if __name__ == "__main__":
    main()