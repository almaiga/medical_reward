#!/usr/bin/env python3
"""
Example script showing how to generate SFT training data.
Run this after setting up your API keys.
"""

import os
import subprocess


def main():
    print("=== SFT Data Generation Example ===")

    # Check if API keys are set
    openai_key = os.getenv("OPENAI_API_KEY")
    claude_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key and not claude_key:
        print("❌ No API keys found!")
        print("Please set either:")
        print("  export OPENAI_API_KEY='your-openai-key'")
        print("  export ANTHROPIC_API_KEY='your-claude-key'")
        return

    # Determine which API to use
    if openai_key:
        api_provider = "openai"
        model = "gpt-5"
        print(f"✅ Using OpenAI API with {model}")
    else:
        api_provider = "claude"
        model = "claude-3-haiku-20240307"
        print(f"✅ Using Claude API with {model}")

    # Generate a small sample first (10 examples)
    print("\n🚀 Generating sample SFT data (10 examples)...")

    cmd = [
        "python3",
        "script/generate_sft_data.py",
        "--api_provider",
        api_provider,
        "--model",
        model,
        "--max_examples",
        "10",
        "--output_dir",
        "data/sft_training",
    ]

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
        print("✅ Sample generation completed!")
        print(result.stdout)

        print("\n📁 Generated files:")
        print("  - Raw data: data/sft_training/*_raw.jsonl")
        print("  - SFT format: data/sft_training/*_sft.jsonl")

        print("\n🎯 Next steps:")
        print("1. Review the generated examples")
        print("2. Run with more examples: --max_examples 100")
        print("3. Use the SFT format file for fine-tuning")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running generation script: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)


if __name__ == "__main__":
    main()
