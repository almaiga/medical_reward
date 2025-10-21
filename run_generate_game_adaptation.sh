#!/bin/bash
# Generate Game Format Adaptation Data with GPT-5
# 
# This script creates training examples that teach the model:
# 1. Game format (attacker vs assessor prompts)
# 2. Strategic reasoning (how to play the game effectively)
# 3. Pre-fill CoT format (matches educational SFT + GRPO)
#
# Uses GPT-4o-mini for high-quality strategic reasoning
# Cost: ~$0.05 for 125 rows × 2 GPT calls per row

set -e

echo "========================================="
echo "Game Format Adaptation Data Generation"
echo "========================================="
echo ""
echo "Mode: GPT-augmented (high quality)"
echo "Model: gpt-4o-mini"
echo "Cost: ~\$0.05 for 250 API calls"
echo ""

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
    exit 1
fi

echo "✅ OpenAI API key found"
echo ""

# Run generation script with GPT
python3 script/generate_game_format_adaptation.py \
    --medec_path data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv \
    --start_id 733 \
    --end_id 857 \
    --use_gpt \
    --gpt_model gpt-4o-mini \
    --output_path data/adaptation/game_format_adaptation.jsonl

echo ""
echo "========================================="
echo "✅ Data generation complete!"
echo "========================================="
echo ""
echo "Generated: data/adaptation/game_format_adaptation.jsonl"
echo ""
echo "Next step: Run adaptation training"
echo ""
echo "  bash run_sft_game_adaptation.sh"
echo ""
