#!/bin/bash
# Train Game Format Adaptation (1 epoch)
#
# This adapts your educational SFT model to understand:
# 1. GRPO game format (attacker vs assessor prompts)
# 2. Strategic reasoning (how to play the game)
# 3. Pre-fill CoT format (thinking before output)
#
# Training: 1 epoch, low LR to preserve base knowledge
# Time: ~30 minutes for 500 examples

set -e

echo "========================================="
echo "Game Format Adaptation Training"
echo "========================================="
echo ""

# Configuration
MODEL_ID="trainer_output/qwen3_sft_20241021_120000"  # UPDATE THIS to your educational model
DATA_PATH="data/adaptation/game_format_adaptation.jsonl"
OUTPUT_DIR="trainer_output/qwen3_game_adapted_$(date +%Y%m%d_%H%M%S)"

echo "Model: $MODEL_ID"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Data file not found: $DATA_PATH"
    echo ""
    echo "Please run data generation first:"
    echo "  bash run_generate_game_adaptation.sh"
    echo ""
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_ID" ]; then
    echo "❌ Model not found: $MODEL_ID"
    echo ""
    echo "Please update MODEL_ID in this script to point to your educational SFT model"
    echo ""
    exit 1
fi

echo "✅ Data and model found"
echo ""
echo "Starting adaptation training..."
echo "  - Epochs: 1 (preserve base knowledge)"
echo "  - Learning rate: 1e-5 (low to avoid forgetting)"
echo "  - Batch size: 4"
echo "  - Expected time: ~30 minutes"
echo ""

# Run adaptation training
python3 script/train_qwen3_sft.py \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_PATH" \
    --epochs 1 \
    --batch_size 4 \
    --grad_accumulation 4 \
    --learning_rate 1e-5 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "✅ Adaptation training complete!"
echo "========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Run GRPO training"
echo ""
echo "  python3 script/train_selfplay_advanced.py \\"
echo "    --model_id $OUTPUT_DIR \\"
echo "    --num_samples 16 \\"
echo "    --rounds 3"
echo ""
