#!/bin/bash

# SFT Training Script for Qwen with TRL
# Supervised Fine-Tuning only

set -e  # Exit on any error

echo "=== Qwen SFT Training with TRL ==="

# Configuration
MODEL_ID="mlabonne/Qwen3-4B-abliterated"
DATA_PATH="data/sft_training/20251017_161801_sft_merged.jsonl"
OUTPUT_DIR="trainer_output/qwen3_trl"

echo ""
echo "🔍 Checking required files..."

# Check if original data exists
if [ ! -f "data/sft_training/20251017_161801_sft_merged.jsonl" ]; then
    echo "❌ Original data file not found"
    echo "Please make sure you have the SFT training data."
    exit 1
fi

# Check if fixed data exists, if not create it
if [ ! -f "$DATA_PATH" ]; then
    echo "⚠️  Creating fixed data file..."
    python3 script/fix_sft_format.py \
        data/sft_training/20251017_161801_sft_merged.jsonl \
        --output_path "$DATA_PATH"
else
    echo "✅ Found: $DATA_PATH"
fi

echo ""
echo "📊 Validating SFT data..."
python3 script/validate_sft_data.py "$DATA_PATH"

echo ""
echo "🧪 Testing TRL compatibility..."
python3 script/test_trl_compatibility.py "$DATA_PATH"

echo ""
echo "🚀 Starting SFT training..."
echo "💡 TIP: Open another terminal and run this to monitor progress:"
echo "   python3 script/monitor_training.py --training_dir $OUTPUT_DIR"
echo ""
echo "🔄 Training will show detailed progress with time estimates..."
echo ""

python3 script/train_qwen3_trl.py \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_PATH" \
    --epochs 3 \
    --batch_size 2 \
    --grad_accumulation 8 \
    --learning_rate 2e-5 \
    --output_dir "$OUTPUT_DIR" \
    --test_format

echo ""
echo "✅ SFT training complete!"

# Find the trained model
LATEST_OUTPUT=$(ls -td ${OUTPUT_DIR}_* 2>/dev/null | head -1)
if [ -n "$LATEST_OUTPUT" ]; then
    echo "📁 Model saved to: $LATEST_OUTPUT"
    echo ""
    echo "🎯 Next step: Run selfplay training"
    echo "   bash run_selfplay_training.sh $LATEST_OUTPUT"
else
    echo "⚠️  Could not find trained model output"
fi