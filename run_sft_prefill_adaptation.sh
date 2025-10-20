#!/bin/bash

# Pre-Fill Adaptation Training Script
# Trains model on pre-fill CoT format to bridge SFT → GRPO gap
# Uses format consistent with educational SFT: <think>reasoning</think><output>response</output>

set -e  # Exit on any error

echo "=== Pre-Fill Adaptation Training ==="

# Get base model path from argument or use default
if [ $# -eq 0 ]; then
    # Use your educational SFT model by default
    BASE_MODEL="trainer_output/qwen3_trl_20251020_183808"
    echo "Using default base model: $BASE_MODEL"
else
    BASE_MODEL="$1"
    echo "Using specified base model: $BASE_MODEL"
fi

# Configuration
ADAPTATION_DATA="data/adaptation/prefill_cot_adaptation.jsonl"
OUTPUT_DIR="trainer_output/qwen3_adapted_prefill"

echo ""
echo "🔍 Checking requirements..."

# Check if base model exists
if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ Base model not found: $BASE_MODEL"
    echo "Please run SFT training first or provide a valid model path"
    exit 1
fi

# Check if adaptation data exists, if not generate it
if [ ! -f "$ADAPTATION_DATA" ]; then
    echo "⚠️  Adaptation data not found, generating..."
    python3 script/generate_prefill_adaptation_data.py \
        --start_id 733 \
        --num_rows 200 \
        --output_path "$ADAPTATION_DATA"
else
    echo "✅ Found adaptation data: $ADAPTATION_DATA"
fi

# Count examples
NUM_EXAMPLES=$(wc -l < "$ADAPTATION_DATA")
echo "✅ Found base model: $BASE_MODEL"
echo "✅ Adaptation data: $NUM_EXAMPLES examples"

echo ""
echo "📊 Adaptation Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Data: $ADAPTATION_DATA"
echo "  Examples: $NUM_EXAMPLES"
echo "  Format: PRE-FILL (<think> first, then <output>)"
echo "  Epochs: 1 (prevent overfitting)"
echo "  Learning Rate: 1e-5 (half of original SFT)"
echo "  Batch Size: 4"
echo "  Gradient Accumulation: 4 (effective batch = 16)"
echo "  Output: $OUTPUT_DIR"

echo ""
echo "🎯 What This Does:"
echo "  - Teaches model GRPO's prompt format"
echo "  - Uses PRE-FILL format (matches educational SFT)"
echo "  - Preserves medical knowledge from educational SFT"
echo "  - Bridges gap between educational and task-focused styles"

echo ""
echo "⏱️  Estimated time: ~10-15 minutes"

echo ""
read -p "Press Enter to start adaptation training (or Ctrl+C to cancel)..."

echo ""
echo "🚀 Starting adaptation training..."

python3 script/train_qwen3_trl.py \
    --model_id "$BASE_MODEL" \
    --data_path "$ADAPTATION_DATA" \
    --epochs 1 \
    --batch_size 4 \
    --grad_accumulation 4 \
    --learning_rate 1e-5 \
    --output_dir "$OUTPUT_DIR" \
    --test_format

echo ""
echo "✅ Adaptation training complete!"

# Find the adapted model
LATEST_OUTPUT=$(ls -td ${OUTPUT_DIR}_* 2>/dev/null | head -1)
if [ -n "$LATEST_OUTPUT" ]; then
    echo "📁 Adapted model saved to: $LATEST_OUTPUT"
    echo ""
    echo "🎯 Next step: Run selfplay training"
    echo "   bash run_selfplay_training.sh $LATEST_OUTPUT"
    echo ""
    echo "Expected improvements:"
    echo "  ✅ Model generates valid responses with correct format"
    echo "  ✅ <think>reasoning</think><output>response</output>"
    echo "  ✅ 60-80% faithfulness pass rate"
    echo "  ✅ Non-zero reward variance (enables learning)"
    echo "  ✅ No mode collapse"
else
    echo "⚠️  Could not find adapted model output"
fi

echo ""
echo "🎉 Adaptation pipeline finished!"
