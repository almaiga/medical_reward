#!/bin/bash

# SFT Training Script for Qwen with TRL
# Single-stage training on merged educational + adaptation data

set -e  # Exit on any error

# Disable hf_transfer to avoid issues
unset HF_HUB_ENABLE_HF_TRANSFER

echo "=== Qwen SFT Training with Clean Stratified Data ==="

# Configuration
MODEL_ID="mlabonne/Qwen3-8B-abliterated"
EDUCATIONAL_DATA="data/sft_clean/educational_stratified.jsonl"
ADAPTATION_DATA="data/sft_clean/adaptation_stratified.jsonl"
MERGED_DATA="data/sft_clean/merged_all.jsonl"
OUTPUT_DIR="trainer_output/qwen3_sft_complete"

echo ""
echo "🔍 Checking required files..."

# Check if clean data exists
if [ ! -f "$EDUCATIONAL_DATA" ]; then
    echo "❌ Educational data not found: $EDUCATIONAL_DATA"
    echo "Please run: python3 script/organize_clean_data.py"
    exit 1
fi

if [ ! -f "$ADAPTATION_DATA" ]; then
    echo "❌ Adaptation data not found: $ADAPTATION_DATA"
    echo "Please run: python3 script/organize_clean_data.py"
    exit 1
fi

echo "✅ Found: $EDUCATIONAL_DATA"
echo "✅ Found: $ADAPTATION_DATA"

# Merge datasets
echo ""
echo "🔗 Merging educational + adaptation data..."
cat "$EDUCATIONAL_DATA" "$ADAPTATION_DATA" > "$MERGED_DATA"
echo "✅ Created: $MERGED_DATA"

# Count examples
TOTAL_EXAMPLES=$(wc -l < "$MERGED_DATA")
echo ""
echo "📊 Training Data Summary:"
echo "  Educational: 913 notes (75% of MEDEC, all 5 error types)"
echo "  Adaptation: 306 notes → 1,224 examples (25% of MEDEC, game format)"
echo "  Total examples: $TOTAL_EXAMPLES"
echo ""

# ============================================================================
# SINGLE-STAGE SFT TRAINING
# ============================================================================

echo "=" | tr '\n' '=' | head -c 70; echo ""
echo "SFT TRAINING ON MERGED DATA"
echo "=" | tr '\n' '=' | head -c 70; echo ""
echo ""
echo "📚 Training on educational + adaptation data..."
echo "💡 TIP: Open another terminal to monitor:"
echo "   python3 script/monitor_training.py --training_dir $OUTPUT_DIR"
echo ""

python3 script/train_qwen3_trl.py \
    --model_id "$MODEL_ID" \
    --data_path "$MERGED_DATA" \
    --epochs 3 \
    --batch_size 1 \
    --grad_accumulation 16 \
    --learning_rate 2e-5 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "✅ SFT training complete!"
echo "📁 Model saved to: $OUTPUT_DIR"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "=" | tr '\n' '=' | head -c 70; echo ""
echo "✅ SFT TRAINING COMPLETE!"
echo "=" | tr '\n' '=' | head -c 70; echo ""
echo ""
echo "📊 Training Summary:"
echo "  Model: $OUTPUT_DIR"
echo "  Examples trained: $TOTAL_EXAMPLES"
echo "  Epochs: 3"
echo ""
echo "🎯 Next Step: Run GRPO Self-Play Training"
echo ""
echo "   python3 script/train_selfplay_advanced.py \\"
echo "     --model_id $OUTPUT_DIR \\"
echo "     --num_samples 16 \\"
echo "     --rounds 3"
echo ""
echo "=" | tr '\n' '=' | head -c 70; echo ""