#!/bin/bash

# Selfplay Training Script
# GRPO training with attacker vs assessor

set -e  # Exit on any error

echo "=== Selfplay Training (GRPO) ==="

# Get model path from argument or use your trained model
if [ $# -eq 0 ]; then
    # Use your trained model by default - point to the checkpoint directory
    MODEL_PATH="trainer_output/qwen3-4b-medical-selfplay-sft"
    echo "Using your trained model: $MODEL_PATH"
else
    MODEL_PATH="$1"
    echo "Using specified model: $MODEL_PATH"
fi

# Configuration
JUDGE_MODEL="google/medgemma-4b-it"
NUM_SAMPLES=256
NUM_GENERATIONS=4  # Reduced for more diversity
LEARNING_RATE=5e-6
ROUNDS=4
MAX_ASSESSOR_BATCH=256

# GPU Utilization Settings
PER_DEVICE_BATCH_SIZE=8      # Increased to 8 for better GPU utilization
GRADIENT_ACCUMULATION_STEPS=8 # Keep at 8
# Effective batch size: 8 × 8 = 64

# Optional: Set custom output directory for the final model
# Leave empty to use default (trainer_output/<timestamp>_<model>_grpo_final)
OUTPUT_DIR=""

echo ""
echo "🔍 Checking requirements..."

# Check if selfplay script exists
if [ ! -f "script/train_selfplay_advanced.py" ]; then
    echo "❌ Selfplay script not found: script/train_selfplay_advanced.py"
    exit 1
fi

# Check if model exists (if it's a local path)
if [[ "$MODEL_PATH" == *"trainer_output"* ]] && [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model path not found: $MODEL_PATH"
    echo "Please run SFT training first or provide a valid model path"
    exit 1
fi

# Check if MEDEC data exists
if [ ! -f "data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv" ]; then
    echo "❌ MEDEC data not found"
    echo "Please make sure MEDEC dataset is in: data_copy/MEDEC/MEDEC-MS/"
    exit 1
fi

echo "✅ All requirements found"

echo ""
echo "🎯 Selfplay Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Judge: $JUDGE_MODEL"
echo "  Samples: $NUM_SAMPLES"
echo "  Generations: $NUM_GENERATIONS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Rounds: $ROUNDS"
echo "  Max Assessor Batch: $MAX_ASSESSOR_BATCH"

echo ""
echo "🚀 Starting selfplay training..."

# Build command with optional output_dir
CMD="python3 script/train_selfplay_advanced.py \
    --model_id \"$MODEL_PATH\" \
    --judge_model_id \"$JUDGE_MODEL\" \
    --num_samples $NUM_SAMPLES \
    --num_generations $NUM_GENERATIONS \
    --learning_rate $LEARNING_RATE \
    --rounds $ROUNDS \
    --max_assessor_batch $MAX_ASSESSOR_BATCH \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
    echo "  Output: $OUTPUT_DIR"
fi

eval $CMD

echo ""
echo "✅ Selfplay training complete!"
echo ""
echo "📊 Check results in the results/ directory"
echo "📁 Look for files matching: results/*_grpo_assessor.jsonl"
echo "📁 Interaction logs: results/*_interactions.jsonl"
echo "💾 Final model saved in: trainer_output/"
echo ""
echo "🎉 Training pipeline finished!"