#!/bin/bash

# Selfplay Training Script
# GRPO training with attacker vs assessor

set -e  # Exit on any error

echo "=== Selfplay Training (GRPO) ==="

# Get model path from argument or use your trained model
if [ $# -eq 0 ]; then
    # Use your trained model by default
    MODEL_PATH="trainer_output/qwen3_trl_20251007_110755"
    echo "Using your trained model: $MODEL_PATH"
else
    MODEL_PATH="$1"
    echo "Using specified model: $MODEL_PATH"
fi

# Configuration
JUDGE_MODEL="mlabonne/Qwen3-4B-abliterated"
NUM_SAMPLES=16
NUM_GENERATIONS=4  # Increased for better reward variance
LEARNING_RATE=1e-5
ROUNDS=2
MAX_ASSESSOR_BATCH=64

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

python3 script/train_selfplay_advanced.py \
    --model_id "$MODEL_PATH" \
    --judge_model_id "$JUDGE_MODEL" \
    --num_samples $NUM_SAMPLES \
    --num_generations $NUM_GENERATIONS \
    --learning_rate $LEARNING_RATE \
    --rounds $ROUNDS \
    --max_assessor_batch $MAX_ASSESSOR_BATCH

echo ""
echo "✅ Selfplay training complete!"
echo ""
echo "📊 Check results in the results/ directory"
echo "📁 Look for files matching: results/*_grpo_assessor.jsonl"
echo "📁 Interaction logs: results/*_interactions.jsonl"
echo ""
echo "🎉 Training pipeline finished!"