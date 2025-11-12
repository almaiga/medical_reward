#!/bin/bash

# Run inference on full MEDEC test set
# Usage: ./run_full_inference.sh [model_path] [model_name]

set -e

# Default values
MODEL_PATH=${1:-"Qwen/Qwen3-4B"}
MODEL_NAME=${2:-"qwen3_4b_base"}
DATASET="all"  # Test on both MS and UW datasets
TEMPERATURE=0.7
MAX_TOKENS=512
THINKING_BUDGET=1024
OUTPUT_DIR="results/inference"

echo "=========================================="
echo "🔬 Medical Error Detection - Full Test Set"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Name: $MODEL_NAME"
echo "Dataset: $DATASET (MS + UW)"
echo "Temperature: $TEMPERATURE"
echo "Thinking Budget: $THINKING_BUDGET"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference on full test set
python script/inference_error_detection.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_TOKENS" \
    --thinking_budget "$THINKING_BUDGET" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "✅ Inference completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View summary:"
echo "  cat $OUTPUT_DIR/${MODEL_NAME}_${DATASET}_*_summary.json | jq '.metrics'"
echo ""
echo "View detailed results:"
echo "  head -n 5 $OUTPUT_DIR/${MODEL_NAME}_${DATASET}_*_results.jsonl | jq '.'"
echo ""
