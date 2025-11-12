#!/bin/bash

# Simple inference script for a single model
# Usage: ./run_inference.sh <model_path> [model_name]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [model_name]"
    echo ""
    echo "Examples:"
    echo "  $0 Qwen/Qwen2.5-3B-Instruct"
    echo "  $0 trainer_output/qwen3-4b-medical-selfplay-sft sft_model"
    echo "  $0 Qwen/Qwen2.5-3B-Instruct base_model --dataset ms --max_samples 50"
    exit 1
fi

MODEL_PATH=$1
MODEL_NAME=${2:-$(basename $MODEL_PATH)}

# Shift past the first two arguments
shift
if [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; then
    shift
fi

echo "=========================================="
echo "🔬 Medical Error Detection Inference"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Name: $MODEL_NAME"
echo "=========================================="
echo ""

python script/inference_error_detection.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset all \
    --temperature 0.3 \
    --max_new_tokens 512 \
    --output_dir results/inference \
    "$@"

echo ""
echo "✅ Inference completed!"
echo "Results saved to: results/inference/"
