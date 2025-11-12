#!/bin/bash

# Medical Error Detection Inference - Model Comparison Script
# Tests different Qwen model versions on MEDEC test data

set -e

echo "=========================================="
echo "🔬 Medical Error Detection Inference"
echo "=========================================="
echo ""

# Configuration
DATASET="all"  # Options: ms, uw, all
MAX_SAMPLES=""  # Leave empty for all samples, or set a number like 50
TEMPERATURE=0.3
MAX_TOKENS=512
OUTPUT_DIR="results/inference"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset ms|uw|all] [--max_samples N] [--temperature T]"
            exit 1
            ;;
    esac
done

# Build common arguments
COMMON_ARGS="--dataset $DATASET --temperature $TEMPERATURE --max_new_tokens $MAX_TOKENS --output_dir $OUTPUT_DIR"
if [ -n "$MAX_SAMPLES" ]; then
    COMMON_ARGS="$COMMON_ARGS --max_samples $MAX_SAMPLES"
fi

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Max samples: ${MAX_SAMPLES:-all}"
echo "  Temperature: $TEMPERATURE"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Function to run inference
run_inference() {
    local model_path=$1
    local model_name=$2
    local extra_args=$3
    
    echo "=========================================="
    echo "Testing: $model_name"
    echo "Model: $model_path"
    echo "=========================================="
    
    if [ ! -d "$model_path" ] && [[ ! "$model_path" =~ ^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$ ]]; then
        echo "⚠️  Model not found: $model_path"
        echo "Skipping..."
        echo ""
        return
    fi
    
    python script/inference_error_detection.py \
        --model_path "$model_path" \
        --model_name "$model_name" \
        $COMMON_ARGS \
        $extra_args
    
    echo ""
    echo "✅ Completed: $model_name"
    echo ""
}

# ============================================
# Test 1: Base Model (Qwen2.5-3B-Instruct)
# ============================================
echo "📦 Test 1: Base Model"
run_inference "Qwen/Qwen2.5-3B-Instruct" "base_qwen2.5-3b" ""

# ============================================
# Test 2: Fine-tuned Model (SFT)
# ============================================
echo "📦 Test 2: Fine-tuned Model (SFT)"
SFT_MODEL="trainer_output/qwen3-4b-medical-selfplay-sft"
if [ -d "$SFT_MODEL" ]; then
    run_inference "$SFT_MODEL" "sft_medical" ""
else
    echo "⚠️  SFT model not found at: $SFT_MODEL"
    echo "Run download_pretrained_model.sh first or train your own model"
    echo ""
fi

# ============================================
# Test 3: Self-Play Model (SFT + GRPO)
# ============================================
echo "📦 Test 3: Self-Play Model (SFT + GRPO)"
# Find the latest self-play checkpoint
SELFPLAY_MODEL=$(find trainer_output -maxdepth 1 -type d -name "*selfplay*" ! -name "*sft" | sort -r | head -n 1)
if [ -n "$SELFPLAY_MODEL" ] && [ -d "$SELFPLAY_MODEL" ]; then
    run_inference "$SELFPLAY_MODEL" "selfplay_medical" ""
else
    echo "⚠️  Self-play model not found in trainer_output/"
    echo "Train a model using run_selfplay_training.sh first"
    echo ""
fi

# ============================================
# Test 4: Abliterated Model (if available)
# ============================================
echo "📦 Test 4: Abliterated Model (Optional)"
# Check if user has an abliterated model
ABLITERATED_MODEL="trainer_output/qwen-abliterated"
if [ -d "$ABLITERATED_MODEL" ]; then
    run_inference "$ABLITERATED_MODEL" "abliterated" ""
else
    echo "⚠️  Abliterated model not found at: $ABLITERATED_MODEL"
    echo "This is optional - skipping"
    echo ""
fi

# ============================================
# Test 5: Zero-shot (no few-shot examples)
# ============================================
echo "📦 Test 5: Base Model (Zero-shot)"
run_inference "Qwen/Qwen2.5-3B-Instruct" "base_zeroshot" "--no_few_shot"

# ============================================
# Test 6: No CoT (direct prediction)
# ============================================
echo "📦 Test 6: Base Model (No CoT)"
run_inference "Qwen/Qwen2.5-3B-Instruct" "base_no_cot" "--no_cot"

# ============================================
# Summary
# ============================================
echo "=========================================="
echo "✅ All inference tests completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  ls -lh $OUTPUT_DIR"
echo ""
echo "To compare metrics:"
echo "  cat $OUTPUT_DIR/*_summary.json | jq '.metrics'"
echo ""
