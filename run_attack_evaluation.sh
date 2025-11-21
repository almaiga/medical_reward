#!/bin/bash
# Evaluate attack plausibility and difficulty using medgemma-4b

set -e

# Default values
INPUT_FILE="data/Trainer Output Interactions.jsonl"
MAX_SAMPLES=50
MODEL="google/medgemma-4b-it"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --quick-view)
            QUICK_VIEW=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--input FILE] [--max-samples N] [--model MODEL] [--quick-view]"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "Attack Plausibility Evaluation"
echo "=================================================="
echo "Input: $INPUT_FILE"
echo "Max samples: $MAX_SAMPLES"
echo "Model: $MODEL"
echo "=================================================="

if [ "$QUICK_VIEW" = true ]; then
    echo "Running quick comparison view..."
    python quick_attack_comparison.py \
        --input "$INPUT_FILE" \
        --num-examples 10 \
        --random
else
    echo "Running full evaluation with medgemma-4b..."
    python evaluate_attack_plausibility.py \
        --input "$INPUT_FILE" \
        --max-samples "$MAX_SAMPLES" \
        --model "$MODEL"
fi

echo ""
echo "Evaluation complete!"
