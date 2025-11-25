#!/bin/bash

# Cleanup and Upload Script
# Run this after training completes

set -e

echo "=========================================="
echo "MODEL CLEANUP AND UPLOAD"
echo "=========================================="

# Get the latest training directory
LATEST_MODEL=$(ls -td trainer_output/*/ | head -1 | sed 's:/$::')

if [ -z "$LATEST_MODEL" ]; then
    echo "❌ No trained model found"
    exit 1
fi

echo "📁 Found model: $LATEST_MODEL"

# Check for checkpoints
CHECKPOINTS=$(find "$LATEST_MODEL" -type d -name "checkpoint-*" 2>/dev/null)

if [ -n "$CHECKPOINTS" ]; then
    echo ""
    echo "🗑️  Found checkpoints to delete:"
    echo "$CHECKPOINTS"
    
    # Calculate size
    CHECKPOINT_SIZE=$(du -sh "$LATEST_MODEL"/checkpoint-* 2>/dev/null | awk '{sum+=$1} END {print sum}')
    echo "💾 Space to free: ~${CHECKPOINT_SIZE}GB"
    
    read -p "Delete checkpoints? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$LATEST_MODEL"/checkpoint-*
        echo "✅ Checkpoints deleted"
    fi
else
    echo "✅ No checkpoints found (already clean)"
fi

echo ""
echo "📊 Final model size:"
du -sh "$LATEST_MODEL"

echo ""
echo "=========================================="
echo "UPLOAD TO HUGGING FACE"
echo "=========================================="

read -p "Enter your Hugging Face username: " HF_USERNAME
read -p "Enter model name (e.g., qwen3-4b-medical-sft): " MODEL_NAME

echo ""
echo "📤 Uploading to: $HF_USERNAME/$MODEL_NAME"
echo "📁 From: $LATEST_MODEL"

read -p "Proceed with upload? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli upload "$HF_USERNAME/$MODEL_NAME" \
        "$LATEST_MODEL" \
        --repo-type model
    
    echo ""
    echo "✅ Upload complete!"
    echo "🔗 Model available at: https://huggingface.co/$HF_USERNAME/$MODEL_NAME"
    echo ""
    echo "To use your model:"
    echo "  from transformers import AutoModelForCausalLM"
    echo "  model = AutoModelForCausalLM.from_pretrained('$HF_USERNAME/$MODEL_NAME')"
else
    echo "Upload cancelled"
fi

echo ""
echo "=========================================="
echo "DONE"
echo "=========================================="
