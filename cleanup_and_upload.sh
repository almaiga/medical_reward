#!/bin/bash

# Cleanup and Upload Script
# Run this after training completes

set -e

echo "=========================================="
echo "MODEL CLEANUP AND UPLOAD"
echo "=========================================="

# Get the latest training directory
LATEST_MODEL=$(ls -td trainer_output/qwen3_sft_complete_* | head -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "❌ No trained model found"
    exit 1
fi

echo "📁 Found model: $LATEST_MODEL"

# Clean up intermediate checkpoints (keep only final model)
if ls "$LATEST_MODEL"/checkpoint-* 1> /dev/null 2>&1; then
    echo ""
    echo "🗑️  Removing intermediate checkpoints..."
    rm -rf "$LATEST_MODEL"/checkpoint-*
    echo "✅ Checkpoints deleted"
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
