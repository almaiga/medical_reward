#!/bin/bash

# Download pre-trained Qwen3-4B medical SFT model from Hugging Face

set -e

echo "=== Downloading Pre-trained Model ==="
echo ""
echo "📥 Downloading: Abdine/qwen3-4b-medical-selfplay-sft"
echo "📁 Target: trainer_output/qwen3-4b-medical-selfplay-sft"
echo ""

# Create output directory
mkdir -p trainer_output/qwen3-4b-medical-selfplay-sft

# Download using huggingface-cli
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False

echo ""
echo "✅ Model downloaded successfully!"
echo "📁 Location: trainer_output/qwen3-4b-medical-selfplay-sft"
echo ""
