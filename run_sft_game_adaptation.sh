#!/bin/bash
# Train Game Format Adaptation (3 epochs)
#
# This adapts your educational SFT model to understand:
# 1. GRPO game format (attacker vs assessor prompts)
# 2. Copy/modify skill (copy entire notes, make subtle changes)
# 3. Pre-fill CoT format (thinking before output)
#
# Training: 3 epochs to internalize copy/modify behavior
# Time: ~90 minutes for 500 examples × 3 epochs

set -e

echo "========================================="
echo "Game Format Adaptation Training"
echo "========================================="
echo ""

# ============================================
# CONFIGURATION - Update these paths
# ============================================
MODEL_ID="trainer_output/qwen3_trl_20251020_142117"  # Your educational SFT model
DATA_PATH="data/adaptation/game_format_adaptation.jsonl"  # Game adaptation data (500 examples)
OUTPUT_DIR="trainer_output/qwen3_game_adapted_$(date +%Y%m%d_%H%M%S)"

# Training hyperparameters (optimized for copy/modify learning)
EPOCHS=3                    # Enough repetition to internalize behavior
BATCH_SIZE=4                # Per-device batch size
GRAD_ACCUMULATION=4         # Effective batch = 16
LEARNING_RATE=1e-5          # Low LR to preserve base knowledge
MAX_SEQ_LENGTH=2048         # Enough for full medical notes

echo "📋 Configuration:"
echo "  Base model: $MODEL_ID"
echo "  Training data: $DATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "⚙️  Hyperparameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUMULATION)))"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max sequence: $MAX_SEQ_LENGTH tokens"
echo ""

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Data file not found: $DATA_PATH"
    echo ""
    echo "Please run data generation first:"
    echo "  bash run_generate_game_adaptation.sh"
    echo ""
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_ID" ]; then
    echo "❌ Model not found: $MODEL_ID"
    echo ""
    echo "Please update MODEL_ID in this script to point to your educational SFT model"
    echo ""
    exit 1
fi

echo "✅ Data and model found"
echo ""

# Count examples in data
EXAMPLE_COUNT=$(wc -l < "$DATA_PATH")
echo "Training data: $EXAMPLE_COUNT examples"
echo ""

echo "Starting adaptation training..."
echo "  - Epochs: 3 (enough repetition to learn copy/modify skill)"
echo "  - Learning rate: 1e-5 (low to avoid forgetting base knowledge)"
echo "  - Batch size: 4 per device"
echo "  - Gradient accumulation: 4 steps (effective batch = 16)"
echo "  - Max sequence length: 2048 tokens"
echo "  - Total training steps: ~$((EXAMPLE_COUNT * 3 / 16))"
echo "  - Expected time: ~90 minutes"
echo ""
echo "Training objectives:"
echo "  1. Learn to COPY medical notes word-for-word (safe game)"
echo "  2. Learn to make ONE subtle modification (harmful game)"
echo "  3. Maintain <think>reasoning</think><output>content</output> format"
echo ""

# Run adaptation training with detailed configuration
python3 script/train_qwen3_trl.py \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accumulation "$GRAD_ACCUMULATION" \
    --learning_rate "$LEARNING_RATE" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --test_format

echo ""
echo "========================================="
echo "✅ Adaptation training complete!"
echo "========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "📊 Training Summary:"
echo "  - Base model: $MODEL_ID"
echo "  - Training data: $EXAMPLE_COUNT examples × 3 epochs"
echo "  - Format: PRE-FILL CoT (<think>...</think><output>...</output>)"
echo "  - Skills learned: Copy medical notes + Make subtle modifications"
echo ""
echo "🧪 Next step: Test the model can copy before GRPO"
echo ""
echo "  # Quick test:"
echo "  python3 -c \""
echo "  from transformers import AutoTokenizer, AutoModelForCausalLM"
echo "  import torch"
echo "  model = AutoModelForCausalLM.from_pretrained('$OUTPUT_DIR', torch_dtype=torch.bfloat16, device_map='auto')"
echo "  tok = AutoTokenizer.from_pretrained('$OUTPUT_DIR')"
echo "  # Test if model can copy a note..."
echo "  \""
echo ""
echo "🎮 If test passes, run GRPO training:"
echo ""
echo "  python3 script/train_selfplay_advanced.py \\"
echo "    --model_id $OUTPUT_DIR \\"
echo "    --num_samples 16 \\"
echo "    --rounds 3"
echo ""