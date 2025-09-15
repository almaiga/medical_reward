models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "Intelligent-Internet/II-Medical-8B" 
)

for model in "${models[@]}"; do
    if [ -n "$model" ]; then
        python script/run_baseline.py --model_id "$model" --output_dir results || echo "Error running model: $model"
    fi
done
