# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm
import os
from datetime import datetime

#Set parameters
model_name = "Qwen/Qwen3-4B-Instruct-2507"
num_samples = 100
useful_columns = ['Text ID', 'Text', 'Sentences', 'Error Flag']
ds = pd.read_csv("data/MEDEC/MEDEC-UW/MEDEC-UW-ValidationSet-with-GroundTruth-and-ErrorType.csv")[useful_columns]
print(ds.head())
print(ds.columns)

prompt_system = "You are a skilled medical doctor reviewing clinical texts for accuracy."
prompt_user = (
    "Each text is either entirely correct or contains exactly one medical error.\n"
    "Medical errors are clinically harmful inaccuracies in patient records.\n"
    "If the error is minor and not clinically harmful mark it as correct.\n"
    "Analyze the following medical note carefully.\n"
    "Medical Note: {medical_note}\n"
    "You may include minimal thinking steps, but keep them concise.\n"
    "Respond in JSON format with two fields:\n"
    '- \"answer\": either \"CORRECT\" or \"INCORRECT\"\n'
    '- \"reasoning\": a brief, single-sentence explanation (no more than 25 words)\n'
    "Do not include any unnecessary text or extended commentary."
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to MPS if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = model.to(device)

# Exclude examples from testing candidates
testing_candidates = ds.copy()

# Select 100 examples for testing
eval_ds = testing_candidates.sample(n=100, random_state=42).reset_index(drop=True)
eval_ids = set(eval_ds["Text ID"])

# Select demo examples from the remaining data (not used for testing)
remaining_ds = testing_candidates[~testing_candidates["Text ID"].isin(eval_ids)]
pos_examples = remaining_ds[remaining_ds["Error Flag"] == 1].head(2)
neg_examples = remaining_ds[remaining_ds["Error Flag"] == 0].head(2)
demo_examples = pd.concat([pos_examples, neg_examples])
demo_ids = set(demo_examples["Text ID"])

# Exclude demo examples from main evaluation
eval_ds = ds[~ds["Text ID"].isin(demo_ids)]

# Prepare in-context examples for the prompt
def format_demo_example(row):
    note = row["Text"]
    label = "INCORRECT" if row["Error Flag"] == 1 else "CORRECT"
    reasoning = "Example reasoning."  # Optionally, you can add a real reasoning if available
    return (
        f"Medical Note: {note}\n"
        f'{{"answer": "{label}", "reasoning": "{reasoning}"}}\n'
    )

in_context_examples = "\n".join([format_demo_example(row) for _, row in demo_examples.iterrows()])

results = []
for idx, row in tqdm(list(eval_ds.iterrows()), desc="Inference"):
    medical_note = row["Text"]
    print("Medical Note: ", medical_note)
    error_flag = row["Error Flag"]  # 1: incorrect, 0: correct

    # Add in-context examples to the user prompt
    formatted_user = (
        in_context_examples +
        "\n" +
        prompt_user.format(medical_note=medical_note)
    )

    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": formatted_user}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    import time
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=1024)  # Reduce max_new_tokens to save GPU memory
    end_time = time.time()

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

    # Convert ground truth and model output for comparison
    gt_label = "INCORRECT" if error_flag == 1 else "CORRECT"
    # Try to extract answer from model response
    import re
    match = re.search(r'"answer"\s*:\s*"?(CORRECT|INCORRECT)"?', response)
    model_label = match.group(1) if match else "UNKNOWN"

    print(f"Example {idx}:")
    print("Model response:", response)
    print("Model label:", model_label)
    print("Ground truth Error Flag:", error_flag, "->", gt_label)
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    print("-" * 60)

    results.append({
        "idx": idx,
        "response": response,
        "model_label": model_label,
        "error_flag": error_flag,
        "gt_label": gt_label,
        "inference_time": end_time - start_time
    })

    # Free up GPU memory after each inference (if using CUDA)
    if device.type == "cuda":
        torch.cuda.empty_cache()

# Save results to CSV with timestamp, model, and sample count in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_short = model_name.replace("/", "_")
results_dir = "data/results"
os.makedirs(results_dir, exist_ok=True)
results_path = f"{results_dir}/results_{model_short}_{num_samples}_{timestamp}.csv"

pd.DataFrame(results).to_csv(results_path, index=False)
print(f"Results saved to {results_path}")