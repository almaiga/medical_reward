# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from tqdm import tqdm
import os
from datetime import datetime

#Set parameters
model_name = "Qwen/Qwen3-8B"
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

results = []
for idx, row in tqdm(list(ds.head(num_samples).iterrows()), desc="Inference"):
    medical_note = row["Text"]
    print("Medical Note: ", medical_note)
    error_flag = row["Error Flag"]  # 1: incorrect, 0: correct

    formatted_user = prompt_user.format(medical_note=medical_note)
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
    outputs = model.generate(**inputs, max_new_tokens=1024)
    end_time = time.time()

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    print(f"Example {idx}:")
    print("Model response:", response)
    print("Ground truth Error Flag:", error_flag)
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    print("-" * 60)

    results.append({
        "idx": idx,
        "response": response,
        "error_flag": error_flag,
        "inference_time": end_time - start_time
    })

# Save results to CSV with timestamp, model, and sample count in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_short = model_name.replace("/", "_")
results_dir = "data/results"
os.makedirs(results_dir, exist_ok=True)
results_path = f"{results_dir}/results_{model_short}_{num_samples}_{timestamp}.csv"

pd.DataFrame(results).to_csv(results_path, index=False)
print(f"Results saved to {results_path}")