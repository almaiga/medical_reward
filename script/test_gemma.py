model_name = "google/medgemma-4b-it"

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

#Set parameters
ds = load_dataset("wenhu/Health-Bench")["test"]
prompt = ds[0]["prompt"]
print(prompt)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

messages = [
    prompt
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
outputs = model.generate(**inputs, max_new_tokens=2048)
end_time = time.time()

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
print(f"Inference time: {end_time - start_time:.2f} seconds")