import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset # Or your custom data loading function

# 1. Load your Model and Tokenizer
model_id = "Qwen/Qwen3-4B" # Or "Qwen/Qwen3-4B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load your "Golden" SFT Dataset
# This should be a dataset with a 'text' column where each entry is a complete
# prompt-and-response pair (e.g., "USER: Do X\nASSISTANT: <think>...</think><output>...</output>")
sft_dataset = load_dataset("your_username/your_sft_dataset", split="train")

# 3. Configure Training Arguments
training_args = TrainingArguments(
    output_dir="./sft_model_checkpoint",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_bf16_supported(), # Use bf16 if available
)

# 4. Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=sft_dataset,
    max_seq_length=2048,
    dataset_text_field="text", # The column in your dataset with the prompt/response text
)

# 5. Start Training
print("Starting Supervised Fine-Tuning...")
trainer.train()

# 6. Save the final model
final_model_path = "./my-medical-sft-model"
print(f"Saving final SFT model to {final_model_path}...")
trainer.save_model(final_model_path)