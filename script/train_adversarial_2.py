import torch
import argparse
import re
import os
import csv
import json
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import Dataset
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer
from tqdm import tqdm

# --- Helper Functions (Unchanged) ---

def get_device():
    """Detects and returns the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_training_model(model_id: str, device: torch.device, is_ppo: bool = True):
    """Loads a model and tokenizer for training."""
    print(f"Loading model: {model_id} (PPO: {is_ppo}) to device: {device}")
    ModelClass = AutoModelForCausalLMWithValueHead if is_ppo else AutoModelForCausalLM
    dtype = torch.float16 if device.type != 'cpu' else torch.float32

    model = ModelClass.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    return model, tokenizer

def parse_thought_and_output(response_text: str, output_marker: str):
    """Extracts chain of thought and final output from a model's response."""
    thought_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else "NO THOUGHT"
    response_without_thought = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    output = response_without_thought.split(output_marker)[-1].strip()
    return thought, output

def load_and_prepare_dataset(num_samples: int):
    """Loads and prepares the MEDEC dataset."""
    print("Loading MEDEC training data from local CSV...")
    try:
        path = "data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
        df = pd.read_csv(path)
        df.dropna(subset=['Text', 'Error Flag'], inplace=True)
        df['Corrected Text'] = df['Corrected Text'].fillna('')
        dataset = Dataset.from_pandas(df)
        return dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    except Exception as e:
        print(f"Failed to load or process dataset. Error: {e}")
        return None

def get_rubric_based_reward(original_note: str, corrected_note: str, judge_model, judge_tokenizer, device) -> torch.Tensor:
    """Calculates reward using the judge model."""
    prompt = f"""[INST] You are a medical expert. Evaluate the 'Corrected Note' based on the 'Original Correct Note'.
Assign a single score from -1.0 to 1.0 based on clinical accuracy and relevance.
-1.0: The correction introduces a new critical error.
0.0: The correction is irrelevant or fails to fix the error.
1.0: The correction is perfect and accurately fixes the clinical error.

Original Correct Note:
{original_note}

Corrected Note:
{corrected_note}

Provide only the numeric score.
Score: [/INST]"""
    
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(judge_model.device)
    outputs = judge_model.generate(**inputs, max_new_tokens=10, temperature=0.0, pad_token_id=judge_tokenizer.eos_token_id)
    response_text = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        reward_str = response_text.split("Score: [/INST]")[1].strip()
        reward = float(re.search(r'(-?\d+\.?\d*)', reward_str).group(1))
    except (IndexError, AttributeError, ValueError):
        reward = 0.0
        
    return torch.tensor(reward, device=device)

def main():
    parser = argparse.ArgumentParser(description="Train a medical LLM with adversarial self-play.")
    parser.add_argument("--model_id", type=str, required=True, help="Base model ID for attacker and defender.")
    parser.add_argument("--judge_model_id", type=str, default="Intelligent-Internet/II-Medical-8B", help="The model ID for the judge.")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of self-play training steps.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # PPOConfig as per the latest trl documentation
    ppo_config = PPOConfig(
        batch_size=1,
        learning_rate=1.41e-5,
        mini_batch_size=1,
    )

    # Generation kwargs
    generation_kwargs = {
        "max_new_tokens": 350,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
    }

    # Load all models
    attacker_model, attacker_tokenizer = load_training_model(args.model_id, device, is_ppo=True)
    defender_model, defender_tokenizer = load_training_model(args.model_id, device, is_ppo=True)
    judge_model, judge_tokenizer = load_training_model(args.judge_model_id, device, is_ppo=False)

    dataset = load_and_prepare_dataset(num_samples=args.num_steps)
    if dataset is None: return

    # Create two separate PPOTrainer instances
    attacker_ppo_trainer = PPOTrainer(
        args=ppo_config, model=attacker_model, tokenizer=attacker_tokenizer, dataset=dataset
    )
    defender_ppo_trainer = PPOTrainer(
        args=ppo_config, model=defender_model, tokenizer=defender_tokenizer, dataset=dataset
    )

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model_id.replace("/", "_")
    log_dir = "results/training"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{timestamp}_{model_name_safe}_{args.num_steps}_adversarial_log.csv"
    log_filepath = os.path.join(log_dir, log_filename)
    log_filename_jsonl = f"{timestamp}_{model_name_safe}_{args.num_steps}_adversarial_log.jsonl"
    log_jsonl_path = os.path.join(log_dir, log_filename_jsonl)
     
    with open(log_filepath, 'w', newline='', encoding='utf-8') as csvfile, \
         open(log_jsonl_path, 'a', encoding='utf-8') as jsonlfile:
         csv_writer = csv.writer(csvfile)
         csv_writer.writerow([
             "timestamp", "step", "source_type", "original_note", 
             "attacker_thought", "attacked_note", "defender_thought", "defender_response", 
             "defender_reward", "attacker_reward"
         ])

         print("\n--- Starting Adversarial Training ---")
         progress_bar = tqdm(enumerate(defender_ppo_trainer.dataloader), total=len(defender_ppo_trainer.dataloader), desc="Adversarial Training")

         for step, batch in progress_bar:
             is_self_play_round = batch['Error Flag'][0] == 0
             source_type = "Self-Play" if is_self_play_round else "Supervised-Correction"
             original_note = batch['Text'][0] if is_self_play_round else batch['Corrected Text'][0]

             if not original_note: continue

             # 1. Attacker's turn
             if is_self_play_round:
                attack_prompt = (
                    "You are an AI red-teaming assistant. Introduce a subtle but clinically significant error into the note.\n"
                    "Provide your reasoning inside <think>...</think>, then output only under '### Note with Error:'.\n\n"
                    f"### Original Note:\n{original_note}\n\n### Your Process:\n"
                )
                attack_query_tensor = attacker_tokenizer.encode(attack_prompt, return_tensors='pt').to(device)
                 
                response_tensor = attacker_ppo_trainer.generate(attack_query_tensor, return_prompt=False, **generation_kwargs)
                full_attack_response = attacker_tokenizer.decode(response_tensor[0], skip_special_tokens=True)
                attacker_thought, attacked_note_text = parse_thought_and_output(full_attack_response, "### Note with Error:")
             else:
                 attacked_note_text = batch['Text'][0]
                 attacker_thought = "N/A (from dataset)"

             if not attacked_note_text: continue

            # 2. Defender's turn
            defend_prompt = (
                "You are a medical assistant. Review the medical note for errors and provide a corrected version.\n"
                "Provide your reasoning inside <think>...</think>, then output only under '### Corrected Note:'.\n\n"
                f"### Medical Note:\n{attacked_note_text}\n\n### Your Process:\n"
            )
            defend_query_tensor = defender_tokenizer.encode(defend_prompt, return_tensors='pt').to(device)
             
            defender_response_tensor = defender_ppo_trainer.generate(defend_query_tensor, return_prompt=False, **generation_kwargs)
            full_defender_response = defender_tokenizer.decode(defender_response_tensor[0], skip_special_tokens=True)
            defender_thought, corrected_note_text = parse_thought_and_output(full_defender_response, "### Corrected Note:")

            # 3. Judge and Reward
            defender_reward = get_rubric_based_reward(original_note, corrected_note_text, judge_model, judge_tokenizer, device)
            attacker_reward = -defender_reward

            # 4. PPO Step for each player
            defender_stats = defender_ppo_trainer.step([defend_query_tensor[0]], [defender_response_tensor[0]], [defender_reward])
             
            if is_self_play_round:
                attacker_stats = attacker_ppo_trainer.step([attack_query_tensor[0]], [response_tensor[0]], [attacker_reward])

            # Logging (CSV row)
            csv_writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step, source_type, original_note,
                attacker_thought, attacked_note_text, defender_thought, corrected_note_text,
                defender_reward.item(), attacker_reward.item() if is_self_play_round else "N/A"
             ])
            # Logging (JSONL object per episode)
            json_entry = {
                "timestamp": datetime.now().isoformat(),
                "episode": int(step),
                "source_type": source_type,
                "original_note": original_note,
                "attacker": {
                    "prompt": attack_prompt if is_self_play_round else None,
                    "thought": attacker_thought,
                    "attacked_note": attacked_note_text,
                    "reward": float(attacker_reward.item()) if is_self_play_round else None,
                 },
                "defender": {
                    "prompt": defend_prompt,
                    "thought": defender_thought,
                    "corrected_note": corrected_note_text,
                    "reward": float(defender_reward.item()),
                },
             }
            jsonlfile.write(json.dumps(json_entry, ensure_ascii=False) + "\n")
            jsonlfile.flush()

            progress_bar.set_description(f"Step {step} | Defender Reward: {defender_reward.item():.2f}")

            print(f"\n--- Training Complete ---")

    # Save models
    os.makedirs("models", exist_ok=True)
    save_path_attacker = f"models/{timestamp}_{model_name_safe}_attacker"
    save_path_defender = f"models/{timestamp}_{model_name_safe}_defender"
    
    attacker_model.save_pretrained(save_path_attacker)
    attacker_tokenizer.save_pretrained(save_path_attacker)
    
    defender_model.save_pretrained(save_path_defender)
    defender_tokenizer.save_pretrained(save_path_defender)
    
    print(f"Log saved to {log_filepath}")
    print(f"JSONL log saved to {log_jsonl_path}")
    print(f"Attacker model saved to {save_path_attacker}")
    print(f"Defender model saved to {save_path_defender}")

if __name__ == "__main__":
    main()