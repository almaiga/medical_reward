import torch
import argparse
import re
import os
import csv
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer
from tqdm import tqdm

def get_device():
    """Detects and returns the best available device."""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
        return torch.device("mps")
    else:
        print("No GPU available. Using CPU.")
        return torch.device("cpu")

def load_training_model(model_id: str, device: torch.device, is_ppo: bool = True):
    """
    Loads a model and tokenizer, placing it on the specified device.
    """
    print(f"Loading model: {model_id} (PPO: {is_ppo}) to device: {device}")
    
    ModelClass = AutoModelForCausalLMWithValueHead if is_ppo else AutoModelForCausalLM
    
    dtype = torch.float16 if device.type != 'cpu' else torch.float32

    model = ModelClass.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    return model, tokenizer

def get_rubric_based_reward(original_note: str, corrected_note: str, judge_model, judge_tokenizer, device) -> torch.Tensor:
    """
    Calculates a reward score by asking a judge model to evaluate the correction.
    """
    prompt = f"""
            [INST] You are a medical expert. Evaluate the 'Corrected Note' based on the 'Original Correct Note'.
            Assign a single score from -1.0 to 1.0 based on clinical accuracy and relevance.
                -1.0: The correction introduces a new critical error or makes the note significantly worse.
                0.0: The correction is irrelevant, fails to fix the error, or is incomplete.
                1.0: The correction is perfect and accurately fixes the clinical error.

            Original Correct Note:
            {original_note}

            Corrected Note:
            {corrected_note}

            Provide only the numeric score.
            Score: [/INST]
            """
    
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(judge_model.device)
    outputs = judge_model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=judge_tokenizer.eos_token_id)
    response_text = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        reward_str = response_text.split("Score: [/INST]")[1].strip()
        reward = float(re.search(r'(-?\d+\.?\d*)', reward_str).group(1))
    except (IndexError, AttributeError, ValueError):
        reward = 0.0
        
    return torch.tensor(reward, device=device)

def parse_thought_and_output(response_text: str, output_marker: str):
    """
    Extracts the chain of thought and the final output from the model's response.
    """
    thought_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else "NO THOUGHT"
    
    response_without_thought = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    output = response_without_thought.split(output_marker)[-1].strip()
    return thought, output

def load_and_prepare_dataset(num_samples: int):
    """Loads and prepares the MEDEC dataset from a local CSV file."""
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

def main():
    parser = argparse.ArgumentParser(description="Train a medical LLM with adversarial self-play.")
    parser.add_argument("--model_id", type=str, required=True, help="Base model ID for attacker and defender.")
    parser.add_argument("--judge_model_id", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="The model ID for the judge.")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of self-play training steps.")
    args = parser.parse_args()

    device = get_device()
    
    ppo_config = PPOConfig(
        batch_size=1,
        learning_rate=1.41e-5,
        mini_batch_size=1,
    )

    # Base generation kwargs
    generation_kwargs = {
        "max_new_tokens": 350,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": 128001 # Explicitly set pad_token_id for generation
    }

    attacker_model, attacker_tokenizer = load_training_model(args.model_id, device, is_ppo=True)
    defender_model, defender_tokenizer = load_training_model(args.model_id, device, is_ppo=True)
    judge_model, judge_tokenizer = load_training_model(args.judge_model_id, device, is_ppo=False)

    print("Loading reference models...")
    attacker_ref_model, _ = load_training_model(args.model_id, device, is_ppo=False)
    defender_ref_model, _ = load_training_model(args.model_id, device, is_ppo=False)

    dataset = load_and_prepare_dataset(num_samples=args.num_steps)
    if dataset is None: return

    attacker_ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=attacker_model,
        ref_model=attacker_ref_model,  # The required reference model
        tokenizer=attacker_tokenizer,
        dataset=dataset
    )

    defender_ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=defender_model,
        ref_model=defender_ref_model,  # The required reference model
        tokenizer=defender_tokenizer,
        dataset=dataset
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model_id.replace("/", "_")
    log_dir = "results/training"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{timestamp}_{model_name_safe}_{args.num_steps}_adversarial_log.csv"
    log_filepath = os.path.join(log_dir, log_filename)
    
    with open(log_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "timestamp", "step", "source_type", "original_note", 
            "attacker_thought", "attacked_note", "defender_thought", "defender_response", 
            "defender_reward", "attacker_reward"
        ])

        print("\n--- Starting Adversarial Training ---")
        progress_bar = tqdm(range(args.num_steps), desc="Adversarial Training")

        for step in progress_bar:
            row = dataset[step % len(dataset)]
            
            is_self_play_round = row['Error Flag'] == 0
            source_type = "Self-Play" if is_self_play_round else "Supervised-Correction"
            
            original_note = row['Text'] if is_self_play_round else row['Corrected Text']
            if not original_note: continue

            if is_self_play_round:
                attack_prompt = f"""
                                    You are an AI red-teaming assistant. 
                                    Your goal is to introduce a subtle but clinically significant error into a medical note to test a defender AI.
                                    
                                    First, think step-by-step inside <think> tags about what error to introduce. 
                                    Then, provide only the modified note under '### Note with Error:'.

                                    ### Original Note:
                                    {original_note}

                                    ### Your Process:
                                """
                attack_inputs = attacker_tokenizer(attack_prompt, return_tensors="pt").to(device)
                
                # CORRECTED: Add attention_mask to generation_kwargs
                current_generation_kwargs = generation_kwargs.copy()
                current_generation_kwargs["attention_mask"] = attack_inputs["attention_mask"]

                # CORRECTED: Pass the 1D input_ids tensor to generate
                attack_response_tensor = attacker_ppo_trainer.generate(
                    attack_inputs["input_ids"][0],
                    **current_generation_kwargs
                )
                full_attack_response = attacker_tokenizer.decode(attack_response_tensor[0], skip_special_tokens=True)
                attacker_thought, attacked_note_text = parse_thought_and_output(full_attack_response, "### Note with Error:")
            else:
                attacked_note_text = row['Text']
                attacker_thought = "N/A (from dataset)"
                attack_inputs = None
                attack_response_tensor = None

            if not attacked_note_text: continue

            defend_prompt = f"""You are a medical assistant.
                                Review the note for errors and provide a corrected version.
                                First, think step-by-step inside <think> tags to analyze the note. 
                                Then, provide only the corrected note under '### Corrected Note:'.

                                ### Medical Note:
                                {attacked_note_text}

                                ### Your Process:
                            """
            defend_inputs = defender_tokenizer(defend_prompt, return_tensors="pt").to(device)
            
            # CORRECTED: Add attention_mask to generation_kwargs
            current_generation_kwargs = generation_kwargs.copy()
            current_generation_kwargs["attention_mask"] = defend_inputs["attention_mask"]

            # CORRECTED: Pass the 1D input_ids tensor to generate
            defender_response_tensor = defender_ppo_trainer.generate(
                defend_inputs["input_ids"][0],
                **current_generation_kwargs
            )
            full_defender_response = defender_tokenizer.decode(defender_response_tensor[0], skip_special_tokens=True)
            defender_thought, corrected_note_text = parse_thought_and_output(full_defender_response, "### Corrected Note:")

            defender_reward = get_rubric_based_reward(original_note, corrected_note_text, judge_model, judge_tokenizer, device)
            attacker_reward = -defender_reward

            # The step function needs the 2D tensors
            defender_ppo_trainer.step([defend_inputs['input_ids'][0]], [defender_response_tensor[0]], [defender_reward])
            
            if is_self_play_round and attack_inputs is not None:
                attacker_ppo_trainer.step([attack_inputs['input_ids'][0]], [attack_response_tensor[0]], [attacker_reward])

            csv_writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step, source_type, original_note,
                attacker_thought, attacked_note_text, defender_thought, corrected_note_text,
                defender_reward.item(), attacker_reward.item() if is_self_play_round else "N/A"
            ])
            
            progress_bar.set_description(f"Step {step} | Defender Reward: {defender_reward.item():.2f}")

    print(f"\nTraining complete. Log saved to {log_filepath}")
    
    os.makedirs("models", exist_ok=True)
    save_path_attacker = f"models/{timestamp}_{model_name_safe}_attacker"
    save_path_defender = f"models/{timestamp}_{model_name_safe}_defender"
    
    attacker_model.save_pretrained(save_path_attacker)
    attacker_tokenizer.save_pretrained(save_path_attacker)
    
    defender_model.save_pretrained(save_path_defender)
    defender_tokenizer.save_pretrained(save_path_defender)
    
    print(f"Attacker model saved to {save_path_attacker}")
    print(f"Defender model saved to {save_path_defender}")

if __name__ == "__main__":
    main()