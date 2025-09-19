import torch
import argparse
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm

# --- Configuration ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def load_training_model(model_id):
    """Loads the model to be trained, including the value head for PPO."""
    print(f"Loading model for training: {model_id}...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        #quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Training model and tokenizer loaded.")
    return model, tokenizer

def get_rubric_based_reward(original_note: str, corrected_note: str, judge_model, judge_tokenizer) -> torch.Tensor:
    """
    Uses a judge model to score the corrected note based on a rubric.
    """
    judge_prompt = f"""You are a medical expert evaluating an AI's attempt to correct a medical note.
                    Compare the 'Corrected Note' to the 'Original Correct Note'.
                    Score the 'Corrected Note' on a scale of 1 to 10 for each of the following metrics based on how well it matches the original's clinical accuracy and intent.

                    1. **Accuracy (1-10):** Includes only factually correct information supported by evidence.
                    2. **Completeness (1-10):** Addresses all necessary clinical points from the original note.
                    3. **Context Awareness (1-10):** The correction is appropriate given the clinical context.

                    Respond ONLY with a JSON object containing the scores, like this:
                    {{"accuracy": <score>, "completeness": <score>, "context_awareness": <score>}}

                    ### Original Correct Note:
                    {original_note}

                    ### Corrected Note:
                    {corrected_note}

                    ### Evaluation:
                    """
    
    inputs = judge_tokenizer(judge_prompt, return_tensors="pt").to(judge_model.device)
    outputs = judge_model.generate(**inputs, max_new_tokens=100, pad_token_id=judge_tokenizer.eos_token_id)
    response_text = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract scores from the judge's response
    try:
        # Use regex to find all numbers in the response, which is more robust
        scores = [int(s) for s in re.findall(r'\d+', response_text)]
        if len(scores) >= 3:
            # Average the scores and normalize to a -1 to 1 range
            avg_score = sum(scores[:3]) / 3.0
            reward = (avg_score - 5.5) / 4.5  # Maps 1-10 scale to approx -1 to 1
        else:
            reward = -1.0 # Penalize if the output format is wrong
    except (ValueError, IndexError):
        reward = -1.0 # Penalize if parsing fails

    return torch.tensor(reward)

def main():
    parser = argparse.ArgumentParser(description="Train a medical LLM with self-play.")
    parser.add_argument("--model_id", type=str, required=True, help="Your ablated model ID.")
    parser.add_argument("--judge_model_id", type=str, default="Intelligent-Internet/II-Medical-8B", help="The model ID for the judge.")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of self-play steps.")
    args = parser.parse_args()

    ppo_config = PPOConfig(batch_size=1, learning_rate=1.41e-5, mini_batch_size=1)

    model, tokenizer = load_training_model(args.model_id)
    print("Loading judge model...")
    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model_id, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token

    dataset = load_dataset("abachaa/MEDIQA-CORR-2024", split="train").shuffle(seed=42).select(range(args.num_steps))
    ppo_trainer = PPOTrainer(config=ppo_config, model=model, ref_model=None, tokenizer=tokenizer)

    for step, row in tqdm(enumerate(dataset), total=args.num_steps):
        # *** CRITICAL REFINEMENT: Start the loop with the ground-truth correct note ***
        original_note = row['correct_note']
        if not original_note: # Skip if the correct note is empty
            continue

        # === 1. Attacker Role: Corrupt the correct note ===
        attack_prompt = f"""You are an AI red-teaming assistant. Your task is to introduce a subtle but clinically significant error into the following medical note.

### Original Note:
{original_note}

### Note with Error:
"""
        attack_input_ids = tokenizer.encode(attack_prompt, return_tensors='pt').to(model.device)
        attack_output = model.generate(
            attack_input_ids,
            max_new_tokens=len(attack_input_ids[0]) + 50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        attacked_note_text = tokenizer.decode(attack_output[0], skip_special_tokens=True).split("### Note with Error:")[-1].strip()
        
        # === 2. Defender Role: Correct the generated error ===
        defend_prompt = f"""You are a medical assistant. Please review the following medical note for errors and provide the corrected version.

### Medical Note:
{attacked_note_text}

### Corrected Note:
"""
        query_tensor = tokenizer.encode(defend_prompt, return_tensors='pt').to(model.device)
        
        response_tensor = ppo_trainer.generate(query_tensor[0], return_prompt=False, max_new_tokens=256)
        corrected_note_text = tokenizer.decode(response_tensor, skip_special_tokens=True).strip()

        # === 3. Judge Role: Get reward ===
        reward = get_rubric_based_reward(original_note, corrected_note_text, judge_model, judge_tokenizer)
        rewards = [reward]

        # === 4. PPO Step: Update the model ===
        # Ensure tensors are on the same device
        query_tensor = query_tensor.to(model.device)
        response_tensor = response_tensor.to(model.device)
        
        stats = ppo_trainer.step([query_tensor[0]], [response_tensor], rewards)
        
        if step % 10 == 0:
            print(f"\n--- Round {step} ---")
            print(f"ATTACKER OUTPUT:\n{attacked_note_text}\n")
            print(f"DEFENDER OUTPUT:\n{corrected_note_text}\n")
            print(f"REWARD SCORE: {rewards[0].item():.2f}")
            if 'ppo/objective/kl' in stats:
                print(f"PPO KL Divergence: {stats['ppo/objective/kl']:.2f}")

    print("Training complete. Saving model...")
    # --- Generate dynamic filename ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model_id.replace("/", "_")
    judge_name_safe = args.judge_model_id.replace("/", "_")
    
    save_path = f"{timestamp}_{model_name_safe}_judged_by_{judge_name_safe}_{args.num_steps}_steps"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()