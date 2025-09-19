import os
import re
import argparse
from datetime import datetime

import torch
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer


def get_device():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
        return torch.device("mps")
    print("No GPU available. Using CPU.")
    return torch.device("cpu")


def load_causal_lm(model_id: str, device: torch.device):
    print(f"Loading model: {model_id} to device: {device}")
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.eos_token_id
    return model, tok


def get_rubric_based_reward(original_note: str, corrected_note: str, judge_model, judge_tokenizer, device) -> float:
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
""".strip()
    with torch.no_grad():
        inputs = judge_tokenizer(prompt, return_tensors="pt").to(device)
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id,
        )
        text = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        score_str = text.split("Score: [/INST]")[1].strip()
        score = float(re.search(r"(-?\d+\.?\d*)", score_str).group(1))
    except Exception:
        score = 0.0
    return float(score)


def load_medec_subset(num_samples: int) -> Dataset:
    print("Loading MEDEC training data from local CSV...")
    path = "data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
    df = pd.read_csv(path)
    df.dropna(subset=["Text", "Error Flag"], inplace=True)
    df["Corrected Text"] = df["Corrected Text"].fillna("")
    ds = Dataset.from_pandas(df)
    return ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))


def build_datasets(base_ds: Dataset):
    # Attacker: train on clean notes (Error Flag == 0), produce attacked note
    ds_attacker = base_ds.filter(lambda x: int(x["Error Flag"]) == 0)
    def to_attack_prompt(row):
        original = row.get("Text", "") or ""
        return {
            "prompt": (
                "You are an AI red-teaming assistant. Introduce a subtle but clinically significant error into the note.\n"
                "Provide only the modified note under '### Note with Error:' and nothing else.\n\n"
                "Original Correct Note:\n"
                f"{original}\n\n"
                "### Note with Error:\n"
            )
        }
    ds_attacker = ds_attacker.map(to_attack_prompt, remove_columns=[c for c in ds_attacker.column_names if c != "prompt"])

    # Defender: train on erroneous notes (Error Flag != 0), produce corrected note
    ds_defender = base_ds.filter(lambda x: int(x["Error Flag"]) != 0)
    def to_defend_prompt(row):
        attacked = row.get("Text", "") or ""
        original = row.get("Corrected Text", "") or ""
        return {
            "prompt": (
                "You are a medical assistant. Review the medical note for errors and provide a corrected version.\n"
                "Provide only the corrected note under '### Corrected Note:' and nothing else.\n\n"
                "Original Correct Note:\n"
                f"{original}\n\n"
                "Medical Note:\n"
                f"{attacked}\n\n"
                "### Corrected Note:\n"
            )
        }
    ds_defender = ds_defender.map(to_defend_prompt, remove_columns=[c for c in ds_defender.column_names if c != "prompt"])

    return ds_attacker, ds_defender


def extract_original_from_prompt(prompt: str) -> str:
    m = re.search(r"Original Correct Note:\n(.*?)\n\n", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_attacked_from_prompt(prompt: str) -> str:
    # Used when defender trains; the prompt includes Medical Note section
    m = re.search(r"Medical Note:\n(.*?)\n\n### Corrected Note:", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def main():
    parser = argparse.ArgumentParser(description="Adversarial self-play with GRPO: attacker and defender with rubric judge.")
    parser.add_argument("--model_id", type=str, required=True, help="Base model ID for attacker and defender policies.")
    parser.add_argument("--judge_model_id", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Judge model ID.")
    parser.add_argument("--num_samples", type=int, default=200, help="Dataset subset size.")
    parser.add_argument("--num_generations", type=int, default=2, help="Completions per prompt for GRPO.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    args = parser.parse_args()

    device = get_device()

    # Load policies and judge
    attacker_model, attacker_tok = load_causal_lm(args.model_id, device)
    defender_model, defender_tok = load_causal_lm(args.model_id, device)
    judge_model, judge_tok = load_causal_lm(args.judge_model_id, device)

    # Prepare datasets
    base_ds = load_medec_subset(args.num_samples)
    ds_attacker, ds_defender = build_datasets(base_ds)
    if len(ds_attacker) == 0:
        print("Warning: attacker dataset empty (no Error Flag == 0 rows). Attacker will be skipped.")
    if len(ds_defender) == 0:
        print("Warning: defender dataset empty (no Error Flag != 0 rows). Defender will be skipped.")

    # Defender reward: judge(original, defender_correction)
    def defender_reward_fn(prompts, completions, **kwargs):
        if not prompts or not completions:
            return []
        # Align prompts with completions for num_generations > 1
        if len(completions) % len(prompts) == 0:
            repeat = len(completions) // len(prompts)
            aligned_prompts = [p for p in prompts for _ in range(repeat)]
        else:
            aligned_prompts = prompts[: len(completions)]
        scores = []
        for p, c in zip(aligned_prompts, completions):
            original = extract_original_from_prompt(p)
            corrected = c.strip()
            score = get_rubric_based_reward(original, corrected, judge_model, judge_tok, device)
            scores.append(float(score))
        return scores

    # Attacker reward: -judge(original, defender(current_policy, attacked))
    def attacker_reward_fn(prompts, completions, **kwargs):
        if not prompts or not completions:
            return []
        # Align prompts with completions for num_generations > 1
        if len(completions) % len(prompts) == 0:
            repeat = len(completions) // len(prompts)
            aligned_prompts = [p for p in prompts for _ in range(repeat)]
        else:
            aligned_prompts = prompts[: len(completions)]
        scores = []
        for p, attacked_note in zip(aligned_prompts, completions):
            original = extract_original_from_prompt(p)
            attacked = attacked_note.strip()
            # Let current defender try to fix it, then judge the fix
            defend_prompt = (
                "You are a medical assistant. Review the medical note for errors and provide a corrected version.\n"
                "Provide only the corrected note under '### Corrected Note:' and nothing else.\n\n"
                f"Original Correct Note:\n{original}\n\n"
                f"Medical Note:\n{attacked}\n\n"
                "### Corrected Note:\n"
            )
            with torch.no_grad():
                inputs = defender_tok(defend_prompt, return_tensors="pt").to(device)
                out = defender_model.generate(
                    **inputs,
                    max_new_tokens=350,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=defender_tok.eos_token_id,
                )
                correction = defender_tok.decode(out[0], skip_special_tokens=True)
            score = get_rubric_based_reward(original, correction, judge_model, judge_tok, device)
            scores.append(float(-score))  # attacker aims to make defender fail
        return scores

    # Configs
    common_cfg = dict(
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,  # TRL 0.23.0 requires divisible by num_generations
        max_prompt_length=1024,
        max_completion_length=350,
        temperature=0.7,
        top_p=0.95,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=10,
        save_steps=0,
        report_to=None,
    )

    attacker_config = GRPOConfig(**common_cfg)
    defender_config = GRPOConfig(**common_cfg)

    # Trainers (require TRL with GRPO)
    if len(ds_attacker) > 0:
        attacker_trainer = GRPOTrainer(
            model=attacker_model,
            config=attacker_config,
            processing_class=attacker_tok,
            train_dataset=ds_attacker,
            reward_funcs=[attacker_reward_fn],
        )
    else:
        attacker_trainer = None

    if len(ds_defender) > 0:
        defender_trainer = GRPOTrainer(
            model=defender_model,
            config=defender_config,
            processing_class=defender_tok,
            train_dataset=ds_defender,
            reward_funcs=[defender_reward_fn],
        )
    else:
        defender_trainer = None

    # Train sequentially (simple, avoids interleaving complexity)
    if attacker_trainer:
        print("\n--- Training attacker with GRPO ---")
        attacker_trainer.train()
    if defender_trainer:
        print("\n--- Training defender with GRPO ---")
        defender_trainer.train()

    # Save models
    os.makedirs("models", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.model_id.replace("/", "_")
    if attacker_trainer:
        save_a = f"models/{ts}_{name}_attacker_grpo"
        attacker_model.save_pretrained(save_a)
        attacker_tok.save_pretrained(save_a)
        print(f"Attacker model saved to {save_a}")
    if defender_trainer:
        save_d = f"models/{ts}_{name}_defender_grpo"
        defender_model.save_pretrained(save_d)
        defender_tok.save_pretrained(save_d)
        print(f"Defender model saved to {save_d}")


if __name__ == "__main__":
    main()