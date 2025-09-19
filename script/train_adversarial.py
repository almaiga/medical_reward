import os
import re
import json
import argparse
from datetime import datetime

import torch
import pandas as pd
from datasets import Dataset
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
        # If you want chain-of-thought, include an explicit <think> section in your instruction.
        return {
            "prompt": (
                "You are an AI red-teaming assistant. Introduce a subtle but clinically significant error into the note.\n"
                "Provide only the modified note under '### Note with Error:' and nothing else.\n\n"
                "Original Correct Note:\n"
                f"{original}\n\n"
                "### Note with Error:\n"
            )
        }

    ds_attacker = ds_attacker.map(
        to_attack_prompt, remove_columns=[c for c in ds_attacker.column_names if c != "prompt"]
    )

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

    ds_defender = ds_defender.map(
        to_defend_prompt, remove_columns=[c for c in ds_defender.column_names if c != "prompt"]
    )

    return ds_attacker, ds_defender


def extract_original_from_prompt(prompt: str) -> str:
    m = re.search(r"Original Correct Note:\n(.*?)\n\n", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_attacked_from_prompt(prompt: str) -> str:
    m = re.search(r"Medical Note:\n(.*?)\n\n### Corrected Note:", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def main():
    parser = argparse.ArgumentParser(description="Adversarial self-play with GRPO: single shared policy, rubric judge.")
    parser.add_argument("--model_id", type=str, required=True, help="Base model ID for policy (used as both attacker/defender).")
    parser.add_argument("--judge_model_id", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Judge model ID.")
    parser.add_argument("--num_samples", type=int, default=200, help="Dataset subset size.")
    parser.add_argument("--num_generations", type=int, default=2, help="Completions per prompt for GRPO.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--rounds", type=int, default=6, help="Number of self-play rounds (1 epoch per round).")
    args = parser.parse_args()

    device = get_device()

    # Shared policy and judge
    policy_model, policy_tok = load_causal_lm(args.model_id, device)
    judge_model, judge_tok = load_causal_lm(args.judge_model_id, device)

    # Make sampling effective via generation_config to avoid ignored kwargs warnings
    gc = policy_model.generation_config
    gc.do_sample = True
    gc.temperature = 0.7
    gc.top_p = 0.95
    gc.max_new_tokens = 350
    gc.pad_token_id = policy_tok.eos_token_id
    gc.eos_token_id = policy_tok.eos_token_id

    # Prepare datasets
    base_ds = load_medec_subset(args.num_samples)
    ds_attacker, ds_defender = build_datasets(base_ds)
    if len(ds_attacker) == 0:
        print("Warning: attacker dataset empty (no Error Flag == 0 rows).")
    if len(ds_defender) == 0:
        print("Warning: defender dataset empty (no Error Flag != 0 rows).")

    # JSONL logging
    os.makedirs("results/selfplay", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_id.replace("/", "_")
    log_path = f"results/selfplay/{ts}_{model_name}_grpo_selfplay.jsonl"

    round_idx_holder = {"round": 0}
    atk_step = {"i": 0}
    def_step = {"i": 0}

    def log_jsonl(entry: dict):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Reward functions (aligned with num_generations in TRL 0.23.0)
    def defender_reward_fn(prompts, completions, **kwargs):
        if not prompts or not completions:
            return []
        # Align prompts with completions
        if len(completions) % len(prompts) == 0:
            repeat = len(completions) // len(prompts)
            aligned_prompts = [p for p in prompts for _ in range(repeat)]
        else:
            aligned_prompts = prompts[: len(completions)]

        scores = []
        for p, c in zip(aligned_prompts, completions):
            original = extract_original_from_prompt(p)
            attacked = extract_attacked_from_prompt(p)
            corrected = c.strip()
            score = get_rubric_based_reward(original, corrected, judge_model, judge_tok, device)
            scores.append(float(score))
            # Log
            log_jsonl(
                {
                    "round": round_idx_holder["round"],
                    "role": "defender",
                    "step": def_step["i"],
                    "prompt": p,
                    "original_note": original,
                    "attacked_note": attacked,
                    "defender_completion": corrected,
                    "judge_score": score,
                }
            )
            def_step["i"] += 1
        return scores

    def attacker_reward_fn(prompts, completions, **kwargs):
        if not prompts or not completions:
            return []
        # Align prompts with completions
        if len(completions) % len(prompts) == 0:
            repeat = len(completions) // len(prompts)
            aligned_prompts = [p for p in prompts for _ in range(repeat)]
        else:
            aligned_prompts = prompts[: len(completions)]

        scores = []
        for p, attacked_note in zip(aligned_prompts, completions):
            original = extract_original_from_prompt(p)
            attacked = attacked_note.strip()

            # Let current shared policy act as defender
            defend_prompt = (
                "You are a medical assistant. Review the medical note for errors and provide a corrected version.\n"
                "Provide only the corrected note under '### Corrected Note:' and nothing else.\n\n"
                f"Original Correct Note:\n{original}\n\n"
                f"Medical Note:\n{attacked}\n\n"
                "### Corrected Note:\n"
            )
            with torch.no_grad():
                inputs = policy_tok(defend_prompt, return_tensors="pt").to(device)
                out = policy_model.generate(
                    **inputs,
                    max_new_tokens=350,
                    do_sample=True,
                    pad_token_id=policy_tok.eos_token_id,
                )
                defender_completion = policy_tok.decode(out[0], skip_special_tokens=True).strip()

            score = get_rubric_based_reward(original, defender_completion, judge_model, judge_tok, device)
            scores.append(float(-score))  # attacker is rewarded when defender fails

            # Log
            log_jsonl(
                {
                    "round": round_idx_holder["round"],
                    "role": "attacker",
                    "step": atk_step["i"],
                    "prompt": p,
                    "original_note": original,
                    "attacked_note": attacked,
                    "defender_completion": defender_completion,
                    "judge_score": score,
                    "attacker_reward": -score,
                }
            )
            atk_step["i"] += 1
        return scores

    # Configs for TRL 0.23.0 (no stop_token_ids; align batch sizes)
    common_cfg = dict(
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,  # must divide evenly by num_generations
        max_prompt_length=1024,
        max_completion_length=350,
        # Sampling handled via model.generation_config
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=10,
        save_steps=0,
        report_to=None,
        dataloader_pin_memory=False,  # avoid MPS warning
        gradient_checkpointing=False,
        num_train_epochs=1,  # one epoch per round
    )
    atk_cfg = GRPOConfig(**common_cfg)
    def_cfg = GRPOConfig(**common_cfg)

    # Interleaved self-play rounds; both trainers share the same policy model
    for r in range(args.rounds):
        round_idx_holder["round"] = r + 1
        print(f"\n=== Self-play round {r+1}/{args.rounds} ===")

        # Attacker epoch
        if len(ds_attacker) > 0:
            print("--- Attacker epoch ---")
            attacker_trainer = GRPOTrainer(
                model=policy_model,
                args=atk_cfg,
                processing_class=policy_tok,
                train_dataset=ds_attacker,
                reward_funcs=[attacker_reward_fn],
            )
            attacker_trainer.train()

        # Defender epoch
        if len(ds_defender) > 0:
            print("--- Defender epoch ---")
            defender_trainer = GRPOTrainer(
                model=policy_model,
                args=def_cfg,
                processing_class=policy_tok,
                train_dataset=ds_defender,
                reward_funcs=[defender_reward_fn],
            )
            defender_trainer.train()

    # Save the single shared policy
    os.makedirs("models", exist_ok=True)
    save_dir = f"models/{ts}_{model_name}_shared_grpo"
    policy_model.save_pretrained(save_dir)
    policy_tok.save_pretrained(save_dir)
    print(f"Shared self-play policy saved to {save_dir}")
    print(f"JSONL log written to {log_path}")


if __name__ == "__main__":
    main()