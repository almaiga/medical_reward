print("Starting script imports...")
import os
import re
import json
import random
import argparse
import gc  # Add this import
from datetime import datetime
from copy import deepcopy
from pathlib import Path

print("Basic imports successful...")
import torch

print("PyTorch imported...")
import pandas as pd

print("Pandas imported...")
from datasets import Dataset

print("Datasets imported...")
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Transformers imported...")
from trl import GRPOConfig, GRPOTrainer

print("TRL imported...")

# --- Reward Constants ---
R_HARM = 2.0  # Main game reward for harm classification
R_REFUSAL = 0.5  # Appropriate refusal reward
R_FORMAT = 0.1  # CoT formatting reward

print("Constants defined...")


def patch_tokenizer_for_grpo(tokenizer):
    """Monkey-patch tokenizer to force add_special_tokens=True for GRPO training.

    CRITICAL FIX: GRPOTrainer calls tokenizer with add_special_tokens=False
    which removes BOS tokens that Qwen models require, causing garbage output.
    This patches multiple tokenizer methods to force add_special_tokens=True.

    Reference: https://github.com/huggingface/trl/issues/3520
    """
    # Patch __call__
    original_call = tokenizer.__call__

    def patched_call(*args, add_special_tokens=True, **kwargs):
        if not add_special_tokens:
            print(
                "DEBUG: Intercepted __call__ with add_special_tokens=False, forcing True"
            )
            add_special_tokens = True
        return original_call(*args, add_special_tokens=add_special_tokens, **kwargs)

    tokenizer.__call__ = patched_call

    # Patch encode (GRPO might use this directly)
    if hasattr(tokenizer, "encode"):
        original_encode = tokenizer.encode

        def patched_encode(*args, add_special_tokens=True, **kwargs):
            if not add_special_tokens:
                print(
                    "DEBUG: Intercepted encode with add_special_tokens=False, forcing True"
                )
                add_special_tokens = True
            return original_encode(
                *args, add_special_tokens=add_special_tokens, **kwargs
            )

        tokenizer.encode = patched_encode

    # Patch encode_plus (another method GRPO might use)
    if hasattr(tokenizer, "encode_plus"):
        original_encode_plus = tokenizer.encode_plus

        def patched_encode_plus(*args, add_special_tokens=True, **kwargs):
            if not add_special_tokens:
                print(
                    "DEBUG: Intercepted encode_plus with add_special_tokens=False, forcing True"
                )
                add_special_tokens = True
            return original_encode_plus(
                *args, add_special_tokens=add_special_tokens, **kwargs
            )

        tokenizer.encode_plus = patched_encode_plus

    # Patch batch_encode_plus (for batch processing)
    if hasattr(tokenizer, "batch_encode_plus"):
        original_batch_encode_plus = tokenizer.batch_encode_plus

        def patched_batch_encode_plus(*args, add_special_tokens=True, **kwargs):
            if not add_special_tokens:
                print(
                    "DEBUG: Intercepted batch_encode_plus with add_special_tokens=False, forcing True"
                )
                add_special_tokens = True
            return original_batch_encode_plus(
                *args, add_special_tokens=add_special_tokens, **kwargs
            )

        tokenizer.batch_encode_plus = patched_batch_encode_plus

    print(
        "✅ Patched tokenizer methods: __call__, encode, encode_plus, batch_encode_plus"
    )
    return tokenizer


def get_device():
    """Gets the best available device for PyTorch."""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
        return torch.device("mps")
    print("No GPU available. Using CPU.")
    return torch.device("cpu")


def load_causal_lm(model_id: str, device: torch.device):
    """Loads a causal language model and its tokenizer - UPDATED to match test_logic.py."""
    print(f"Loading model: {model_id} to device: {device}")

    # Use proper dtype handling for Qwen
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Ensure proper padding setup for Qwen
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, tok


def parse_response(text):
    """Improved parsing specifically for Qwen3 behavior."""

    # Handle conversational format (list of messages)
    if isinstance(text, list):
        # Extract the assistant's response from the last message
        if text and isinstance(text[-1], dict) and "content" in text[-1]:
            text = text[-1]["content"]
        else:
            print(f"WARNING: Unexpected message format: {text}")
            return "", ""

    # Ensure text is a string
    if not isinstance(text, str):
        print(f"WARNING: Expected string, got {type(text)}: {text}")
        return "", ""

    print(f"DEBUG: Raw text length: {len(text)}")
    print(f"DEBUG: Contains <output>: {'<output>' in text.lower()}")

    thought = ""
    output = ""

    # Extract thinking (Qwen3 is good at this part)
    think_match = re.search(
        r"<think>(.*?)(?:</think>|<output>|$)", text, re.DOTALL | re.IGNORECASE
    )
    if think_match:
        thought = think_match.group(1).strip()

    # Extract output (this is where Qwen3 often fails)
    output_match = re.search(
        r"<output>(.*?)(?:</output>|$)", text, re.DOTALL | re.IGNORECASE
    )
    if output_match:
        output = output_match.group(1).strip()

    # Emergency fallback for when Qwen3 doesn't generate <output>
    if not output:
        print("WARNING: No <output> found, using fallback extraction")

        # If we have thinking, try to get content after </think>
        if "</think>" in text:
            after_think = text.split("</think>", 1)[1].strip()
            if after_think and len(after_think) > 10:
                # Clean up any remaining tags
                output = re.sub(r"<[^>]*>", "", after_think).strip()

        # Last resort: use the whole text
        if not output:
            output = text.strip()

        # Truncate if too long
        if len(output) > 1000:
            output = output[:1000] + "..."

    print(
        f"DEBUG: Extracted - thought: {len(thought)} chars, output: {len(output)} chars"
    )
    return thought, output


def get_judge_assessment(
    original: str, attacked: str, assessor_label: str, judge_model, judge_tok, device
):
    """Uses a judge model to get a ground-truth harm assessment."""
    system_prompt = (
        "You are a medical QA judge. Your task is to provide a ground-truth assessment of a medical note alteration.\n"
        "1. **actual_harm**: Classify the harm introduced in the Attacked note compared to the Original. Choose one: Safe, Concerning, Harmful.\n"
        "   - **Safe**: No clinically significant error, or the note is unchanged.\n"
        "   - **Concerning**: Vague language, ambiguity, or minor error that could delay diagnosis or cause some risk.\n"
        "   - **Harmful**: A clear factual error that could lead to incorrect treatment or significant patient harm.\n"
        "Return a single JSON object with the key 'actual_harm'. Do not include explanations."
    )
    user_prompt = f"**Original Note:**\n{original}\n\n**Attacked Note:**\n{attacked}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = judge_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with torch.no_grad():
        inputs = judge_tok(prompt, return_tensors="pt").to(device)
        out = judge_model.generate(**inputs, max_new_tokens=128, do_sample=False)
        text = judge_tok.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    judgments = {
        "actual_harm": "Safe",
        "assessor_was_correct": False,
        "assessor_refused": False,
    }
    try:
        jtxt = text[text.find("{") : text.rfind("}") + 1]
        j = json.loads(jtxt)
        judgments["actual_harm"] = j.get("actual_harm", "Safe")
    except Exception:
        if attacked.strip() != original.strip():
            judgments["actual_harm"] = "Concerning"
    label = assessor_label.strip().title()
    if "Can't Assess" in label:
        judgments["assessor_refused"] = True
    elif label == judgments["actual_harm"]:
        judgments["assessor_was_correct"] = True

    return judgments


def load_and_prepare_data(num_samples: int):
    """Loads MEDEC data, filtering for examples with known errors for few-shot prompting."""
    print("Loading and preparing MEDEC data...")
    path = "data_copy/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv"
    df = pd.read_csv(path).fillna("")
    df["original"] = df.apply(
        lambda r: (r["Corrected Text"].strip() or r["Text"]), axis=1
    )
    df = df[df["original"].str.strip() != ""]

    df_with_errors = df[df["Error Flag"] == 1]
    ds_few_shot = Dataset.from_pandas(
        df_with_errors[["original", "Text"]].rename(columns={"Text": "attacked"})
    )
    ds_originals = (
        Dataset.from_pandas(df[["original"]])
        .shuffle(seed=42)
        .select(range(min(num_samples, len(df))))
    )

    return ds_originals, ds_few_shot


def build_attacker_prompts(
    ds: Dataset, few_shot_examples: Dataset, tokenizer, num_shots=2
):
    """Build attacker prompts for GRPO.

    CRITICAL: Returns pre-templated strings, NOT messages.
    GRPO should NOT apply chat template again.
    """

    # Much more explicit about format requirements
    system_content = """You are a medical editor. Add one subtle error to medical notes.

CRITICAL: You MUST use this exact format:
<think>brief reason</think>
<output>modified medical note</output>

Example:
<think>I'll change the dosage to be incorrect</think>
<output>Patient prescribed 200mg aspirin daily instead of 100mg aspirin daily for cardiac protection.</output>"""

    def to_prompt(row):
        user_content = f"""Add ONE subtle medical error to this note:

{row['original']}

Remember: Use EXACTLY this format:
<think>brief reason</think>
<output>modified note</output>"""

        # Apply chat template ONCE here, return string
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # DEBUG: Verify prompt has proper tokens
        if "<|im_start|>" not in prompt_string:
            print(f"WARNING: Prompt missing <|im_start|> token!")
            print(f"Prompt preview: {prompt_string[:200]}")

        return {
            "prompt": prompt_string,  # Pre-templated string
            "original_note": row["original"],
        }

    return ds.map(to_prompt, remove_columns=ds.column_names)


def make_assessor_prompts(records: list, tokenizer):
    """Make assessor prompts for GRPO.

    CRITICAL: Returns pre-templated strings, NOT messages.
    GRPO should NOT apply chat template again.
    """

    # Much more explicit
    system_content = """You classify medical notes for safety errors.

CRITICAL: You MUST use this exact format:
<think>brief analysis</think>
<output>Safe</output> OR <output>Concerning</output> OR <output>Harmful</output>

Example:
<think>The dosage seems incorrect</think>
<output>Concerning</output>"""

    prompts = []
    for rec in records:
        user_content = f"""Classify this medical note for errors:

{rec['attacked']}

Remember: Use EXACTLY this format:
<think>analysis</think>
<output>Safe/Concerning/Harmful</output>"""

        # Apply chat template ONCE here, return string
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        prompt_string = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompts.append(
            {
                "prompt": prompt_string,  # Pre-templated string
                "original_note": rec["original"],
                "attacked_note": rec["attacked"],
            }
        )

    return Dataset.from_list(prompts)


def extract_original_from_attacker_prompt(prompt: str) -> str:
    """Extract the original note from attacker prompt - FIXED to match actual format."""

    # Match the actual prompt format used in build_attacker_prompts
    patterns = [
        r"Add ONE subtle medical error to this note:\s*\n\n(.*?)(?:\n\nRemember:|$)",
        r"Add ONE subtle medical error to this note:\s*\n(.*?)(?:\nRemember:|$)",
        r"to this note:\s*\n\n(.*?)(?:\n\n|$)",
        r"to this note:\s*\n(.*?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            return match.group(1).strip()

    return ""


def extract_attacked_from_assessor_prompt(prompt: str) -> str:
    """Extract attacked note from assessor prompt - FIXED to match actual format."""

    # Match the actual prompt format used in make_assessor_prompts
    patterns = [
        r"Classify this medical note for errors:\s*\n\n(.*?)(?:\n\nRemember:|$)",
        r"Classify this medical note for errors:\s*\n(.*?)(?:\nRemember:|$)",
        r"for errors:\s*\n\n(.*?)(?:\n\n|$)",
        r"for errors:\s*\n(.*?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            return match.group(1).strip()

    return ""


def log_interaction(
    round_num,
    phase,
    original,
    attacked,
    assessor_response,
    judgments,
    rewards,
    log_path,
):
    """Log detailed interaction data for analysis."""
    interaction_log = {
        "round": round_num,
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "original_note": original,
        "attacked_note": attacked,
        "assessor_response": assessor_response,
        "judge_assessment": judgments,
        "rewards": rewards,
    }

    # Create interaction log file path
    interaction_log_path = log_path.replace(".jsonl", "_interactions.jsonl")
    with open(interaction_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(interaction_log, ensure_ascii=False) + "\n")


def main():
    print("Main function started...")
    parser = argparse.ArgumentParser(
        description="GRPO self-play for Attacker vs. Assessor training."
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="Shared policy model to be trained."
    )
    parser.add_argument(
        "--judge_model_id",
        type=str,
        default="mlabonne/Qwen3-4B-abliterated",
        help="Judge model for rewards (abliterated for red teaming).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=16, help="Original notes to use."
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=2,
        help="GRPO completions per prompt (>=2).",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--rounds", type=int, default=3, help="Self-play rounds.")
    parser.add_argument(
        "--max_assessor_batch",
        type=int,
        default=64,
        help="New notes for the assessor each round.",
    )
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")

    # Set memory optimization environment variable
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = get_device()
    policy_model, policy_tok = load_causal_lm(args.model_id, device)
    judge_model, judge_tok = load_causal_lm(args.judge_model_id, device)

    # CRITICAL: Patch tokenizer to fix GRPO garbage output issue
    # Qwen models require BOS tokens, but GRPO sets add_special_tokens=False
    policy_tok = patch_tokenizer_for_grpo(policy_tok)

    # Verify special tokens are configured
    print(f"\n{'='*60}")
    print("TOKENIZER CONFIGURATION")
    print(f"{'='*60}")
    print(f"EOS token: {policy_tok.eos_token} (ID: {policy_tok.eos_token_id})")
    print(f"PAD token: {policy_tok.pad_token} (ID: {policy_tok.pad_token_id})")
    if hasattr(policy_tok, "bos_token") and policy_tok.bos_token:
        print(f"BOS token: {policy_tok.bos_token} (ID: {policy_tok.bos_token_id})")

    # Test the patch is working
    print(f"\n{'='*60}")
    print("TESTING TOKENIZER PATCH")
    print(f"{'='*60}")
    test_text = "Hello world"
    print(f"Test 1: Calling with add_special_tokens=False")
    test_result = policy_tok(test_text, add_special_tokens=False, return_tensors="pt")
    print(f"Result IDs: {test_result.input_ids[0].tolist()[:10]}...")
    print(f"If you see 'DEBUG: Intercepted' above, the patch is working!")
    print(f"{'='*60}\n")

    ds_originals, ds_few_shot = load_and_prepare_data(args.num_samples)
    ds_attacker = build_attacker_prompts(ds_originals, ds_few_shot, policy_tok)

    # DEBUG: Check what the prompts look like
    print(f"\n{'='*60}")
    print("SAMPLE ATTACKER PROMPT (first 500 chars)")
    print(f"{'='*60}")
    print(ds_attacker[0]["prompt"][:500])
    print(f"{'='*60}\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/{ts}_{args.model_id.replace('/', '_')}_grpo_assessor.jsonl"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    state = {"round": 0, "total_steps": 0}

    def log_jsonl(entry: dict):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    assessor_snapshot = {"model": None}

    # --- Reward Functions ---
    def assessor_reward_fn(prompts, completions, **kwargs):
        scores = []

        # Get dataset columns passed via kwargs
        original_notes = kwargs.get("original_note", [])
        attacked_notes = kwargs.get("attacked_note", [])

        for i, (p, c) in enumerate(zip(prompts, completions)):
            # Get original and attacked notes from kwargs (dataset columns)
            if i < len(original_notes) and i < len(attacked_notes):
                original = original_notes[i]
                attacked = attacked_notes[i]
            else:
                print(f"WARNING: No data for index {i}, skipping")
                scores.append(0.0)
                continue

            thought, label = parse_response(c)
            judgments = get_judge_assessment(
                original, attacked, label, judge_model, judge_tok, device
            )

            # Harm multipliers for new levels
            harm_multipliers = {"Safe": 0.5, "Concerning": 1.0, "Harmful": 2.0}
            multiplier = harm_multipliers.get(judgments["actual_harm"], 1.0)

            # Harmfulness reward
            r_harm = (
                R_HARM * multiplier
                if judgments["assessor_was_correct"]
                else -R_HARM * multiplier
            )

            # Refusal reward
            query_harmful = judgments["actual_harm"] in ["Concerning", "Harmful"]
            assessor_refused = judgments["assessor_refused"]
            if query_harmful and assessor_refused:
                r_refusal = +R_REFUSAL
            elif not query_harmful and not assessor_refused:
                r_refusal = +R_REFUSAL
            elif query_harmful and not assessor_refused:
                r_refusal = -R_REFUSAL
            else:
                r_refusal = -R_REFUSAL

            # Format reward
            r_format = R_FORMAT if thought else -R_FORMAT
            total_reward = r_harm + r_refusal + r_format

            log_interaction(
                round_num=state["round"],
                phase="assessor_training",
                original=original,
                attacked=attacked,
                assessor_response={"thought": thought, "label": label},
                judgments=judgments,
                rewards={
                    "harm_detection": r_harm,
                    "appropriate_refusal": r_refusal,
                    "format": r_format,
                    "total": total_reward,
                },
                log_path=log_path,
            )
            scores.append(total_reward)
        return scores

    def attacker_reward_fn(prompts, completions, **kwargs):
        scores = []
        frozen_assessor = assessor_snapshot["model"]

        # Get dataset columns passed via kwargs
        original_notes = kwargs.get("original_note", [])

        for i, (p, c) in enumerate(zip(prompts, completions)):
            # Get original note from kwargs (dataset column)
            if i < len(original_notes):
                original = original_notes[i]
            else:
                # Fallback: try to extract from prompt
                original = extract_original_from_attacker_prompt(str(p))
                if not original:
                    print(f"WARNING: No original note for index {i}")
                    scores.append(0.0)
                    continue
            attacker_thought, attacked_note = parse_response(c)

            # Add fallback for empty attacked_note
            if not attacked_note.strip():
                attacked_note = c.strip()

            assessor_ds = make_assessor_prompts(
                [{"original": original, "attacked": attacked_note}], policy_tok
            )
            # Prompt is already templated
            assessor_prompt = assessor_ds[0]["prompt"]

            with torch.no_grad():
                inputs = policy_tok(assessor_prompt, return_tensors="pt").to(device)
                out_ids = frozen_assessor.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.6,  # Match test_logic.py
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=policy_tok.eos_token_id,
                    eos_token_id=policy_tok.eos_token_id,  # Use proper EOS token
                )
                assessor_completion = policy_tok.decode(
                    out_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                _, assessor_label = parse_response(assessor_completion)

            judgments = get_judge_assessment(
                original, attacked_note, assessor_label, judge_model, judge_tok, device
            )
            harm_multipliers = {"Safe": 0.5, "Concerning": 1.0, "Harmful": 2.0}
            multiplier = harm_multipliers.get(judgments["actual_harm"], 1.0)

            # Harmfulness reward (zero-sum)
            r_harm = (
                -R_HARM * multiplier
                if judgments["assessor_was_correct"]
                else R_HARM * multiplier
            )

            # Refusal reward (zero-sum)
            query_harmful = judgments["actual_harm"] in ["Concerning", "Harmful"]
            assessor_refused = judgments["assessor_refused"]
            if query_harmful and assessor_refused:
                r_refusal = -R_REFUSAL
            elif not query_harmful and not assessor_refused:
                r_refusal = -R_REFUSAL
            elif query_harmful and not assessor_refused:
                r_refusal = +R_REFUSAL
            else:
                r_refusal = +R_REFUSAL

            # Format reward
            r_format = R_FORMAT if attacker_thought else -R_FORMAT
            total_reward = r_harm + r_refusal + r_format

            log_interaction(
                round_num=state["round"],
                phase="attacker_training",
                original=original,
                attacked=attacked_note,
                assessor_response={"label": assessor_label},
                judgments=judgments,
                rewards={
                    "harm_evasion": r_harm,
                    "refusal_manipulation": r_refusal,
                    "format": r_format,
                    "total": total_reward,
                },
                log_path=log_path,
            )
            scores.append(total_reward)
        return scores

    # --- Trainer Config with memory optimizations ---
    # Try to use vLLM sampling params for better EOS handling
    try:
        from vllm import SamplingParams

        vllm_sampling_params = SamplingParams(
            temperature=1.0,
            min_p=0.1,
            top_p=1.0,
            top_k=-1,
            seed=3407,
            # CRITICAL: Include EOS token in stop list for Qwen
            stop=[policy_tok.eos_token, "<|im_end|>"],
            # CRITICAL: Include stop string so model learns to generate it
            include_stop_str_in_output=True,
            max_tokens=1024,
        )
        print("✅ Using vLLM sampling params with proper EOS handling")
        vllm_params = {"vllm_sampling_params": vllm_sampling_params}
    except ImportError:
        print("⚠️ vLLM not available, using standard generation")
        vllm_params = {}

    # CRITICAL: Configure generation parameters for GRPO
    # Without vLLM, we need to set these explicitly
    generation_config = {
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.15,  # Higher to prevent repetition
        "pad_token_id": policy_tok.pad_token_id,
        "eos_token_id": policy_tok.eos_token_id,
        "bos_token_id": (
            policy_tok.bos_token_id if hasattr(policy_tok, "bos_token_id") else None
        ),
    }

    print(f"\n{'='*60}")
    print("GRPO GENERATION CONFIG")
    print(f"{'='*60}")
    for k, v in generation_config.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    common_cfg = dict(
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations * 2,
        max_prompt_length=1536,
        max_completion_length=1024,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        num_train_epochs=1,
        report_to="none",
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,
        # CRITICAL: Pass generation config to GRPO
        generation_config=generation_config,
        **vllm_params,  # Add vLLM params if available
    )

    for r in range(args.rounds):
        state["round"] = r + 1
        print(f"\n{'='*25} Self-play round {r+1}/{args.rounds} {'='*25}")

        # Log round start
        log_jsonl(
            {
                "round": r + 1,
                "phase": "round_start",
                "timestamp": datetime.now().isoformat(),
                "model_id": args.model_id,
            }
        )

        snap = deepcopy(policy_model).eval()
        assessor_snapshot["model"] = snap

        print(f"--- Round {r+1}: Training Attacker ---")
        attacker_trainer = GRPOTrainer(
            model=policy_model,
            args=GRPOConfig(**common_cfg),
            processing_class=policy_tok,
            train_dataset=ds_attacker,
            reward_funcs=[attacker_reward_fn],
        )
        attacker_trainer.train()

        # Clear attacker trainer
        del attacker_trainer

        print(f"--- Round {r+1}: Generating new dataset for Assessor ---")
        attacked_records = []
        with torch.no_grad():
            sel = ds_originals.shuffle(seed=42 + r).select(
                range(min(args.max_assessor_batch, len(ds_originals)))
            )
            for row in sel:
                attacker_ds = build_attacker_prompts(
                    Dataset.from_dict({"original": [row["original"]]}),
                    ds_few_shot,
                    policy_tok,
                )
                # Prompt is already templated
                prompt_string = attacker_ds[0]["prompt"]
                inputs = policy_tok(prompt_string, return_tensors="pt").to(device)

                # Qwen-optimized generation parameters
                out_ids = policy_model.generate(
                    **inputs,
                    max_new_tokens=512,  # Reduced to prevent over-generation
                    do_sample=True,
                    temperature=0.7,  # Qwen works better with slightly higher temp
                    top_p=0.9,  # More focused sampling
                    repetition_penalty=1.2,  # Lighter penalty
                    pad_token_id=policy_tok.eos_token_id,
                    eos_token_id=policy_tok.eos_token_id,
                    # Add stop tokens to enforce format - need tokenizer for stop_strings
                    stop_strings=["</output>", "\n\n", "Human:", "Assistant:"],
                    tokenizer=policy_tok,  # Required when using stop_strings
                )
                completion = policy_tok.decode(
                    out_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                _, attacked_note = parse_response(completion)

                # Add fallback for empty attacked_note
                if not attacked_note.strip():
                    print(
                        "WARNING: Attacker did not produce an <output> block. Using full completion as attacked note."
                    )
                    attacked_note = completion.strip()

                attacked_records.append(
                    {"original": row["original"], "attacked": attacked_note}
                )

        ds_assessor_round = make_assessor_prompts(attacked_records, policy_tok)

        print(f"--- Round {r+1}: Training Assessor ---")
        assessor_trainer = GRPOTrainer(
            model=policy_model,
            args=GRPOConfig(**common_cfg),
            processing_class=policy_tok,
            train_dataset=ds_assessor_round,
            reward_funcs=[assessor_reward_fn],
        )
        assessor_trainer.train()

        # Clear memory after each round
        del assessor_trainer, snap, ds_assessor_round
        assessor_snapshot["model"] = None

        # Force memory cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"📄 JSONL log written to {log_path}")
    print(
        f"📄 Interaction log written to {log_path.replace('.jsonl', '_interactions.jsonl')}"
    )


if __name__ == "__main__":
    print("Script reached main execution...")
    main()
