print("Starting script imports...")
import os
import re
import json
import random
import argparse
import gc
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
R_HARM = 2.0      # Main game reward for harm classification
R_REFUSAL = 0.5   # Appropriate refusal reward  
R_FORMAT = 0.1    # CoT formatting reward

print("Constants defined...")

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
    """Loads a causal language model and its tokenizer."""
    print(f"Loading model: {model_id} to device: {device}")
    
    # Use proper dtype handling
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        trust_remote_code=True,
        device_map="auto"
    )
    
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Ensure proper padding setup
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    return model, tok

def parse_qwen3_thinking_response(text: str):
    """Parse Qwen3's native thinking format response."""
    thought = ""
    output = ""
    
    # Qwen3 generates <think>...</think> followed by the actual response
    if "<think>" in text.lower() and "</think>" in text.lower():
        # Handle case-insensitive matching
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        think_match = think_pattern.search(text)
        
        if think_match:
            thought = think_match.group(1).strip()
            # Get everything after </think> as output
            output = think_pattern.sub('', text).strip()
    else:
        # No thinking block found, entire text is output
        output = text.strip()
    
    # Clean up any remaining tags or artifacts
    output = re.sub(r'<[^>]*>', '', output).strip()
    
    print(f"DEBUG: Parsed - thought: {len(thought)} chars, output: {len(output)} chars")
    return thought, output

def parse_custom_format_response(text: str):
    """Parse custom format response when thinking is disabled."""
    thought = ""
    output = ""
    
    # Look for our custom format markers
    if "REASONING:" in text and "MODIFIED:" in text:
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?:MODIFIED:|$)', text, re.DOTALL)
        modified_match = re.search(r'MODIFIED:\s*(.*?)$', text, re.DOTALL)
        
        if reasoning_match:
            thought = reasoning_match.group(1).strip()
        if modified_match:
            output = modified_match.group(1).strip()
    elif "CLASSIFICATION:" in text:
        # For assessor responses
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?:CLASSIFICATION:|$)', text, re.DOTALL)
        class_match = re.search(r'CLASSIFICATION:\s*(.*?)$', text, re.DOTALL)
        
        if reasoning_match:
            thought = reasoning_match.group(1).strip()
        if class_match:
            output = class_match.group(1).strip()
    else:
        # Fallback: use the entire text as output
        output = text.strip()
    
    return thought, output

def get_judge_assessment(original: str, attacked: str, assessor_label: str, judge_model, judge_tok, device):
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
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    # Use enable_thinking=False for judge to get direct answers
    prompt = judge_tok.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False
    )

    with torch.no_grad():
        inputs = judge_tok(prompt, return_tensors="pt").to(device)
        out = judge_model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False,
            temperature=0.7,
            top_p=0.8,
            top_k=20
        )
        text = judge_tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    judgments = {"actual_harm": "Safe", "assessor_was_correct": False, "assessor_refused": False}
    try:
        jtxt = text[text.find("{") : text.rfind("}") + 1]
        j = json.loads(jtxt)
        judgments["actual_harm"] = j.get("actual_harm", "Safe")
    except Exception:
        if attacked.strip() != original.strip():
            judgments["actual_harm"] = "Concerning"
    
    label = assessor_label.strip().title()
    if "Can't Assess" in label or "Cannot" in label.lower():
        judgments["assessor_refused"] = True
    elif label == judgments["actual_harm"]:
        judgments["assessor_was_correct"] = True
        
    return judgments

def load_and_prepare_data(num_samples: int):
    """Loads MEDEC data, filtering for examples with known errors for few-shot prompting."""
    print("Loading and preparing MEDEC data...")
    path = "data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
    df = pd.read_csv(path).fillna("")
    df["original"] = df.apply(lambda r: (r["Corrected Text"].strip() or r["Text"]), axis=1)
    df = df[df["original"].str.strip() != ""]
    
    df_with_errors = df[df["Error Flag"] == 1]
    ds_few_shot = Dataset.from_pandas(df_with_errors[["original", "Text"]].rename(columns={"Text": "attacked"}))
    ds_originals = Dataset.from_pandas(df[["original"]]).shuffle(seed=42).select(range(min(num_samples, len(df))))
    
    return ds_originals, ds_few_shot

# Option 1: Use Qwen3's native thinking mode
def build_attacker_prompts_thinking_mode(ds: Dataset, few_shot_examples: Dataset, tokenizer, num_shots=2):
    """Build attacker prompts using Qwen3's native thinking mode."""
    
    system_content = """You are a medical editor tasked with introducing subtle medical errors for training purposes.

Your task:
1. Read the provided medical note carefully
2. Introduce ONE subtle but clinically significant error
3. The error should be factual (wrong dosage, incorrect code, wrong medication, etc.)
4. Make the error subtle enough to not be immediately obvious
5. Return the complete modified medical note

Think through your reasoning first, then provide the modified note."""

    def to_prompt(row):
        user_content = f"""Original medical note:
{row['original']}

Please introduce one subtle medical error into this note. Think about what type of error would be clinically significant but not immediately obvious, then provide the modified version."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Enable thinking mode for chain-of-thought reasoning
        prompt_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True  # Use Qwen3's built-in thinking mode
        )
        
        tokenized = tokenizer(prompt_string, truncation=True, max_length=2048)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "prompt": prompt_string,
            "original_note": row['original'],
        }
    
    return ds.map(to_prompt, remove_columns=ds.column_names)

# Option 2: Use custom format with thinking disabled
def build_attacker_prompts_custom_format(ds: Dataset, few_shot_examples: Dataset, tokenizer, num_shots=2):
    """Build attacker prompts with custom format (thinking disabled)."""
    
    system_content = """You are a medical editor. You MUST respond in EXACTLY this format:

REASONING: [Brief explanation of what error you're introducing and why]
MODIFIED: [The complete modified medical note with the error]

Rules:
- Add exactly ONE subtle medical error
- The error must be clinically significant but not obvious
- Examples: wrong dosage, incorrect diagnosis code, wrong medication name
- Always use the exact format above with REASONING: and MODIFIED: labels"""

    def to_prompt(row):
        user_content = f"""Original medical note:
{row['original']}

Introduce one subtle medical error using the specified format."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Disable thinking mode for custom format
        prompt_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking for custom format
        )
        
        tokenized = tokenizer(prompt_string, truncation=True, max_length=2048)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "prompt": prompt_string,
            "original_note": row['original'],
        }
    
    return ds.map(to_prompt, remove_columns=ds.column_names)

def make_assessor_prompts_thinking_mode(records: list, tokenizer):
    """Make assessor prompts using Qwen3's native thinking mode."""
    
    system_content = """You are a medical safety assessor. Your task is to classify medical notes for potential errors.

Classification levels:
- Safe: No clinically significant errors detected
- Concerning: Minor errors or ambiguities that could cause delays or minor risks
- Harmful: Clear factual errors that could lead to patient harm

Analyze the note carefully, think through your assessment, then provide your classification."""

    prompts = []
    for rec in records:
        user_content = f"""Please analyze this medical note and classify it as Safe, Concerning, or Harmful:

{rec['attacked']}

Think through your analysis first, then provide your classification."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        prompt_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True  # Use thinking mode
        )
        
        tokenized = tokenizer(prompt_string, truncation=True, max_length=2048)
        prompts.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "prompt": prompt_string,
            "original_note": rec['original'],
            "attacked_note": rec['attacked']
        })

    return Dataset.from_list(prompts)

def make_assessor_prompts_custom_format(records: list, tokenizer):
    """Make assessor prompts with custom format (thinking disabled)."""
    
    system_content = """You are a medical safety assessor. You MUST respond in EXACTLY this format:

REASONING: [Brief analysis of potential errors or safety concerns]
CLASSIFICATION: [One word only: Safe, Concerning, or Harmful]

Classification levels:
- Safe: No clinically significant errors
- Concerning: Minor errors that could cause delays or minor risks  
- Harmful: Clear errors that could lead to patient harm

Always use the exact format above."""

    prompts = []
    for rec in records:
        user_content = f"""Analyze this medical note for errors:

{rec['attacked']}

Classify using the specified format."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        prompt_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking for custom format
        )
        
        tokenized = tokenizer(prompt_string, truncation=True, max_length=2048)
        prompts.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "prompt": prompt_string,
            "original_note": rec['original'],
            "attacked_note": rec['attacked']
        })

    return Dataset.from_list(prompts)

def extract_original_from_prompt(prompt: str) -> str:
    """Extract the original note from a prompt."""
    patterns = [
        r"Original medical note:\s*\n(.*?)(?:\n\nPlease introduce|$)",
        r"Original medical note:\s*(.*?)(?:\nPlease|$)",
        r"Original note:\s*\n(.*?)(?:\n\n|$)",
        r"Original note:\s*(.*?)(?:\n|$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return ""

def extract_attacked_from_prompt(prompt: str) -> str:
    """Extract attacked note from assessor prompt."""
    patterns = [
        r"analyze this medical note.*?:\s*\n(.*?)(?:\n\nThink through|$)",
        r"Analyze this medical note.*?:\s*\n(.*?)(?:\n\nClassify|$)",
        r"medical note.*?:\s*\n(.*?)$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ""

def log_interaction(round_num, phase, original, attacked, assessor_response, judgments, rewards, log_path):
    """Log detailed interaction data for analysis."""
    interaction_log = {
        "round": round_num,
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "original_note": original,
        "attacked_note": attacked,
        "assessor_response": assessor_response,
        "judge_assessment": judgments,
        "rewards": rewards
    }
    
    interaction_log_path = log_path.replace(".jsonl", "_interactions.jsonl")
    with open(interaction_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(interaction_log, ensure_ascii=False) + "\n")

def main():
    print("Main function started...")
    parser = argparse.ArgumentParser(description="GRPO self-play for Attacker vs. Assessor training.")
    parser.add_argument("--model_id", type=str, required=True, help="Shared policy model to be trained.")
    parser.add_argument("--judge_model_id", type=str, default="Qwen/Qwen3-1.7B", help="Judge model for rewards.")
    parser.add_argument("--num_samples", type=int, default=16, help="Original notes to use.")
    parser.add_argument("--num_generations", type=int, default=2, help="GRPO completions per prompt (>=2).")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--rounds", type=int, default=3, help="Self-play rounds.")
    parser.add_argument("--max_assessor_batch", type=int, default=64, help="New notes for the assessor each round.")
    parser.add_argument("--use_thinking_mode", type=bool, default=True, help="Use Qwen3's native thinking mode (recommended)")
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")

    # Set memory optimization environment variable
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = get_device() 
    policy_model, policy_tok = load_causal_lm(args.model_id, device)
    judge_model, judge_tok = load_causal_lm(args.judge_model_id, device)

    ds_originals, ds_few_shot = load_and_prepare_data(args.num_samples)
    
    # Choose prompting strategy based on argument
    if args.use_thinking_mode:
        print("Using Qwen3's native thinking mode...")
        build_attacker_prompts = build_attacker_prompts_thinking_mode
        make_assessor_prompts = make_assessor_prompts_thinking_mode
        parse_response = parse_qwen3_thinking_response
        # Generation parameters for thinking mode
        gen_params = {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.05,
        }
    else:
        print("Using custom format with thinking disabled...")
        build_attacker_prompts = build_attacker_prompts_custom_format
        make_assessor_prompts = make_assessor_prompts_custom_format
        parse_response = parse_custom_format_response
        # Generation parameters for non-thinking mode
        gen_params = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
        }
    
    ds_attacker = build_attacker_prompts(ds_originals, ds_few_shot, policy_tok)

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
        for p, c in zip(prompts, completions):
            original = extract_original_from_prompt(p)
            attacked = extract_attacked_from_prompt(p)
            
            thought, label = parse_response(c)
            
            # Normalize label for comparison
            label = label.strip().title()
            if "safe" in label.lower():
                label = "Safe"
            elif "concerning" in label.lower():
                label = "Concerning"
            elif "harmful" in label.lower():
                label = "Harmful"
            
            judgments = get_judge_assessment(original, attacked, label, judge_model, judge_tok, device)
            
            # Harm multipliers
            harm_multipliers = {"Safe": 0.5, "Concerning": 1.0, "Harmful": 2.0}
            multiplier = harm_multipliers.get(judgments["actual_harm"], 1.0)
            
            # Harmfulness reward
            r_harm = R_HARM * multiplier if judgments["assessor_was_correct"] else -R_HARM * multiplier
            
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
                
            # Format reward (reward for having reasoning)
            r_format = R_FORMAT if thought and len(thought) > 10 else -R_FORMAT
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
                    "total": total_reward
                },
                log_path=log_path
            )
            scores.append(total_reward)
        return scores

    def attacker_reward_fn(prompts, completions, **kwargs):
        scores = []
        frozen_assessor = assessor_snapshot["model"]
        
        train_dataset = kwargs.get('train_dataset')
        
        for i, (p, c) in enumerate(zip(prompts, completions)):
            original = train_dataset[i]['original_note'] if train_dataset else extract_original_from_prompt(p)
            attacker_thought, attacked_note = parse_response(c)
            
            # Ensure we have a valid attacked note
            if not attacked_note or len(attacked_note) < 10:
                attacked_note = c.strip()
                
            assessor_ds = make_assessor_prompts([{"original": original, "attacked": attacked_note}], policy_tok)
            assessor_prompt = assessor_ds[0]['prompt']
            
            with torch.no_grad():
                inputs = policy_tok(assessor_prompt, return_tensors="pt").to(device)
                out_ids = frozen_assessor.generate(
                    **inputs, 
                    max_new_tokens=1024,
                    **gen_params,
                    pad_token_id=policy_tok.eos_token_id,
                    eos_token_id=policy_tok.eos_token_id,
                )
                assessor_completion = policy_tok.decode(out_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                _, assessor_label = parse_response(assessor_completion)
                
            # Normalize assessor label
            assessor_label = assessor_label.strip().title()
            if "safe" in assessor_label.lower():
                assessor_label = "Safe"
            elif "concerning" in assessor_label.lower():
                assessor_label = "Concerning"
            elif "harmful" in assessor_label.lower():
                assessor_label = "Harmful"
                
            judgments = get_judge_assessment(original, attacked_note, assessor_label, judge_model, judge_tok, device)
            
            harm_multipliers = {"Safe": 0.5, "Concerning": 1.0, "Harmful": 2.0}
            multiplier = harm_multipliers.get(judgments["actual_harm"], 1.0)
            
            # Harmfulness reward (zero-sum)
            r_harm = -R_HARM * multiplier if judgments["assessor_was_correct"] else R_HARM * multiplier
            
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
            r_format = R_FORMAT if attacker_thought and len(attacker_thought) > 10 else -R_FORMAT
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
                    "total": total_reward
                },
                log_path=log_path
            )
            scores.append(total_reward)
        return scores

    # --- Trainer Config with memory optimizations ---
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
    )
    
    for r in range(args.rounds):
        state["round"] = r + 1
        print(f"\n{'='*25} Self-play round {r+1}/{args.rounds} {'='*25}")

        # Log round start
        log_jsonl({
            "round": r + 1,
            "phase": "round_start",
            "timestamp": datetime.now().isoformat(),
            "model_id": args.model_id,
            "thinking_mode": args.use_thinking_mode
        })

        snap = deepcopy(policy_model).eval()
        assessor_snapshot["model"] = snap

        print(f"--- Round {r+1}: Training Attacker ---")
        attacker_trainer = GRPOTrainer(
            model=policy_model, 
            args=GRPOConfig(**common_cfg), 
            processing_class=policy_tok,
            train_dataset=ds_attacker, 
            reward_funcs=[attacker_reward_fn]
        )
        attacker_trainer.train()
        
        # Clear attacker trainer
        del attacker_trainer

        print(f"--- Round {r+1}: Generating new dataset for Assessor ---")
        attacked_records = []
        with torch.no_grad():
            sel = ds_originals.shuffle(seed=42 + r).select(range(min(args.max_assessor_batch, len(ds_originals))))
            for row in sel:
                attacker_ds = build_attacker_prompts(Dataset.from_dict({"original": [row["original"]]}), ds_few_shot, policy_tok)
                prompt = attacker_ds[0]['prompt']
                inputs = policy_tok(prompt, return_tensors="pt").to(device)
                
                # Generate with proper parameters
                out_ids = policy_model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    **gen_params,
                    pad_token_id=policy_tok.eos_token_id,
                    eos_token_id=policy_tok.eos_token_id,
                )
                completion = policy_tok.decode(out_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                _, attacked_note = parse_response(completion)
                
                # Ensure we have valid output
                if not attacked_note or len(attacked_note) < 10:
                    print(f"WARNING: Attacker produced insufficient output. Using full completion.")
                    attacked_note = completion.strip()
                    
                attacked_records.append({"original": row["original"], "attacked": attacked_note})
                
        ds_assessor_round = make_assessor_prompts(attacked_records, policy_tok)

        print(f"--- Round {r+1}: Training Assessor ---")
        assessor_trainer = GRPOTrainer(
            model=policy_model, 
            args=GRPOConfig(**common_cfg), 
            processing_class=policy_tok,
            train_dataset=ds_assessor_round, 
            reward_funcs=[assessor_reward_fn]
        )
        assessor_trainer.train()

        # Clear memory after each round
        del assessor_trainer, snap, ds_assessor_round
        assessor_snapshot["model"] = None
        
        # Force memory cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"✅ Training completed successfully!")
    print(f"📄 JSONL log written to {log_path}")
    print(f"📄 Interaction log written to {log_path.replace('.jsonl', '_interactions.jsonl')}")

if __name__ == "__main__":
    print("Script reached main execution...")
    main()