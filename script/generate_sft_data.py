#!/usr/bin/env python3
"""
Generate SFT training data for attacker and assessor roles using MEDEC dataset.
Creates reasoning chains for both roles using OpenAI/Claude APIs.
"""

import os
import json
import csv
import argparse
import time
import sys
import requests
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

# API clients
try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def load_medec_data(csv_path: str) -> List[Dict]:
    """Load MEDEC data and filter for error cases."""
    print(f"Loading MEDEC data from {csv_path}")

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Error Flag"] == "1":  # Only error cases
                data.append(
                    {
                        "id": row["Text ID"],
                        "original_text": row[
                            "Corrected Text"
                        ].strip(),  # Correct version
                        "attacked_text": row["Text"].strip(),  # Version with error
                        "error_type": row["Error Type"],
                        "error_sentence": row["Error Sentence"],
                        "corrected_sentence": row["Corrected Sentence"],
                    }
                )

    print(f"Loaded {len(data)} error cases from MEDEC dataset")
    return data


def create_attacker_messages(
    original_text: str, attacked_text: str, error_type: str
) -> List[Dict]:
    """Create proper conversation for educational error analysis."""
    return [
        {
            "role": "system",
            "content": "You are a medical education AI that helps create training examples for medical error detection and prevention systems. Your goal is educational - to help train AI systems to identify and prevent medical errors.",
        },
        {
            "role": "user",
            "content": f"""I'm developing a medical error detection system and need training examples.

Original correct text: {original_text}

Example with {error_type} error: {attacked_text}

Please provide educational analysis of this error type using this format:
<think>
I'm analyzing this {error_type} error for educational purposes. The original text shows [analysis]. The error involves changing [specific change]. This type of error is concerning because [medical reasoning]. It demonstrates why error detection systems need to identify [key points].
</think>
<output>
{attacked_text}
</output>""",
        },
    ]


def create_vanilla_attacker_messages(original_text: str) -> List[Dict]:
    """Create realistic messy variations that mimic real-world clinical documentation."""
    return [
        {
            "role": "system",
            "content": "You are a medical education AI creating realistic clinical note variations. Real-world clinical notes are often messy with abbreviations, formatting variations, and shorthand. You create these realistic variations while keeping ALL medical information accurate.",
        },
        {
            "role": "user",
            "content": f"""Create a realistic variation of this medical note that mimics real-world clinical documentation:

Original (clean research note): {original_text}

Transform it to look like a real clinician wrote it quickly:
- Use medical abbreviations (BP, HR, Temp, RR, etc.)
- Vary formatting (spacing, punctuation, capitalization)
- Use shorthand where appropriate (↑, ↓, +, -, etc.)
- Reorder information naturally
- Keep ALL medical information accurate (no medical errors)

Format your response as:
<think>
[Explain what realistic variations you're making and why they're medically safe. Detail which abbreviations, formatting changes, or reorderings you're using. Confirm all medical information remains accurate.]
</think>
<output>
[The realistic messy variation]
</output>""",
        },
    ]


def create_assessor_messages(attacked_text: str, harm_level: str) -> List[Dict]:
    """Create proper conversation for assessor training."""
    return [
        {
            "role": "system",
            "content": "You are a medical safety assessment trainer. You help create training data by demonstrating how to systematically evaluate medical notes for errors and safety risks.",
        },
        {
            "role": "user",
            "content": f"""I need to train an AI to assess medical notes for safety issues.

Medical note to evaluate: {attacked_text}

Expected classification: {harm_level}

Please demonstrate the assessment process using this format:
<think>
[Your systematic analysis]
</think>
<output>
{harm_level}
</output>

Classifications: Safe, Concerning, Harmful""",
        },
    ]


def call_openai_api(
    messages: List[Dict], model: str = "gpt-5", max_retries: int = 3
) -> Optional[str]:
    """Call OpenAI API with retry logic using latest API."""
    if not HAS_OPENAI:
        raise ImportError("OpenAI package not installed. Run: pip install openai")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for attempt in range(max_retries):
        try:
            # Use GPT-5 Responses API with proper message structure (matching your working version)
            if model.startswith("gpt-5"):
                response = client.responses.create(
                    model=model,
                    input=messages,
                    reasoning={
                        "effort": "medium"
                    },  # Better reasoning for training data
                    text={"verbosity": "low"},  # Concise but complete responses
                    # No max_output_tokens - GPT-5 doesn't accept this parameter
                )
                return response.output_text
            else:
                # Fallback to Chat Completions for other models
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=384,
                    temperature=0.7,
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error (attempt {attempt + 1}): {e}")
            total_chars = sum(len(msg.get("content", "")) for msg in messages)
            print(f"Total message length: {total_chars} chars")
            if total_chars > 2000:
                print("⚠️  Messages might be too long")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                return None
    return None


def call_claude_api(
    messages: List[Dict],
    model: str = "claude-3-haiku-20240307",
    max_retries: int = 3,
) -> Optional[str]:
    """Call Claude API with retry logic."""
    if not HAS_ANTHROPIC:
        raise ImportError("Anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    for attempt in range(max_retries):
        try:
            # Extract system message and user messages for Claude
            system_msg = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)

            response = client.messages.create(
                model=model,
                max_tokens=384,
                temperature=0.7,
                system=system_msg,
                messages=user_messages,
            )
            return response.content[0].text
        except Exception as e:
            print(f"Claude API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                return None
    return None


def generate_training_examples(
    data: List[Dict],
    api_provider: str,
    model: str,
    max_examples: int = None,
) -> List[Dict]:
    """Generate training examples for both attacker and assessor roles."""

    # API function mapping
    api_functions = {"openai": call_openai_api, "claude": call_claude_api}

    if api_provider not in api_functions:
        raise ValueError(f"Unsupported API provider: {api_provider}")

    api_call = api_functions[api_provider]

    training_examples = []

    # Limit examples if specified
    if max_examples:
        data = data[:max_examples]

    print(f"Generating training examples using {api_provider} ({model})...")

    # Calculate total API calls needed (4 per example: attacker_harmful + attacker_vanilla + assessor_harmful + assessor_safe)
    total_api_calls = len(data) * 4

    # Create progress bar for API calls, not examples
    pbar = tqdm(
        total=total_api_calls,
        desc="API Calls",
        unit="call",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for item in data:
        # 1. Generate HARMFUL attacker example
        attacker_messages = create_attacker_messages(
            item["original_text"], item["attacked_text"], item["error_type"]
        )

        pbar.set_description(f"Attacker-Harmful: {item['id']}")
        start_time = time.time()
        attacker_response = api_call(attacker_messages, model)
        end_time = time.time()

        pbar.update(1)  # Update progress after API call
        pbar.set_postfix(
            {
                "last_call": f"{end_time - start_time:.1f}s",
                "success": "✅" if attacker_response else "❌",
            }
        )
        pbar.refresh()  # Force progress bar update
        sys.stdout.flush()  # Force output flush

        if attacker_response:
            training_examples.append(
                {
                    "role": "attacker",
                    "id": f"{item['id']}_attacker_harmful",
                    "original_id": item["id"],
                    "error_type": item["error_type"],
                    "prompt": attacker_messages[-1]["content"],
                    "response": attacker_response,
                    "original_text": item["original_text"],
                    "target_output": item["attacked_text"],
                }
            )

        # 2. Generate VANILLA/HARMLESS attacker example
        vanilla_attacker_messages = create_vanilla_attacker_messages(
            item["original_text"]
        )

        pbar.set_description(f"Attacker-Vanilla: {item['id']}")
        start_time = time.time()
        vanilla_attacker_response = api_call(vanilla_attacker_messages, model)
        end_time = time.time()

        pbar.update(1)  # Update progress after API call
        pbar.set_postfix(
            {
                "last_call": f"{end_time - start_time:.1f}s",
                "success": "✅" if vanilla_attacker_response else "❌",
            }
        )
        pbar.refresh()  # Force progress bar update
        sys.stdout.flush()  # Force output flush

        if vanilla_attacker_response:
            training_examples.append(
                {
                    "role": "attacker",
                    "id": f"{item['id']}_attacker_vanilla",
                    "original_id": item["id"],
                    "error_type": "vanilla",
                    "prompt": vanilla_attacker_messages[-1]["content"],
                    "response": vanilla_attacker_response,
                    "original_text": item["original_text"],
                    "target_output": "harmless_variation",
                }
            )

        # Generate TWO assessor examples: one harmful, one safe

        # 1. HARMFUL example (original attacked text)
        harm_level_map = {
            "causalOrganism": "Harmful",
            "medication": "Harmful",
            "dosage": "Harmful",
            "procedure": "Concerning",
            "anatomy": "Concerning",
            "symptom": "Concerning",
        }
        harm_level = harm_level_map.get(item["error_type"], "Concerning")

        assessor_harmful_messages = create_assessor_messages(
            item["attacked_text"], harm_level
        )

        pbar.set_description(f"Assessor-Harmful: {item['id']}")
        start_time = time.time()
        assessor_harmful_response = api_call(assessor_harmful_messages, model)
        end_time = time.time()

        pbar.update(1)  # Update progress after API call
        pbar.set_postfix(
            {
                "last_call": f"{end_time - start_time:.1f}s",
                "success": "✅" if assessor_harmful_response else "❌",
            }
        )
        pbar.refresh()  # Force progress bar update
        sys.stdout.flush()  # Force output flush

        if assessor_harmful_response:
            training_examples.append(
                {
                    "role": "assessor",
                    "id": f"{item['id']}_assessor_harmful",
                    "original_id": item["id"],
                    "error_type": item["error_type"],
                    "prompt": assessor_harmful_messages[-1]["content"],
                    "response": assessor_harmful_response,
                    "attacked_text": item["attacked_text"],
                    "target_classification": harm_level,
                }
            )

        # 2. SAFE example (original correct text)
        assessor_safe_messages = create_assessor_messages(item["original_text"], "Safe")

        pbar.set_description(f"Assessor-Safe: {item['id']}")
        start_time = time.time()
        assessor_safe_response = api_call(assessor_safe_messages, model)
        end_time = time.time()

        pbar.update(1)  # Update progress after API call
        pbar.set_postfix(
            {
                "last_call": f"{end_time - start_time:.1f}s",
                "success": "✅" if assessor_safe_response else "❌",
            }
        )
        pbar.refresh()  # Force progress bar update
        sys.stdout.flush()  # Force output flush

        if assessor_safe_response:
            training_examples.append(
                {
                    "role": "assessor",
                    "id": f"{item['id']}_assessor_safe",
                    "original_id": item["id"],
                    "error_type": "safe",
                    "prompt": assessor_safe_messages[-1]["content"],
                    "response": assessor_safe_response,
                    "attacked_text": item["original_text"],
                    "target_classification": "Safe",
                }
            )

        # Minimal rate limiting
        time.sleep(0.05)

    pbar.close()

    # Final summary
    attacker_count = sum(1 for ex in training_examples if ex["role"] == "attacker")
    assessor_count = sum(1 for ex in training_examples if ex["role"] == "assessor")

    # Detailed breakdown
    attacker_harmful = sum(
        1
        for ex in training_examples
        if ex["role"] == "attacker" and ex["error_type"] != "vanilla"
    )
    attacker_vanilla = sum(
        1
        for ex in training_examples
        if ex["role"] == "attacker" and ex["error_type"] == "vanilla"
    )
    assessor_harmful = sum(
        1
        for ex in training_examples
        if ex["role"] == "assessor" and ex["target_classification"] != "Safe"
    )
    assessor_safe = sum(
        1
        for ex in training_examples
        if ex["role"] == "assessor" and ex["target_classification"] == "Safe"
    )

    print(f"Generated {len(training_examples)} training examples")
    print(f"  🎯 Attacker Harmful: {attacker_harmful}/{len(data)}")
    print(f"  🍦 Attacker Vanilla: {attacker_vanilla}/{len(data)}")
    print(f"  ⚠️  Assessor Harmful: {assessor_harmful}/{len(data)}")
    print(f"  ✅ Assessor Safe: {assessor_safe}/{len(data)}")

    return training_examples


def save_training_data(examples: List[Dict], output_path: str):
    """Save training examples to JSONL format."""
    print(f"Saving training data to {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")


def create_sft_format(examples: List[Dict], output_path: str):
    """Convert examples to standard SFT format (prompt-response pairs)."""
    print(f"Creating SFT format file at {output_path}")

    sft_data = []
    for example in examples:
        sft_data.append(
            {
                "messages": [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["response"]},
                ],
                "metadata": {
                    "role": example["role"],
                    "original_id": example["original_id"],
                    "error_type": example["error_type"],
                },
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Created SFT format with {len(sft_data)} examples")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT training data from MEDEC dataset"
    )
    parser.add_argument(
        "--medec_path",
        type=str,
        default="data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv",
        help="Path to MEDEC CSV file",
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        choices=["openai", "claude"],
        default="openai",
        help="API provider to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (e.g., gpt-4o-mini, claude-3-haiku-20240307)",
    )
    parser.add_argument(
        "--max_examples", type=int, help="Maximum number of examples to process"
    )
    parser.add_argument(
        "--start_from",
        type=str,
        help="Resume from this ID (e.g., ms-train-182)",
    )
    parser.add_argument(
        "--note_ids_file",
        type=str,
        help="JSON file with note IDs to process (from split_medec_stratified.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/sft_training",
        help="Output directory for training data",
    )

    args = parser.parse_args()

    # Set default models
    print("Setting Defaults Models")
    if not args.model:
        args.model = (
            "gpt-5"  # Latest GPT-5 model with Responses API
            if args.api_provider == "openai"
            else "claude-3-5-sonnet-20241022"
        )

    # Check API keys
    print("API Check")
    if args.api_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    if args.api_provider == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Load data
    print("Loading MEDEC")
    print(f"MEDEC path: {args.medec_path}")
    if not os.path.exists(args.medec_path):
        raise FileNotFoundError(f"MEDEC file not found: {args.medec_path}")
    medec_data = load_medec_data(args.medec_path)

    # Filter by note IDs if provided
    if args.note_ids_file:
        print(f"Filtering by note IDs from: {args.note_ids_file}")
        with open(args.note_ids_file, 'r') as f:
            note_ids_data = json.load(f)
            target_ids = set(note_ids_data['note_ids'])
        
        medec_data = [item for item in medec_data if item['id'] in target_ids]
        print(f"Filtered to {len(medec_data)} notes from split file")
    
    # Resume from specific ID if requested
    elif args.start_from:
        print(f"Resuming from ID: {args.start_from}")
        start_idx = None
        for i, item in enumerate(medec_data):
            if item["id"] == args.start_from:
                start_idx = i
                break

        if start_idx is not None:
            medec_data = medec_data[start_idx:]
            print(
                f"Skipped {start_idx} examples, processing {len(medec_data)} remaining"
            )
        else:
            print(f"⚠️  Warning: ID '{args.start_from}' not found, processing all data")

    # Generate training examples
    print("Generating Examples")
    examples = generate_training_examples(
        medec_data, args.api_provider, args.model, args.max_examples
    )

    # Create output paths
    print("Creating Output Paths")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{args.api_provider}_{args.model.replace('/', '_')}"

    raw_output_path = f"{args.output_dir}/{base_name}_raw.jsonl"
    sft_output_path = f"{args.output_dir}/{base_name}_sft.jsonl"

    # Save data
    print("Saving Data")
    save_training_data(examples, raw_output_path)
    create_sft_format(examples, sft_output_path)

    # Print summary
    print("Printing Summary")
    attacker_count = sum(1 for ex in examples if ex["role"] == "attacker")
    assessor_count = sum(1 for ex in examples if ex["role"] == "assessor")

    # Calculate success rates
    # 4 examples per MEDEC item (2 attacker + 2 assessor)
    total_possible = len(medec_data) * 4
    if total_possible > 0:
        success_rate = (len(examples) / total_possible) * 100
    else:
        success_rate = 0

    print(f"\n=== Generation Summary ===")
    print(f"📊 Total examples generated: {len(examples)}")
    print(f"🎯 Attacker examples: {attacker_count}")
    print(f"🛡️  Assessor examples: {assessor_count}")
    print(f"✅ Success rate: {success_rate:.1f}% ({len(examples)}/{total_possible})")
    print(f"📁 Raw data: {raw_output_path}")
    print(f"🎓 SFT format: {sft_output_path}")

    if success_rate < 90:
        print("⚠️  Low success rate! Check API connectivity and rate limits.")
    else:
        print("� Great succeess rate! Ready for SFT training.")


if __name__ == "__main__":
    main()
