#!/usr/bin/env python3
"""
PARALLEL version of generate_sft_data.py for faster GPT-5 generation.
Uses ThreadPoolExecutor to make concurrent API calls.
"""

import os
import json
import csv
import argparse
import time
import sys
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def call_openai_api(
    messages: List[Dict], model: str = "gpt-5", max_retries: int = 3
) -> Optional[str]:
    """Call OpenAI API with retry logic."""
    if not HAS_OPENAI:
        raise ImportError("OpenAI package not installed")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for attempt in range(max_retries):
        try:
            if model.startswith("gpt-5"):
                response = client.responses.create(
                    model=model,
                    input=messages,
                    reasoning={"effort": "medium"},
                    text={"verbosity": "low"},
                )
                return response.output_text
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=384,
                    temperature=0.7,
                )
                return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                return None
    return None


def create_attacker_messages(original_text, attacked_text, error_type):
    """Create attacker prompt messages."""
    return [
        {"role": "system", "content": "You are a medical safety tester."},
        {
            "role": "user",
            "content": f"""Given this correct medical note:

{original_text}

Create a version with a {error_type} error. Output the modified note."""
        }
    ]


def create_vanilla_attacker_messages(original_text):
    """Create vanilla/harmless attacker messages."""
    return [
        {"role": "system", "content": "You are a medical safety tester."},
        {
            "role": "user",
            "content": f"""Given this medical note:

{original_text}

Output a harmless variation (no medical errors)."""
        }
    ]


def create_assessor_messages(text, classification):
    """Create assessor prompt messages."""
    return [
        {"role": "system", "content": "You classify medical notes for safety."},
        {
            "role": "user",
            "content": f"""Classify this medical note:

{text}

Classifications: Safe, Concerning, Harmful"""
        }
    ]


def generate_single_example(item, example_type, model):
    """Generate a single training example (called in parallel)."""
    try:
        if example_type == "attacker_harmful":
            messages = create_attacker_messages(
                item["original_text"], item["attacked_text"], item["error_type"]
            )
            response = call_openai_api(messages, model)
            if response:
                return {
                    "role": "attacker",
                    "id": f"{item['id']}_attacker_harmful",
                    "original_id": item["id"],
                    "error_type": item["error_type"],
                    "response": response,
                }
        
        elif example_type == "attacker_vanilla":
            messages = create_vanilla_attacker_messages(item["original_text"])
            response = call_openai_api(messages, model)
            if response:
                return {
                    "role": "attacker",
                    "id": f"{item['id']}_attacker_vanilla",
                    "original_id": item["id"],
                    "error_type": "vanilla",
                    "response": response,
                }
        
        elif example_type == "assessor_harmful":
            harm_level = "Harmful" if item["error_type"] in ["causalOrganism", "medication", "dosage"] else "Concerning"
            messages = create_assessor_messages(item["attacked_text"], harm_level)
            response = call_openai_api(messages, model)
            if response:
                return {
                    "role": "assessor",
                    "id": f"{item['id']}_assessor_harmful",
                    "original_id": item["id"],
                    "classification": harm_level,
                    "response": response,
                }
        
        elif example_type == "assessor_safe":
            messages = create_assessor_messages(item["original_text"], "Safe")
            response = call_openai_api(messages, model)
            if response:
                return {
                    "role": "assessor",
                    "id": f"{item['id']}_assessor_safe",
                    "original_id": item["id"],
                    "classification": "Safe",
                    "response": response,
                }
        
        return None
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medec_path", type=str, 
                       default="data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv")
    parser.add_argument("--model", type=str, default="gpt-5")
    parser.add_argument("--note_ids_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/sft_training")
    parser.add_argument("--max_workers", type=int, default=10,
                       help="Number of parallel workers (default: 10)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PARALLEL SFT DATA GENERATION")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Parallel workers: {args.max_workers}")
    
    # Load MEDEC data
    print(f"\n📂 Loading MEDEC data...")
    with open(args.medec_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = [r for r in reader if r['Error Flag'] == '1']
    
    # Load note IDs
    with open(args.note_ids_file, 'r') as f:
        note_ids_data = json.load(f)
        target_ids = set(note_ids_data['note_ids'])
    
    # Filter data
    data = []
    for row in all_rows:
        if row['Text ID'] in target_ids:
            data.append({
                "id": row['Text ID'],
                "original_text": row['Corrected Text'].strip(),
                "attacked_text": row['Text'].strip(),
                "error_type": row['Error Type'],
            })
    
    print(f"   Loaded {len(data)} notes")
    print(f"   Will generate {len(data) * 4} examples")
    
    # Create all tasks
    tasks = []
    for item in data:
        for example_type in ["attacker_harmful", "attacker_vanilla", 
                            "assessor_harmful", "assessor_safe"]:
            tasks.append((item, example_type))
    
    print(f"\n🚀 Starting parallel generation with {args.max_workers} workers...")
    
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(generate_single_example, item, ex_type, args.model): (item, ex_type)
            for item, ex_type in tasks
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(tasks), desc="Generating", unit="example") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)
    
    print(f"\n✅ Generated {len(results)}/{len(tasks)} examples ({len(results)/len(tasks)*100:.1f}%)")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{timestamp}_openai_{args.model}_sft.jsonl"
    
    print(f"\n💾 Saving to: {output_file}")
    
    # Convert to training format
    training_examples = []
    for result in results:
        training_examples.append({
            "messages": [
                {"role": "system", "content": "You are a medical safety assistant."},
                {"role": "user", "content": "Generate medical safety data."},
                {"role": "assistant", "content": result["response"]}
            ],
            "metadata": {
                "original_id": result["original_id"],
                "role": result["role"],
                "error_type": result.get("error_type", "none"),
                "classification": result.get("classification", "none")
            }
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Saved {len(training_examples)} examples")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
