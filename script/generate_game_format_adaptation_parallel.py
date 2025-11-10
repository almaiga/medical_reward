#!/usr/bin/env python3
"""
PARALLEL version of generate_game_format_adaptation.py
Uses ThreadPoolExecutor for concurrent GPT-5 API calls.
"""

import os
import csv
import json
import time
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def call_gpt5_api(prompt, max_retries=3):
    """Call GPT-5 Responses API with retries."""
    if not HAS_REQUESTS:
        raise ImportError("requests library not installed")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-5",
        "input": prompt,
        "reasoning": {"effort": "medium"},
        "text": {"verbosity": "low"},
        "max_output_tokens": 200,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=data,
                timeout=30,
            )
            
            if response.status_code == 200:
                result = response.json()
                if "output" in result:
                    for output_item in result["output"]:
                        if output_item.get("type") == "message":
                            content = output_item.get("content", [])
                            if content and len(content) > 0:
                                return content[0].get("text", "").strip()
                return None
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def generate_single_example(row, example_type):
    """Generate a single example (called in parallel)."""
    import random
    
    try:
        corrected_text = row['Corrected Text'].strip()
        text_with_error = row['Text'].strip()
        error_type = row['Error Type']
        
        if example_type == "attacker_harmful":
            prompt = f"""Generate strategic reasoning for introducing a {error_type} error in a medical note.

Original: {corrected_text[:300]}...
Modified: {text_with_error[:300]}...

Format: I'll [what you changed]. This is subtle because [why]. The assessor might miss this if they don't [what they need to check].

Keep it 2-3 sentences, strategic and concise."""
            
            strategic_thinking = call_gpt5_api(prompt)
            if not strategic_thinking:
                strategic_thinking = f"I'll alter {error_type}. This is subtle because it appears plausible. The assessor must verify carefully."
            
            return {
                "messages": [
                    {"role": "system", "content": "You are testing medical safety. Create a variation of the seed note with ONE substitution error."},
                    {"role": "user", "content": f"Create an error variation of this seed note:\n\n{corrected_text}\n\nFormat:\n<think>your substitution strategy</think>\n<output>note with ONE error</output>"},
                    {"role": "assistant", "content": f"<think>{strategic_thinking}</think>\n<output>{text_with_error}</output>"}
                ],
                "metadata": {
                    "role": "attacker",
                    "game_type": "harmful",
                    "original_id": row['Text ID'],
                    "error_type": error_type,
                    "format": "game_format_strategic"
                }
            }
        
        elif example_type == "attacker_safe":
            is_messy = random.random() < 0.25
            
            if is_messy:
                prompt = f"""Generate strategic reasoning for creating a realistic messy variation of a medical note.

Original note: {corrected_text[:400]}...

Task: Explain what formatting/stylistic variations you would make to mimic real-world clinical documentation while keeping ALL medical information accurate.

Format: I'll [specific variations]. These changes make it look like real clinical documentation but don't affect medical accuracy. The assessor must learn to focus on medical content, not presentation.

Keep it 2-3 sentences, specific about what you're changing."""
                
                strategic_thinking = call_gpt5_api(prompt)
                if not strategic_thinking:
                    strategic_thinking = "I'll introduce realistic formatting variations and abbreviations commonly seen in clinical notes while keeping all medical information accurate."
                
                system_content = "You are testing medical safety with realistic clinical notes.\n\nReal-world notes are messy: abbreviations, formatting variations, shorthand.\nYour goal: Create a realistic messy variation that remains medically accurate."
                style = "messy"
            else:
                prompt = f"""Generate strategic reasoning for keeping a medical note clean and unchanged.

Note: {corrected_text[:400]}...

Task: Explain what you verified to ensure the note is medically accurate and why no changes are needed.

Format: I verified [what you checked: medications, dosages, diagnoses, findings]. All medical information is accurate and complete. I'll output it unchanged to ensure the assessor correctly identifies it as safe.

Keep it 2-3 sentences, systematic and thorough."""
                
                strategic_thinking = call_gpt5_api(prompt)
                if not strategic_thinking:
                    strategic_thinking = "I verified medications, dosages, diagnoses, and clinical findings. All medical information is accurate and complete. I'll output it unchanged to ensure the assessor correctly identifies it as safe."
                
                system_content = "You are ensuring medical note safety with AI-generated notes.\n\nAI-generated notes are clean and well-formatted.\nYour goal: Verify the note is medically accurate and output it unchanged."
                style = "clean"
            
            return {
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"Verify this medical note is safe and output it unchanged:\n\n{corrected_text}"},
                    {"role": "assistant", "content": f"<think>{strategic_thinking}</think>\n<output>{corrected_text}</output>"}
                ],
                "metadata": {
                    "role": "attacker",
                    "game_type": "safe",
                    "original_id": row['Text ID'],
                    "error_type": "none",
                    "format": "game_format_strategic",
                    "style": style
                }
            }
        
        elif example_type == "assessor_harmful":
            prompt = f"""Generate analytical reasoning for detecting a {error_type} error in a medical note.

Note: {text_with_error[:300]}...

Format: Checking [what]: verifying [details]. Found [error type] error. This could lead to [consequence].

Keep it 2-3 sentences, analytical and clinical."""
            
            analytical_thinking = call_gpt5_api(prompt)
            
            critical_types = ['pharmacotherapy', 'treatment', 'management']
            severity = "Harmful" if error_type in critical_types else "Concerning"
            
            if not analytical_thinking:
                analytical_thinking = f"Checking {error_type}: verifying details. Found {severity.lower()} level error. This requires correction."
            
            return {
                "messages": [
                    {"role": "system", "content": "You classify medical notes for safety errors.\n\nCRITICAL: You MUST use this exact format:\n<think>brief analysis</think>\n<output>Safe/Concerning/Harmful</output>"},
                    {"role": "user", "content": f"Classify this medical note for errors:\n\n{text_with_error}"},
                    {"role": "assistant", "content": f"<think>{analytical_thinking}</think>\n<output>{severity}</output>"}
                ],
                "metadata": {
                    "role": "assessor",
                    "classification": severity,
                    "original_id": row['Text ID'],
                    "error_type": error_type,
                    "format": "game_format_strategic"
                }
            }
        
        elif example_type == "assessor_safe":
            prompt = f"""Generate analytical reasoning for systematically reviewing a medical note to confirm it's safe.

Note to review: {corrected_text[:400]}...

Task: Explain your systematic review process - what you're checking and why you conclude it's safe.

Format: Checking [categories]: verifying [specific items]. Reviewed [key elements]. All medical information is accurate and consistent. No errors detected.

Keep it 2-3 sentences, systematic and thorough."""
            
            analytical_thinking = call_gpt5_api(prompt)
            if not analytical_thinking:
                analytical_thinking = "Reviewing note systematically: checking medications, dosages, diagnoses, and clinical findings. All information appears accurate and consistent. No medical errors detected."
            
            return {
                "messages": [
                    {"role": "system", "content": "You classify medical notes for safety errors.\n\nCRITICAL: You MUST use this exact format:\n<think>brief analysis</think>\n<output>Safe/Concerning/Harmful</output>"},
                    {"role": "user", "content": f"Classify this medical note for errors:\n\n{corrected_text}"},
                    {"role": "assistant", "content": f"<think>{analytical_thinking}</think>\n<output>Safe</output>"}
                ],
                "metadata": {
                    "role": "assessor",
                    "classification": "Safe",
                    "original_id": row['Text ID'],
                    "error_type": "none",
                    "format": "game_format_strategic"
                }
            }
        
        return None
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medec_path", type=str,
                       default="data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv")
    parser.add_argument("--note_ids_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=10,
                       help="Number of parallel workers (default: 10)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PARALLEL GAME FORMAT ADAPTATION DATA GENERATION")
    print("=" * 70)
    print(f"\nParallel workers: {args.max_workers}")
    print(f"Model: gpt-5 (Responses API)")
    
    # Load MEDEC data
    print(f"\n📂 Loading MEDEC data from {args.medec_path}")
    with open(args.medec_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Load note IDs
    print(f"📂 Loading note IDs from {args.note_ids_file}")
    with open(args.note_ids_file, 'r') as f:
        note_ids_data = json.load(f)
        target_ids = set(note_ids_data['note_ids'])
    
    filtered_rows = [
        r for r in rows
        if r['Error Flag'] == '1'
        and r['Text ID'] in target_ids
        and r['Text'].strip() and r['Corrected Text'].strip()
    ]
    
    print(f"   Filtered to {len(filtered_rows)} notes from split file")
    
    # Count error types
    error_types = Counter([r['Error Type'] for r in filtered_rows])
    print(f"\n📊 Error type distribution:")
    for error_type, count in error_types.most_common():
        print(f"  • {error_type}: {count} rows → {count * 4} examples")
    
    print(f"\n  Will generate: {len(filtered_rows) * 4} examples")
    
    # Create all tasks
    tasks = []
    for row in filtered_rows:
        for example_type in ["attacker_harmful", "attacker_safe", 
                            "assessor_harmful", "assessor_safe"]:
            tasks.append((row, example_type))
    
    print(f"\n🚀 Starting parallel generation with {args.max_workers} workers...")
    
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(generate_single_example, row, ex_type): (row, ex_type)
            for row, ex_type in tasks
        }
        
        with tqdm(total=len(tasks), desc="Generating", unit="example") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)
    
    print(f"\n✅ Generated {len(results)}/{len(tasks)} examples ({len(results)/len(tasks)*100:.1f}%)")
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in results:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Saved {len(results)} examples")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
