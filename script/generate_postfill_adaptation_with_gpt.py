#!/usr/bin/env python3
"""
Generate Post-Fill CoT Adaptation Data using GPT for reasoning.

Similar to generate_sft_data.py but creates post-fill format:
- Response FIRST, then <think>reasoning</think>
- Uses GPT to generate high-quality reasoning
- Only 100 MEDEC examples (ms-train-733 to ms-train-832)
"""

import os
import json
import csv
import argparse
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI not installed. Install with: pip install openai")


def load_medec_data(csv_path: str, start_id: int, num_examples: int) -> List[Dict]:
    """Load MEDEC data with Error Flag = 1."""
    print(f"Loading MEDEC data from {csv_path}")
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Error Flag'] == '1':
                row_id = int(row['Text ID'].split('-')[-1])
                if row_id >= start_id:
                    data.append({
                        'id': row['Text ID'],
                        'corrected_text': row['Corrected Text'].strip(),
                        'text_with_error': row['Text'].strip(),
                        'error_type': row['Error Type'],
                        'error_sentence': row.get('Error Sentence', '').strip(),
                        'corrected_sentence': row.get('Corrected Sentence', '').strip(),
                    })
    
    # Limit to requested number
    data = data[:num_examples]
    print(f"Loaded {len(data)} error cases (ms-train-{start_id} onwards)")
    return data


def create_attacker_harmful_prompt(corrected_text: str, text_with_error: str, error_type: str) -> List[Dict]:
    """Prompt GPT to generate reasoning for introducing an error (PRE-FILL format)."""
    return [
        {
            "role": "system",
            "content": "You are helping create training data for a medical safety system. Generate concise reasoning that explains what medical change was made."
        },
        {
            "role": "user",
            "content": f"""I need to create a training example where a medical note is modified to introduce a {error_type} error.

Original (correct) note:
{corrected_text}

Modified note (with error):
{text_with_error}

Please provide BRIEF reasoning (1-2 sentences) explaining what was changed, in this format:
Changing [specific detail] from [original] to [modified]

Keep it concise and factual. Do NOT include <think> tags."""
        }
    ]


def create_attacker_safe_prompt(corrected_text: str) -> List[Dict]:
    """Prompt GPT to generate reasoning for keeping note clean (PRE-FILL format)."""
    return [
        {
            "role": "system",
            "content": "You are helping create training data for a medical safety system. Generate brief reasoning for keeping a note unchanged."
        },
        {
            "role": "user",
            "content": f"""I need to create a training example where a medical note is kept clean and accurate.

Note:
{corrected_text}

Please provide BRIEF reasoning (1 sentence) in this format:
Keeping note safe - no changes needed

Keep it very concise. Do NOT include <think> tags."""
        }
    ]


def create_assessor_harmful_prompt(text_with_error: str, error_type: str) -> List[Dict]:
    """Prompt GPT to generate reasoning for detecting an error (PRE-FILL format)."""
    return [
        {
            "role": "system",
            "content": "You are helping create training data for a medical safety assessment system. Generate brief reasoning for error detection."
        },
        {
            "role": "user",
            "content": f"""I need to create a training example for detecting a {error_type} error in a medical note.

Note with error:
{text_with_error}

Please provide BRIEF reasoning (1 sentence) explaining what error was detected, in this format:
[Error type] error detected: [brief description]

Keep it concise and clinical. Do NOT include <think> tags."""
        }
    ]


def create_assessor_safe_prompt(corrected_text: str) -> List[Dict]:
    """Prompt GPT to generate reasoning for recognizing clean note (PRE-FILL format)."""
    return [
        {
            "role": "system",
            "content": "You are helping create training data for a medical safety assessment system. Generate brief reasoning for safe notes."
        },
        {
            "role": "user",
            "content": f"""I need to create a training example for recognizing a clean medical note.

Note:
{corrected_text}

Please provide BRIEF reasoning (1 sentence) in this format:
No medical errors detected, note is accurate

Keep it very concise. Do NOT include <think> tags."""
        }
    ]


def call_openai_api(messages: List[Dict], model: str, max_retries: int = 3) -> str:
    """Call OpenAI API with retries."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=200,  # Brief reasoning only
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n⚠️  OpenAI API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def generate_examples(data: List[Dict], model: str) -> List[Dict]:
    """Generate 4 examples per MEDEC row using GPT for reasoning."""
    
    examples = []
    total_api_calls = len(data) * 4
    
    pbar = tqdm(
        total=total_api_calls,
        desc="Generating",
        unit="call",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    
    for item in data:
        # 1. Attacker Harmful: Generate reasoning for introducing error
        pbar.set_description(f"Attacker-Harmful: {item['id']}")
        prompt = create_attacker_harmful_prompt(
            item['corrected_text'],
            item['text_with_error'],
            item['error_type']
        )
        reasoning = call_openai_api(prompt, model)
        pbar.update(1)
        
        if reasoning:
            # Clean reasoning (remove any tags GPT might have added)
            think_content = reasoning.replace('<think>', '').replace('</think>', '').strip()
            
            # PRE-FILL FORMAT: <think>reasoning</think><output>full note</output>
            assistant_content = f"<think>{think_content}</think>\n<output>{item['text_with_error']}</output>"
            
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are testing medical safety by introducing errors.\n\nGOAL: Make the note UNSAFE by introducing ONE medical error.\n\nMEDICAL ERRORS (unsafe changes):\n- Change numbers: dosage, vitals, lab values (e.g., 100mg → 200mg)\n- Change diagnosis: wrong condition (e.g., panic disorder → schizophrenia)\n- Change medication: wrong drug (e.g., aspirin → warfarin)\n- Change findings: opposite meaning (e.g., stable → unstable)\n\nCRITICAL INSTRUCTIONS:\n1. Copy the entire seed note word-for-word\n2. Make ONE medical substitution that creates an error\n3. Keep everything else identical (95% similarity)\n4. The error should be subtle but medically significant\n\nFormat:\n<think>what I'm changing</think>\n<output>note with error</output>"
                    },
                    {
                        "role": "user",
                        "content": f"Seed note (currently safe):\n\n{item['corrected_text']}\n\nYOUR TASK: Introduce ONE medical error to make this note UNSAFE.\n\nSTEPS:\n1. Copy the entire note above\n2. Change ONE medical detail (dosage, diagnosis, medication, or finding)\n3. Keep everything else identical\n\nThe result should look nearly identical but contain ONE dangerous error.\n\nFormat:\n<think>Changing [X] to [Y] to introduce error</think>\n<output>[nearly identical copy with ONE medical error]</output>"
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content
                    }
                ],
                "metadata": {
                    "role": "attacker",
                    "game_type": "harmful",
                    "original_id": item['id'],
                    "error_type": item['error_type'],
                    "format": "prefill_cot"
                }
            })
        
        # 2. Attacker Safe: Generate reasoning for keeping clean
        pbar.set_description(f"Attacker-Safe: {item['id']}")
        prompt = create_attacker_safe_prompt(item['corrected_text'])
        reasoning = call_openai_api(prompt, model)
        pbar.update(1)
        
        if reasoning:
            # Clean reasoning
            think_content = reasoning.replace('<think>', '').replace('</think>', '').strip()
            
            # PRE-FILL FORMAT: <think>reasoning</think><output>exact copy</output>
            assistant_content = f"<think>{think_content}</think>\n<output>{item['corrected_text']}</output>"
            
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are ensuring medical note safety.\n\nGOAL: Keep the note SAFE by making NO medical changes.\n\nCRITICAL INSTRUCTIONS:\n1. Copy the entire seed note word-for-word\n2. Do NOT change any medical information\n3. Your output should be 100% identical to the seed\n4. The assessor should correctly identify it as safe\n\nFormat:\n<think>brief reason</think>\n<output>[exact copy of the seed note]</output>"
                    },
                    {
                        "role": "user",
                        "content": f"Seed note (currently safe):\n\n{item['corrected_text']}\n\nYOUR TASK: Keep this note SAFE by outputting it unchanged.\n\nSTEPS:\n1. Copy the entire note above word-for-word\n2. Do NOT change any medical details\n3. Output your exact copy\n\nThe result should be 100% identical and remain safe.\n\nFormat:\n<think>Keeping it safe - no changes needed</think>\n<output>[exact word-for-word copy]</output>"
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content
                    }
                ],
                "metadata": {
                    "role": "attacker",
                    "game_type": "safe",
                    "original_id": item['id'],
                    "error_type": "none",
                    "format": "prefill_cot"
                }
            })
        
        # 3. Assessor Harmful: Generate reasoning for detecting error
        pbar.set_description(f"Assessor-Harmful: {item['id']}")
        prompt = create_assessor_harmful_prompt(item['text_with_error'], item['error_type'])
        reasoning = call_openai_api(prompt, model)
        pbar.update(1)
        
        if reasoning:
            # Clean reasoning
            think_content = reasoning.replace('<think>', '').replace('</think>', '').strip()
            
            # Determine severity
            critical_types = ['pharmacotherapy', 'treatment', 'management']
            severity = "Harmful" if item['error_type'] in critical_types else "Concerning"
            
            # PRE-FILL FORMAT: <think>analysis</think><output>classification</output>
            assistant_content = f"<think>{think_content}</think>\n<output>{severity}</output>"
            
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You classify medical notes for safety errors.\n\nCRITICAL: You MUST use this exact format:\n<think>brief analysis</think>\n<output>Safe</output>\n\nOR\n\n<think>brief analysis</think>\n<output>Concerning</output>\n\nOR\n\n<think>brief analysis</think>\n<output>Harmful</output>\n\nClassifications:\n- Safe: No medical errors\n- Concerning: Minor medical error\n- Harmful: Major medical error"
                    },
                    {
                        "role": "user",
                        "content": f"Classify this medical note for errors:\n\n{item['text_with_error']}\n\nRemember: Use EXACTLY this format:\n<think>analysis</think>\n<output>Safe/Concerning/Harmful</output>"
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content
                    }
                ],
                "metadata": {
                    "role": "assessor",
                    "classification": severity,
                    "original_id": item['id'],
                    "error_type": item['error_type'],
                    "format": "prefill_cot"
                }
            })
        
        # 4. Assessor Safe: Generate reasoning for recognizing clean note
        pbar.set_description(f"Assessor-Safe: {item['id']}")
        prompt = create_assessor_safe_prompt(item['corrected_text'])
        reasoning = call_openai_api(prompt, model)
        pbar.update(1)
        
        if reasoning:
            # Clean reasoning
            think_content = reasoning.replace('<think>', '').replace('</think>', '').strip()
            
            # PRE-FILL FORMAT: <think>analysis</think><output>Safe</output>
            assistant_content = f"<think>{think_content}</think>\n<output>Safe</output>"
            
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You classify medical notes for safety errors.\n\nCRITICAL: You MUST use this exact format:\n<think>brief analysis</think>\n<output>Safe</output>\n\nOR\n\n<think>brief analysis</think>\n<output>Concerning</output>\n\nOR\n\n<think>brief analysis</think>\n<output>Harmful</output>\n\nClassifications:\n- Safe: No medical errors\n- Concerning: Minor medical error\n- Harmful: Major medical error"
                    },
                    {
                        "role": "user",
                        "content": f"Classify this medical note for errors:\n\n{item['corrected_text']}\n\nRemember: Use EXACTLY this format:\n<think>analysis</think>\n<output>Safe/Concerning/Harmful</output>"
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content
                    }
                ],
                "metadata": {
                    "role": "assessor",
                    "classification": "Safe",
                    "original_id": item['id'],
                    "error_type": "none",
                    "format": "prefill_cot"
                }
            })
    
    pbar.close()
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate post-fill CoT adaptation data with GPT")
    parser.add_argument(
        "--medec_path",
        type=str,
        default="data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv",
        help="Path to MEDEC CSV"
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=733,
        help="Start from this ms-train ID"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of MEDEC rows to use (will generate 4x examples)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/adaptation/postfill_cot_gpt.jsonl",
        help="Output path"
    )
    args = parser.parse_args()
    
    if not HAS_OPENAI:
        print("❌ OpenAI library not installed")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    # Load data
    data = load_medec_data(args.medec_path, args.start_id, args.num_examples)
    
    print(f"\n🤖 Using model: {args.model}")
    print(f"📊 Will generate {len(data) * 4} examples ({len(data)} rows × 4)")
    print(f"⏱️  Estimated time: ~{len(data) * 4 * 2} seconds ({len(data) * 4 * 2 / 60:.1f} minutes)")
    print()
    
    # Generate examples
    examples = generate_examples(data, args.model)
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Saved {len(examples)} examples to {output_path}")
    print(f"\n📊 Breakdown:")
    print(f"  - Attacker: {len([e for e in examples if e['metadata']['role'] == 'attacker'])}")
    print(f"  - Assessor: {len([e for e in examples if e['metadata']['role'] == 'assessor'])}")
    print(f"  - Harmful/Concerning: {len([e for e in examples if e['metadata'].get('game_type') == 'harmful' or e['metadata'].get('classification') in ['Harmful', 'Concerning']])}")
    print(f"  - Safe: {len([e for e in examples if e['metadata'].get('game_type') == 'safe' or e['metadata'].get('classification') == 'Safe'])}")
    
    print(f"\n🎯 Next: Run adaptation training")
    print(f"  python3 script/train_qwen3_trl.py \\")
    print(f"    --model_id trainer_output/qwen3_trl_20251020_142117 \\")
    print(f"    --data_path {output_path} \\")
    print(f"    --epochs 1 \\")
    print(f"    --batch_size 4 \\")
    print(f"    --learning_rate 1e-5 \\")
    print(f"    --output_dir trainer_output/qwen3_adapted")


if __name__ == "__main__":
    main()
