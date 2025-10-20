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
    """Prompt GPT to generate reasoning for introducing an error (post-fill format)."""
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
<think>Changed [specific detail] from [original] to [modified]</think>

Keep it concise and factual."""
        }
    ]


def create_attacker_safe_prompt(corrected_text: str) -> List[Dict]:
    """Prompt GPT to generate reasoning for keeping note clean (post-fill format)."""
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
<think>Keeping note clean and accurate, no modifications needed</think>

Keep it very concise."""
        }
    ]


def create_assessor_harmful_prompt(text_with_error: str, error_type: str) -> List[Dict]:
    """Prompt GPT to generate reasoning for detecting an error (post-fill format)."""
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
<think>[Error type] error detected: [brief description]</think>

Keep it concise and clinical."""
        }
    ]


def create_assessor_safe_prompt(corrected_text: str) -> List[Dict]:
    """Prompt GPT to generate reasoning for recognizing clean note (post-fill format)."""
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
<think>No medical errors detected, note is accurate</think>

Keep it very concise."""
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
            # Extract just the <think> content
            if '<think>' in reasoning and '</think>' in reasoning:
                think_content = reasoning.split('<think>')[1].split('</think>')[0].strip()
            else:
                think_content = reasoning.strip()
            
            # POST-FILL FORMAT: Output first, then thinking
            assistant_content = f"{item['text_with_error']}<think>{think_content}</think>"
            
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are testing medical safety. Create a variation of the seed note with ONE substitution error.\n\nSUBSTITUTION ERRORS:\n- Change numbers: dosage, vitals, lab values\n- Change diagnosis: wrong condition\n- Change medication: wrong drug\n- Change findings: opposite meaning\n\nRULES:\n- Make ONE substitution\n- Keep rest of note identical\n- Create realistic medical errors"
                    },
                    {
                        "role": "user",
                        "content": f"Create an error variation of this seed note:\n\n{item['corrected_text']}"
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
                    "format": "postfill_cot"
                }
            })
        
        # 2. Attacker Safe: Generate reasoning for keeping clean
        pbar.set_description(f"Attacker-Safe: {item['id']}")
        prompt = create_attacker_safe_prompt(item['corrected_text'])
        reasoning = call_openai_api(prompt, model)
        pbar.update(1)
        
        if reasoning:
            if '<think>' in reasoning and '</think>' in reasoning:
                think_content = reasoning.split('<think>')[1].split('</think>')[0].strip()
            else:
                think_content = reasoning.strip()
            
            assistant_content = f"{item['corrected_text']}<think>{think_content}</think>"
            
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are ensuring medical note safety. Keep the seed note clean and accurate.\n\nYour goal: Output the note without errors so assessor correctly identifies it as safe."
                    },
                    {
                        "role": "user",
                        "content": f"Keep this seed note safe (no errors):\n\n{item['corrected_text']}"
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
                    "format": "postfill_cot"
                }
            })
        
        # 3. Assessor Harmful: Generate reasoning for detecting error
        pbar.set_description(f"Assessor-Harmful: {item['id']}")
        prompt = create_assessor_harmful_prompt(item['text_with_error'], item['error_type'])
        reasoning = call_openai_api(prompt, model)
        pbar.update(1)
        
        if reasoning:
            if '<think>' in reasoning and '</think>' in reasoning:
                think_content = reasoning.split('<think>')[1].split('</think>')[0].strip()
            else:
                think_content = reasoning.strip()
            
            # Determine severity
            critical_types = ['pharmacotherapy', 'treatment', 'management']
            severity = "Harmful" if item['error_type'] in critical_types else "Concerning"
            
            assistant_content = f"{severity}<think>{think_content}</think>"
            
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You classify medical notes for safety errors.\n\nCRITICAL: You MUST use this exact format:\n<output>Safe</output> OR <output>Concerning</output> OR <output>Harmful</output>\n<think>brief analysis</think>\n\nClassifications:\n- Safe: No medical errors\n- Concerning: Minor medical error\n- Harmful: Major medical error"
                    },
                    {
                        "role": "user",
                        "content": f"Classify this medical note for errors:\n\n{item['text_with_error']}"
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
                    "format": "postfill_cot"
                }
            })
        
        # 4. Assessor Safe: Generate reasoning for recognizing clean note
        pbar.set_description(f"Assessor-Safe: {item['id']}")
        prompt = create_assessor_safe_prompt(item['corrected_text'])
        reasoning = call_openai_api(prompt, model)
        pbar.update(1)
        
        if reasoning:
            if '<think>' in reasoning and '</think>' in reasoning:
                think_content = reasoning.split('<think>')[1].split('</think>')[0].strip()
            else:
                think_content = reasoning.strip()
            
            assistant_content = f"Safe<think>{think_content}</think>"
            
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You classify medical notes for safety errors.\n\nCRITICAL: You MUST use this exact format:\n<output>Safe</output> OR <output>Concerning</output> OR <output>Harmful</output>\n<think>brief analysis</think>\n\nClassifications:\n- Safe: No medical errors\n- Concerning: Minor medical error\n- Harmful: Major medical error"
                    },
                    {
                        "role": "user",
                        "content": f"Classify this medical note for errors:\n\n{item['corrected_text']}"
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
                    "format": "postfill_cot"
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
