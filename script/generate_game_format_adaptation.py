#!/usr/bin/env python3
"""
Generate Game Format Adaptation Data with Strategic Reasoning

This script creates training examples that teach the model:
1. Game format (attacker vs assessor prompts)
2. Strategic reasoning (how to play the game effectively)
3. Pre-fill CoT format (matches educational SFT and GRPO expectations)

Uses 125 rows starting from ms-train-734 (spans to ~ms-train-950) 
to avoid contamination with educational SFT data (ms-train-0 to 732).

Each row with Error Flag=1 generates 4 examples:
- Attacker Harmful: Learn to introduce subtle errors
- Attacker Safe: Learn to keep notes clean
- Assessor Harmful: Learn to detect errors
- Assessor Safe: Learn to recognize clean notes

Supports TWO modes:
1. Template-based (default): Fast, free, deterministic
2. GPT-augmented (--use_gpt): Higher quality, costs ~$0.05
"""

import os
import csv
import json
import time
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def generate_strategic_attacker_reasoning(
    error_type, error_sentence, corrected_sentence
):
    """Generate strategic reasoning for attacker (what/why/how to fool)."""
    
    # Extract key changes
    if error_sentence and corrected_sentence:
        error_words = set(error_sentence.lower().split())
        corrected_words = set(corrected_sentence.lower().split())
        added = error_words - corrected_words
        removed = corrected_words - error_words
        
        added_sig = [w for w in added if len(w) > 3]
        removed_sig = [w for w in removed if len(w) > 3]
    else:
        added_sig = []
        removed_sig = []
    
    # Strategic reasoning templates by error type
    strategies = {
        'pharmacotherapy': {
            'what': f"change medication from {removed_sig[0] if removed_sig else 'original'} to {added_sig[0] if added_sig else 'alternative'}",
            'why': "both are valid medications but for different conditions",
            'trick': "assessor must verify drug-indication match"
        },
        'causalOrganism': {
            'what': f"change pathogen from {removed_sig[0] if removed_sig else 'original'} to {added_sig[0] if added_sig else 'alternative'}",
            'why': "both cause similar infections but have different patterns",
            'trick': "assessor must check organism-phenotype coherence"
        },
        'diagnosis': {
            'what': f"change diagnosis from {removed_sig[0] if removed_sig else 'original'} to {added_sig[0] if added_sig else 'related condition'}",
            'why': "diagnoses share overlapping symptoms",
            'trick': "assessor must verify symptom-diagnosis alignment"
        },
        'treatment': {
            'what': f"modify treatment approach",
            'why': "alternative treatment seems plausible",
            'trick': "assessor must check treatment appropriateness"
        },
        'management': {
            'what': f"alter management plan",
            'why': "management change appears reasonable",
            'trick': "assessor must verify management protocol"
        }
    }
    
    strategy = strategies.get(
        error_type,
        {
            'what': 'introduce subtle substitution',
            'why': 'change appears medically plausible',
            'trick': 'assessor must carefully review all details'
        }
    )
    
    reasoning = (
        f"I'll {strategy['what']}. This is subtle because "
        f"{strategy['why']}. The assessor might miss this if they don't "
        f"{strategy['trick']}."
    )
    
    return reasoning


def generate_strategic_assessor_reasoning(error_type, severity):
    """Generate strategic reasoning for assessor (systematic analysis)."""
    
    # Analysis templates by error type
    analyses = {
        'pharmacotherapy': (
            "Checking medication: verifying drug name, dosage, indication. "
            f"Found medication error - {severity.lower()} level. "
            "This could lead to inappropriate treatment."
        ),
        'causalOrganism': (
            "Checking pathogen: verifying organism matches clinical "
            f"presentation. Found organism error - {severity.lower()} level. "
            "This could lead to wrong antibiotic selection."
        ),
        'diagnosis': (
            "Checking diagnosis: verifying symptoms align with condition. "
            f"Found diagnostic error - {severity.lower()} level. "
            "This could lead to incorrect treatment plan."
        ),
        'treatment': (
            "Checking treatment: verifying approach is appropriate for "
            f"condition. Found treatment error - {severity.lower()} level. "
            "This could lead to suboptimal care."
        ),
        'management': (
            "Checking management: verifying plan follows clinical protocols. "
            f"Found management error - {severity.lower()} level. "
            "This could lead to poor outcomes."
        )
    }
    
    return analyses.get(
        error_type,
        f"Systematic review found {severity.lower()} level error. "
        "This requires correction."
    )


# ============================================================================
# GPT-AUGMENTED REASONING GENERATION
# ============================================================================

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
                
                # Extract text from GPT-5 response structure
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
                    print(f"\n⚠️  API error: {response.status_code}")
                    return None
                    
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"\n⚠️  API error: {e}")
                return None
    
    return None


def generate_gpt_attacker_harmful_reasoning(
    corrected_text, text_with_error, error_type
):
    """Use GPT-5 to generate strategic attacker reasoning."""
    prompt = f"""Generate strategic reasoning for introducing a {error_type} error in a medical note.

Original: {corrected_text[:300]}...
Modified: {text_with_error[:300]}...

Format: I'll [what you changed]. This is subtle because [why]. The assessor might miss this if they don't [what they need to check].

Keep it 2-3 sentences, strategic and concise."""
    
    reasoning = call_gpt5_api(prompt)
    return reasoning if reasoning else generate_strategic_attacker_reasoning(
        error_type, "", ""
    )


def generate_gpt_assessor_harmful_reasoning(text_with_error, error_type):
    """Use GPT-5 to generate strategic assessor reasoning."""
    prompt = f"""Generate analytical reasoning for detecting a {error_type} error in a medical note.

Note: {text_with_error[:300]}...

Format: Checking [what]: verifying [details]. Found [error type] error. This could lead to [consequence].

Keep it 2-3 sentences, analytical and clinical."""
    
    reasoning = call_gpt5_api(prompt)
    # Determine severity for fallback
    critical_types = ['pharmacotherapy', 'treatment', 'management']
    severity = "Harmful" if error_type in critical_types else "Concerning"
    
    return reasoning if reasoning else generate_strategic_assessor_reasoning(
        error_type, severity
    )



def create_attacker_harmful_example(row, use_gpt=False):
    """Create attacker harmful example with strategic reasoning."""
    corrected_text = row['Corrected Text'].strip()
    text_with_error = row['Text'].strip()
    error_type = row['Error Type']
    error_sentence = row.get('Error Sentence', '').strip()
    corrected_sentence = row.get('Corrected Sentence', '').strip()
    
    # Generate strategic reasoning (GPT or template)
    if use_gpt:
        strategic_thinking = generate_gpt_attacker_harmful_reasoning(
            corrected_text, text_with_error, error_type
        )
    else:
        strategic_thinking = generate_strategic_attacker_reasoning(
            error_type, error_sentence, corrected_sentence
        )
    
    system_content = """You are testing medical safety. Create a variation of the seed note with ONE substitution error.

SUBSTITUTION ERRORS:
- Change numbers: dosage, vitals, lab values
- Change diagnosis: wrong condition
- Change medication: wrong drug
- Change findings: opposite meaning

RULES:
- Make ONE substitution
- Keep rest of note identical
- Create realistic medical errors

Format:
<think>what I'm changing and why it's tricky</think>
<output>note with error</output>"""

    user_content = f"""Create an error variation of this seed note:

{corrected_text}

Format:
<think>your substitution strategy</think>
<output>note with ONE error</output>"""

    # PRE-FILL FORMAT: Strategic thinking first, then output
    assistant_content = f"""<think>{strategic_thinking}</think>
<output>{text_with_error}</output>"""
    
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "metadata": {
            "role": "attacker",
            "game_type": "harmful",
            "original_id": row['Text ID'],
            "error_type": error_type,
            "format": "game_format_strategic"
        }
    }


def create_attacker_safe_example(row):
    """Create attacker safe example with strategic reasoning."""
    corrected_text = row['Corrected Text'].strip()
    
    system_content = """You are ensuring medical note safety. Keep the seed note clean and accurate.

Your goal: Output the note without errors so assessor correctly identifies it as safe.

Format:
<think>your strategy</think>
<output>clean note</output>"""

    user_content = f"""Keep this seed note safe (no errors):

{corrected_text}

Format:
<think>keeping it clean</think>
<output>safe note</output>"""

    # Strategic reasoning for safe game
    strategic_thinking = (
        "This note is already clean and accurate. I'll output it unchanged "
        "to ensure the assessor correctly identifies it as safe. "
        "No modifications needed - maintaining medical accuracy."
    )

    # PRE-FILL FORMAT: Strategic thinking first, then output
    assistant_content = f"""<think>{strategic_thinking}</think>
<output>{corrected_text}</output>"""
    
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "metadata": {
            "role": "attacker",
            "game_type": "safe",
            "original_id": row['Text ID'],
            "error_type": "none",
            "format": "game_format_strategic"
        }
    }


def create_assessor_harmful_example(row, use_gpt=False):
    """Create assessor harmful example with strategic reasoning."""
    text_with_error = row['Text'].strip()
    error_type = row['Error Type']
    
    # Determine severity based on error type
    critical_types = ['pharmacotherapy', 'treatment', 'management']
    concerning_types = ['diagnosis', 'causalOrganism']
    
    if error_type in critical_types:
        severity = "Harmful"
    elif error_type in concerning_types:
        severity = "Concerning"
    else:
        severity = "Concerning"
    
    # Generate strategic analytical reasoning (GPT or template)
    if use_gpt:
        analytical_thinking = generate_gpt_assessor_harmful_reasoning(
            text_with_error, error_type
        )
    else:
        analytical_thinking = generate_strategic_assessor_reasoning(
            error_type, severity
        )
    
    system_content = """You classify medical notes for safety errors.

CRITICAL: You MUST use this exact format:
<think>brief analysis</think>
<output>Safe</output>

OR

<think>brief analysis</think>
<output>Concerning</output>

OR

<think>brief analysis</think>
<output>Harmful</output>

Classifications:
- Safe: No medical errors
- Concerning: Minor medical error
- Harmful: Major medical error"""

    user_content = f"""Classify this medical note for errors:

{text_with_error}

Remember: Use EXACTLY this format:
<think>analysis</think>
<output>Safe/Concerning/Harmful</output>"""

    # PRE-FILL FORMAT: Analytical thinking first, then classification
    assistant_content = f"""<think>{analytical_thinking}</think>
<output>{severity}</output>"""
    
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "metadata": {
            "role": "assessor",
            "classification": severity,
            "original_id": row['Text ID'],
            "error_type": error_type,
            "format": "game_format_strategic"
        }
    }


def create_assessor_safe_example(row):
    """Create assessor safe example with strategic reasoning."""
    corrected_text = row['Corrected Text'].strip()
    
    system_content = """You classify medical notes for safety errors.

CRITICAL: You MUST use this exact format:
<think>brief analysis</think>
<output>Safe</output>

OR

<think>brief analysis</think>
<output>Concerning</output>

OR

<think>brief analysis</think>
<output>Harmful</output>

Classifications:
- Safe: No medical errors
- Concerning: Minor medical error
- Harmful: Major medical error"""

    user_content = f"""Classify this medical note for errors:

{corrected_text}

Remember: Use EXACTLY this format:
<think>analysis</think>
<output>Safe/Concerning/Harmful</output>"""

    # Strategic analytical reasoning for safe notes
    analytical_thinking = (
        "Reviewing note systematically: checking medications, dosages, "
        "diagnoses, and clinical findings. All information appears accurate "
        "and consistent. No medical errors detected."
    )

    # PRE-FILL FORMAT: Analytical thinking first, then classification
    assistant_content = f"""<think>{analytical_thinking}</think>
<output>Safe</output>"""
    
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "metadata": {
            "role": "assessor",
            "classification": "Safe",
            "original_id": row['Text ID'],
            "error_type": "none",
            "format": "game_format_strategic"
        }
    }



def main():
    parser = argparse.ArgumentParser(
        description="Generate game format adaptation data with strategic reasoning"
    )
    parser.add_argument(
        "--medec_path",
        type=str,
        default="data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv",
        help="Path to MEDEC training CSV"
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=734,
        help="Start from this ms-train ID (avoids educational SFT data)"
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=125,
        help="Number of rows to use (will generate 4x examples)"
    )
    parser.add_argument(
        "--use_gpt",
        action="store_true",
        help="Use GPT for higher quality reasoning (costs ~$0.05)"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/adaptation/game_format_adaptation.jsonl",
        help="Output path for adaptation data"
    )
    args = parser.parse_args()
    
    # Validate GPT-5 setup if requested
    if args.use_gpt:
        if not HAS_REQUESTS:
            print("❌ requests library not installed. Run: pip install requests")
            print("   Falling back to template-based generation...")
            args.use_gpt = False
        elif not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY not set")
            print("   Falling back to template-based generation...")
            args.use_gpt = False
    
    print("=" * 70)
    print("GAME FORMAT ADAPTATION DATA GENERATION")
    print("=" * 70)
    print(f"\nGoal: Teach model game format + strategic reasoning")
    print(f"Data: Starting from ms-train-{args.start_id}, taking {args.num_rows} rows")
    print(f"Format: PRE-FILL CoT (matches educational SFT + GRPO)")
    print(f"Mode: {'GPT-5 augmented' if args.use_gpt else 'Template-based'}")
    if args.use_gpt:
        print(f"Model: gpt-5 (Responses API)")
    
    # Load MEDEC data
    print(f"\n📂 Loading MEDEC data from {args.medec_path}")
    with open(args.medec_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Filter to Error Flag = 1, starting from start_id
    filtered_rows = [
        r for r in rows
        if r['Error Flag'] == '1'
        and int(r['Text ID'].split('-')[-1]) >= args.start_id
        and r['Text'].strip() and r['Corrected Text'].strip()
    ]
    
    # Sort by ID and take first num_rows
    filtered_rows = sorted(
        filtered_rows, 
        key=lambda x: int(x['Text ID'].split('-')[-1])
    )[:args.num_rows]
    
    # Show ID range
    if filtered_rows:
        ids = [int(r['Text ID'].split('-')[-1]) for r in filtered_rows]
        id_range = f"ms-train-{min(ids)} to ms-train-{max(ids)}"
    else:
        id_range = "None"
    
    print(f"\n📊 Data Analysis:")
    print(f"  Rows found: {len(filtered_rows)}")
    print(f"  ID range: {id_range}")
    print(f"  Will generate: {len(filtered_rows) * 4} examples")
    
    # Count error types
    error_types = Counter([r['Error Type'] for r in filtered_rows])
    print(f"\n  Error type distribution:")
    for error_type, count in error_types.most_common():
        print(f"    • {error_type}: {count} rows → {count * 4} examples")
    
    # Generate examples
    print(f"\n🔨 Generating examples with strategic reasoning...")
    examples = []
    
    if args.use_gpt:
        # Use progress bar for GPT mode
        total_calls = len(filtered_rows) * 2  # Only harmful examples use GPT
        pbar = tqdm(
            total=total_calls,
            desc="GPT calls",
            unit="call"
        )
        
        for row in filtered_rows:
            # 1. Attacker Harmful (uses GPT)
            examples.append(create_attacker_harmful_example(row, use_gpt=True))
            pbar.update(1)
            
            # 2. Attacker Safe (template)
            examples.append(create_attacker_safe_example(row))
            
            # 3. Assessor Harmful (uses GPT)
            examples.append(create_assessor_harmful_example(row, use_gpt=True))
            pbar.update(1)
            
            # 4. Assessor Safe (template)
            examples.append(create_assessor_safe_example(row))
        
        pbar.close()
    else:
        # Template mode - faster, no progress bar needed
        for i, row in enumerate(filtered_rows, 1):
            if i % 25 == 0:
                print(f"  Progress: {i}/{len(filtered_rows)} rows processed")
            
            # All use templates
            examples.append(create_attacker_harmful_example(row, use_gpt=False))
            examples.append(create_attacker_safe_example(row))
            examples.append(create_assessor_harmful_example(row, use_gpt=False))
            examples.append(create_assessor_safe_example(row))
    
    print(f"  ✅ Generated {len(examples)} examples")
    
    # Analyze generated examples
    print(f"\n📝 Generated Examples Breakdown:")
    
    attacker_examples = [
        e for e in examples if e['metadata']['role'] == 'attacker'
    ]
    assessor_examples = [
        e for e in examples if e['metadata']['role'] == 'assessor'
    ]
    
    harmful_games = [
        e for e in examples
        if e['metadata'].get('game_type') == 'harmful'
        or e['metadata'].get('classification') in ['Harmful', 'Concerning']
    ]
    safe_games = [
        e for e in examples
        if e['metadata'].get('game_type') == 'safe'
        or e['metadata'].get('classification') == 'Safe'
    ]
    
    print(f"  Total: {len(examples)}")
    print(f"  - Attacker: {len(attacker_examples)} "
          f"({len(attacker_examples)/len(examples)*100:.1f}%)")
    print(f"  - Assessor: {len(assessor_examples)} "
          f"({len(assessor_examples)/len(examples)*100:.1f}%)")
    print(f"  - Harmful/Concerning: {len(harmful_games)} "
          f"({len(harmful_games)/len(examples)*100:.1f}%)")
    print(f"  - Safe: {len(safe_games)} "
          f"({len(safe_games)/len(examples)*100:.1f}%)")
    
    # Show sample examples
    print(f"\n📄 Sample Examples:")
    print(f"\n--- Attacker Harmful Example ---")
    sample_attacker = next(
        e for e in examples if e['metadata']['game_type'] == 'harmful'
    )
    print(f"Error Type: {sample_attacker['metadata']['error_type']}")
    print(f"User: {sample_attacker['messages'][1]['content'][:150]}...")
    print(f"Assistant: {sample_attacker['messages'][2]['content'][:200]}...")
    
    print(f"\n--- Assessor Harmful Example ---")
    sample_assessor = next(
        e for e in examples
        if e['metadata']['role'] == 'assessor'
        and e['metadata']['classification'] != 'Safe'
    )
    print(f"Error Type: {sample_assessor['metadata']['error_type']}")
    print(f"Classification: {sample_assessor['metadata']['classification']}")
    print(f"Assistant: {sample_assessor['messages'][2]['content'][:200]}...")
    
    # Save to file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\n✅ SUCCESS! Generated {len(examples)} examples")
    
    print(f"\n🎯 Key Features:")
    print(f"  ✓ Strategic reasoning for both attacker and assessor")
    print(f"  ✓ PRE-FILL format (matches educational SFT + GRPO)")
    print(f"  ✓ Clean data separation (ms-train-{args.start_id}+)")
    print(f"  ✓ Balanced 50/50 attacker/assessor")
    print(f"  ✓ Balanced 50/50 harmful/safe")
    print(f"  ✓ Error-type specific reasoning")
    if args.use_gpt:
        print(f"  ✓ GPT-augmented reasoning (higher quality)")
    
    print(f"\n📚 Next Steps:")
    print(f"\n1. Run adaptation training (1 epoch, ~30 minutes):")
    print(f"   python3 script/train_qwen3_sft.py \\")
    print(f"     --model_id <your_educational_model> \\")
    print(f"     --data_path {output_path} \\")
    print(f"     --epochs 1 \\")
    print(f"     --batch_size 4 \\")
    print(f"     --learning_rate 1e-5 \\")
    print(f"     --output_dir trainer_output/qwen3_game_adapted")
    print(f"\n2. Then run GRPO training:")
    print(f"   python3 script/train_selfplay_advanced.py \\")
    print(f"     --model_id trainer_output/qwen3_game_adapted \\")
    print(f"     --num_samples 16 \\")
    print(f"     --rounds 3")
    
    print(f"\n" + "=" * 70)


if __name__ == "__main__":
    main()
