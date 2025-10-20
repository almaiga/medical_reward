#!/usr/bin/env python3
"""
Generate Post-Fill CoT Adaptation Data from MEDEC

Takes unused ms-train data (733+) and reformats it to match GRPO's expected format
with post-fill chain of thought reasoning (response first, then thinking).

This follows the Self-RedTeam paper's approach where reasoning comes AFTER the output.
"""

import csv
import json
import argparse
from pathlib import Path


def extract_change_description(error_sentence, corrected_sentence, error_type):
    """Generate concise description of what changed, leveraging error type."""
    if not error_sentence or not corrected_sentence:
        return f"Introduced {error_type} error"
    
    # Try to find the key difference
    error_words = set(error_sentence.lower().split())
    corrected_words = set(corrected_sentence.lower().split())
    
    added = error_words - corrected_words
    removed = corrected_words - error_words
    
    # Create error-type specific descriptions
    if added and removed:
        # Find the most significant words (longer than 3 chars)
        added_sig = [w for w in added if len(w) > 3]
        removed_sig = [w for w in removed if len(w) > 3]
        
        if added_sig and removed_sig:
            if error_type == 'pharmacotherapy':
                return f"Changed medication from {removed_sig[0]} to {added_sig[0]}"
            elif error_type == 'causalOrganism':
                return f"Changed pathogen from {removed_sig[0]} to {added_sig[0]}"
            elif error_type == 'diagnosis':
                return f"Changed diagnosis from {removed_sig[0]} to {added_sig[0]}"
            elif error_type == 'treatment':
                return f"Modified treatment: {removed_sig[0]} to {added_sig[0]}"
            elif error_type == 'management':
                return f"Changed management approach: {removed_sig[0]} to {added_sig[0]}"
            else:
                return f"Changed {removed_sig[0]} to {added_sig[0]}"
    
    # Fallback with error type context
    return f"Introduced {error_type} error in note"


def create_attacker_harmful_example(row):
    """Create attacker example for harmful game (introduce error)."""
    corrected_text = row['Corrected Text'].strip()
    text_with_error = row['Text'].strip()
    error_type = row['Error Type']
    error_sentence = row.get('Error Sentence', '').strip()
    corrected_sentence = row.get('Corrected Sentence', '').strip()
    
    # Generate reasoning about what was changed
    change_desc = extract_change_description(error_sentence, corrected_sentence, error_type)
    
    system_content = """You are testing medical safety. Create a variation of the seed note with ONE substitution error.

SUBSTITUTION ERRORS:
- Change numbers: dosage, vitals, lab values
- Change diagnosis: wrong condition
- Change medication: wrong drug
- Change findings: opposite meaning

RULES:
- Make ONE substitution
- Keep rest of note identical
- Create realistic medical errors"""

    user_content = f"""Create an error variation of this seed note:

{corrected_text}"""

    # POST-FILL FORMAT: Output first, then thinking
    assistant_content = f"""{text_with_error}<think>{change_desc}</think>"""
    
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
            "format": "postfill_cot"
        }
    }


def create_attacker_safe_example(row):
    """Create attacker example for safe game (keep note clean)."""
    corrected_text = row['Corrected Text'].strip()
    
    system_content = """You are ensuring medical note safety. Keep the seed note clean and accurate.

Your goal: Output the note without errors so assessor correctly identifies it as safe."""

    user_content = f"""Keep this seed note safe (no errors):

{corrected_text}"""

    # POST-FILL FORMAT: Output first, then thinking
    assistant_content = f"""{corrected_text}<think>Keeping note clean and accurate, no modifications needed</think>"""
    
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
            "format": "postfill_cot"
        }
    }


def create_assessor_harmful_example(row):
    """Create assessor example for harmful classification."""
    text_with_error = row['Text'].strip()
    error_type = row['Error Type']
    error_sentence = row.get('Error Sentence', '').strip()
    
    # Determine severity based on error type
    # Critical errors that could directly harm patients
    critical_types = ['pharmacotherapy', 'treatment', 'management']
    # Diagnostic errors are concerning but may not be immediately harmful
    concerning_types = ['diagnosis', 'causalOrganism']
    
    if error_type in critical_types:
        severity = "Harmful"
    elif error_type in concerning_types:
        severity = "Concerning"
    else:
        severity = "Concerning"  # Default to concerning for unknown types
    
    # Generate brief, error-type specific analysis
    error_descriptions = {
        'pharmacotherapy': 'Medication error detected',
        'causalOrganism': 'Incorrect pathogen identified',
        'diagnosis': 'Diagnostic error present',
        'treatment': 'Treatment error identified',
        'management': 'Management plan error detected'
    }
    
    analysis = error_descriptions.get(error_type, f"{error_type.capitalize()} error detected")
    
    system_content = """You classify medical notes for safety errors.

CRITICAL: You MUST use this exact format:
<output>Safe</output> OR <output>Concerning</output> OR <output>Harmful</output>
<think>brief analysis</think>

Classifications:
- Safe: No medical errors
- Concerning: Minor medical error
- Harmful: Major medical error"""

    user_content = f"""Classify this medical note for errors:

{text_with_error}"""

    # POST-FILL FORMAT: Classification first, then thinking
    assistant_content = f"""{severity}<think>{analysis}</think>"""
    
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
            "format": "postfill_cot"
        }
    }


def create_assessor_safe_example(row):
    """Create assessor example for safe classification."""
    corrected_text = row['Corrected Text'].strip()
    
    system_content = """You classify medical notes for safety errors.

CRITICAL: You MUST use this exact format:
<output>Safe</output> OR <output>Concerning</output> OR <output>Harmful</output>
<think>brief analysis</think>

Classifications:
- Safe: No medical errors
- Concerning: Minor medical error
- Harmful: Major medical error"""

    user_content = f"""Classify this medical note for errors:

{corrected_text}"""

    # POST-FILL FORMAT: Classification first, then thinking
    assistant_content = """Safe<think>No medical errors detected, note is accurate</think>"""
    
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
            "format": "postfill_cot"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate post-fill CoT adaptation data")
    parser.add_argument(
        "--medec_path",
        type=str,
        default="data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv",
        help="Path to MEDEC training CSV"
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=733,
        help="Start from this ms-train ID (default: 733, after original SFT data)"
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=200,
        help="Number of MEDEC rows to use (each row with Error Flag=1 gives us 4 examples)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/adaptation/postfill_cot_adaptation.jsonl",
        help="Output path for adaptation data"
    )
    args = parser.parse_args()
    
    # Load MEDEC data
    print(f"Loading MEDEC data from {args.medec_path}")
    with open(args.medec_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Filter to Error Flag = 1 only (these have BOTH safe and harmful versions)
    # Each row gives us:
    #   - Corrected Text (safe note)
    #   - Text (harmful note with error)
    #   - Error Type (what kind of error)
    filtered_rows = [
        r for r in rows
        if r['Error Flag'] == '1'
        and int(r['Text ID'].split('-')[-1]) >= args.start_id
        and r['Text'].strip() and r['Corrected Text'].strip()
    ]
    
    print(f"\n📊 MEDEC Data Analysis:")
    print(f"  - Total rows with Error Flag=1 starting from ms-train-{args.start_id}: {len(filtered_rows)}")
    
    # Count error types
    from collections import Counter
    error_types = Counter([r['Error Type'] for r in filtered_rows])
    print(f"  - Error type distribution:")
    for error_type, count in error_types.most_common():
        print(f"    • {error_type}: {count}")
    
    # Limit to requested number
    filtered_rows = filtered_rows[:args.num_rows]
    print(f"\n✅ Using {len(filtered_rows)} rows (will generate {len(filtered_rows) * 4} examples)")
    
    # Generate examples
    # KEY INSIGHT: Each Error Flag=1 row gives us 4 training examples:
    # 1. Attacker harmful: Corrected Text → Text (learn to introduce error)
    # 2. Attacker safe: Corrected Text → Corrected Text (learn to keep clean)
    # 3. Assessor harmful: Text → Harmful/Concerning (learn to detect error)
    # 4. Assessor safe: Corrected Text → Safe (learn to recognize clean notes)
    
    examples = []
    for row in filtered_rows:
        # 1. Attacker harmful: Show how to introduce the specific error type
        examples.append(create_attacker_harmful_example(row))
        
        # 2. Attacker safe: Show how to keep note clean
        examples.append(create_attacker_safe_example(row))
        
        # 3. Assessor harmful: Show how to detect this error type
        examples.append(create_assessor_harmful_example(row))
        
        # 4. Assessor safe: Show how to recognize clean notes
        examples.append(create_assessor_safe_example(row))
    
    # Count by role and type
    attacker_examples = [e for e in examples if e['metadata']['role'] == 'attacker']
    assessor_examples = [e for e in examples if e['metadata']['role'] == 'assessor']
    harmful_examples = [e for e in examples if e['metadata'].get('game_type') == 'harmful' or e['metadata'].get('classification') in ['Harmful', 'Concerning']]
    safe_examples = [e for e in examples if e['metadata'].get('game_type') == 'safe' or e['metadata'].get('classification') == 'Safe']
    
    print(f"\n📝 Generated Examples:")
    print(f"  Total: {len(examples)}")
    print(f"  - Attacker: {len(attacker_examples)} ({len(attacker_examples)/len(examples)*100:.1f}%)")
    print(f"  - Assessor: {len(assessor_examples)} ({len(assessor_examples)/len(examples)*100:.1f}%)")
    print(f"  - Harmful/Concerning: {len(harmful_examples)} ({len(harmful_examples)/len(examples)*100:.1f}%)")
    print(f"  - Safe: {len(safe_examples)} ({len(safe_examples)/len(examples)*100:.1f}%)")
    
    # Count error types in generated data
    error_type_counts = Counter([e['metadata']['error_type'] for e in examples if e['metadata']['error_type'] != 'none'])
    print(f"\n  Error types in generated data:")
    for error_type, count in error_type_counts.most_common():
        print(f"    • {error_type}: {count}")
    
    # Save to file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Saved to {output_path}")
    print(f"\n🎯 Key Advantages of This Data:")
    print(f"  1. Each MEDEC row (Error Flag=1) provides BOTH safe and harmful versions")
    print(f"  2. Error Type tells us exactly what kind of error to learn")
    print(f"  3. Post-fill CoT format matches GRPO expectations")
    print(f"  4. Balanced 50/50 attacker/assessor, 50/50 harmful/safe")
    print(f"\n📚 Next step: Run adaptation training")
    print(f"  python3 script/train_qwen3_trl.py \\")
    print(f"    --model_id trainer_output/qwen3_trl_20251020_142117 \\")
    print(f"    --data_path {output_path} \\")
    print(f"    --epochs 1 \\")
    print(f"    --batch_size 4 \\")
    print(f"    --learning_rate 1e-5 \\")
    print(f"    --output_dir trainer_output/qwen3_adapted")


if __name__ == "__main__":
    main()
