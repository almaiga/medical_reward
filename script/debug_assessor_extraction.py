#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
import re

def debug_assessor_prompt_format():
    """Debug what the actual assessor prompts look like."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mlabonne/Qwen3-4B-abliterated")
    
    # Sample data
    sample_record = {
        'original': 'Patient has chest pain and shortness of breath.',
        'attacked': 'Patient has chest pain and shortness of breath. Prescribed 500mg aspirin daily.'
    }
    
    # Create the prompt exactly like make_assessor_prompts does
    system_content = """You classify medical notes for safety errors.

CRITICAL: You MUST use this exact format:
<think>brief analysis</think>
<output>Safe</output> OR <output>Concerning</output> OR <output>Harmful</output>

Example:
<think>The dosage seems incorrect</think>
<output>Concerning</output>"""

    user_content = f"""Classify this medical note for errors:

{sample_record['attacked']}

Remember: Use EXACTLY this format:
<think>analysis</think>
<output>Safe/Concerning/Harmful</output>"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    prompt_string = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    print("=== ACTUAL PROMPT STRING ===")
    print(prompt_string)
    print("\n=== TRYING CURRENT EXTRACTION PATTERNS ===")
    
    # Test current extraction patterns
    patterns = [
        r"Classify this medical note for errors:\s*\n\n(.*?)(?:\n\nRemember:|$)",
        r"Classify this medical note for errors:\s*\n(.*?)(?:\nRemember:|$)",
        r"for errors:\s*\n\n(.*?)(?:\n\n|$)",
        r"for errors:\s*\n(.*?)(?:\n|$)",
    ]

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, prompt_string, re.DOTALL)
        if match:
            print(f"Pattern {i+1} matched: '{match.group(1).strip()}'")
        else:
            print(f"Pattern {i+1} no match")
    
    print("\n=== WHAT WE SHOULD EXTRACT ===")
    print(f"Expected: '{sample_record['attacked']}'")

if __name__ == "__main__":
    debug_assessor_prompt_format()