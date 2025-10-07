#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re

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

def debug_with_sample_garbage():
    """Debug with the garbage data the user is seeing."""
    
    # This looks like what the user is seeing
    garbage_prompt = "useruseru"
    
    print("=== DEBUGGING GARBAGE PROMPT ===")
    print(f"Input prompt: '{garbage_prompt}'")
    print(f"Length: {len(garbage_prompt)}")
    print(f"Repr: {repr(garbage_prompt)}")
    
    result = extract_attacked_from_assessor_prompt(garbage_prompt)
    print(f"Extraction result: '{result}'")
    
    print("\n=== CHECKING IF THIS IS TOKENIZED DATA ===")
    # Maybe the prompt is coming in as token IDs or something weird
    if garbage_prompt.startswith("user"):
        print("Looks like repeated 'user' strings - this suggests tokenization issue")
    
    print("\n=== WHAT SHOULD HAPPEN ===")
    print("The prompt should be the full chat template string with <|im_start|> tags")
    print("If we're getting 'useruseru', something is wrong with how prompts are passed to reward function")

if __name__ == "__main__":
    debug_with_sample_garbage()