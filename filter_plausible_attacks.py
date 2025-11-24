#!/usr/bin/env python3
"""
Filter training data to keep only plausible attacks.
Use this to clean your self-play training data.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


class PlausibilityFilter:
    def __init__(self, model_name: str = "google/medgemma-4b-it"):
        """Initialize the plausibility filter."""
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
    def create_prompt(self, original_note: str, attacked_note: str) -> str:
        """Create a simple plausibility check prompt."""
        prompt = f"""You are a medical expert. Evaluate if the modified clinical note is medically plausible.

ORIGINAL NOTE:
{original_note}

MODIFIED NOTE:
{attacked_note}

Is the modified note medically plausible? Consider:
- Does it maintain medical coherence?
- Are the changes realistic?
- Could this occur in clinical practice?

Answer with JSON: {{"plausible": true/false, "reason": "brief explanation"}}"""
        
        return prompt
    
    def is_plausible(self, original_note: str, attacked_note: str) -> tuple[bool, str]:
        """Check if an attack is medically plausible."""
        prompt = self.create_prompt(original_note, attacked_note)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,  # Lower temperature for more consistent judgments
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Parse response
        try:
            # Remove markdown if present
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            # Find JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result.get('plausible', False), result.get('reason', response)
        except:
            pass
        
        # Fallback: look for keywords
        response_lower = response.lower()
        if 'plausible' in response_lower and 'not' not in response_lower.split('plausible')[0][-20:]:
            return True, response
        elif 'implausible' in response_lower or 'not plausible' in response_lower:
            return False, response
        
        # Default to False if unclear
        return False, response


def main():
    parser = argparse.ArgumentParser(description="Filter plausible attacks")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--model", type=str, default="google/medgemma-4b-it", help="Judge model")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading interactions from {args.input}...")
    interactions = []
    with open(args.input, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'original_note' in data and 'attacked_note' in data:
                    interactions.append(data)
            except:
                pass
    
    print(f"Loaded {len(interactions)} interactions")
    
    # Initialize filter
    filter_model = PlausibilityFilter(args.model)
    
    # Filter interactions
    plausible_interactions = []
    implausible_interactions = []
    
    for interaction in tqdm(interactions, desc="Filtering"):
        original = interaction['original_note']
        attacked = interaction['attacked_note']
        
        is_plausible, reason = filter_model.is_plausible(original, attacked)
        
        # Add plausibility info to interaction
        interaction['plausibility_check'] = {
            'plausible': is_plausible,
            'reason': reason
        }
        
        if is_plausible:
            plausible_interactions.append(interaction)
        else:
            implausible_interactions.append(interaction)
    
    # Save plausible interactions
    with open(args.output, 'w') as f:
        for item in plausible_interactions:
            f.write(json.dumps(item) + '\n')
    
    # Save implausible interactions for inspection
    implausible_path = args.output.replace('.jsonl', '_implausible.jsonl')
    with open(implausible_path, 'w') as f:
        for item in implausible_interactions:
            f.write(json.dumps(item) + '\n')
    
    # Print summary
    print(f"\n{'='*80}")
    print("FILTERING RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal interactions: {len(interactions)}")
    print(f"Plausible: {len(plausible_interactions)} ({len(plausible_interactions)/len(interactions)*100:.1f}%)")
    print(f"Implausible: {len(implausible_interactions)} ({len(implausible_interactions)/len(interactions)*100:.1f}%)")
    print(f"\nPlausible interactions saved to: {args.output}")
    print(f"Implausible interactions saved to: {implausible_path}")
    print(f"\n{'='*80}")
    
    # Show some examples
    if implausible_interactions:
        print("\nExample implausible attacks:")
        for i, item in enumerate(implausible_interactions[:3], 1):
            print(f"\n{i}. Reason: {item['plausibility_check']['reason'][:200]}...")


if __name__ == "__main__":
    main()
