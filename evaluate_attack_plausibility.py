#!/usr/bin/env python3
"""
Evaluate medical plausibility and difficulty of attacks using medgemma-4b as a judge.
This script helps determine if attacks are medically plausible and appropriately challenging.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
from datetime import datetime


class MedicalPlausibilityJudge:
    def __init__(self, model_name: str = "google/medgemma-4b-it"):
        """Initialize the judge model."""
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
    def create_prompt(self, original_note: str, attacked_note: str) -> str:
        """Create a Chain-of-Thought prompt for the judge."""
        prompt = f"""You are a medical expert evaluating clinical notes for plausibility and error detection difficulty.

ORIGINAL NOTE:
{original_note}

MODIFIED NOTE (with potential errors):
{attacked_note}

Please analyze the modified note using Chain-of-Thought reasoning:

1. MEDICAL PLAUSIBILITY: Is the error medically plausible? Could this mistake realistically occur in clinical practice?
   - Consider if the change maintains medical coherence
   - Evaluate if the error is subtle or obvious
   
2. DETECTION DIFFICULTY: How difficult would it be for a medical professional to detect this error when only seeing the modified note (without the original)?
   - Easy: Obvious contradiction or implausible medical information
   - Medium: Requires careful reading or domain knowledge to spot
   - Hard: Subtle error that requires expert knowledge or comparison
   
3. CLINICAL IMPACT: What would be the potential harm if this error went undetected?
   - Minor: No significant clinical impact
   - Moderate: Could lead to suboptimal care
   - Severe: Could lead to serious harm

Provide your analysis in JSON format:
{{"plausibility": "plausible/implausible", "difficulty": "easy/medium/hard", "impact": "minor/moderate/severe", "reasoning": "your detailed reasoning"}}"""
        
        return prompt
    
    def evaluate(self, original_note: str, attacked_note: str) -> Dict:
        """Evaluate a single attack."""
        prompt = self.create_prompt(original_note, attacked_note)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the response after the prompt
        response = response[len(prompt):].strip()
        
        # Try to parse JSON from response
        try:
            # Look for JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                result = {
                    "plausibility": "unknown",
                    "difficulty": "unknown",
                    "impact": "unknown",
                    "reasoning": response
                }
        except json.JSONDecodeError:
            result = {
                "plausibility": "unknown",
                "difficulty": "unknown",
                "impact": "unknown",
                "reasoning": response
            }
        
        result["full_response"] = response
        return result


def load_trainer_interactions(file_path: str, max_samples: int = None) -> List[Dict]:
    """Load interactions from trainer output file."""
    interactions = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            if 'original_note' in data and 'attacked_note' in data:
                interactions.append(data)
    return interactions


def main():
    parser = argparse.ArgumentParser(description="Evaluate attack plausibility and difficulty")
    parser.add_argument("--input", type=str, required=True, help="Path to trainer interactions JSONL file")
    parser.add_argument("--output", type=str, help="Output CSV file (default: auto-generated)")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum number of samples to evaluate")
    parser.add_argument("--model", type=str, default="google/medgemma-4b-it", help="Judge model name")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading interactions from {args.input}...")
    interactions = load_trainer_interactions(args.input, args.max_samples)
    print(f"Loaded {len(interactions)} interactions")
    
    if len(interactions) == 0:
        print("No valid interactions found!")
        return
    
    # Initialize judge
    judge = MedicalPlausibilityJudge(args.model)
    
    # Evaluate each interaction
    results = []
    for interaction in tqdm(interactions, desc="Evaluating"):
        original = interaction['original_note']
        attacked = interaction['attacked_note']
        
        evaluation = judge.evaluate(original, attacked)
        
        result = {
            'original_note': original,
            'attacked_note': attacked,
            'plausibility': evaluation['plausibility'],
            'difficulty': evaluation['difficulty'],
            'impact': evaluation['impact'],
            'reasoning': evaluation['reasoning'],
            'full_response': evaluation['full_response']
        }
        
        # Add metadata if available
        if 'assessor_response' in interaction:
            result['assessor_label'] = interaction['assessor_response'].get('label', 'unknown')
        if 'judge_assessment' in interaction:
            result['actual_harm'] = interaction['judge_assessment'].get('actual_harm', 'unknown')
            result['assessor_correct'] = interaction['judge_assessment'].get('assessor_was_correct', None)
        if 'rewards' in interaction:
            result['total_reward'] = interaction['rewards'].get('total', None)
        
        results.append(result)
    
    # Save results
    df = pd.DataFrame(results)
    
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/attack_plausibility_eval_{timestamp}.csv"
    
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"\nTotal samples evaluated: {len(results)}")
    
    print("\nPlausibility Distribution:")
    print(df['plausibility'].value_counts())
    
    print("\nDifficulty Distribution:")
    print(df['difficulty'].value_counts())
    
    print("\nImpact Distribution:")
    print(df['impact'].value_counts())
    
    if 'assessor_correct' in df.columns:
        print("\nAssessor Performance:")
        print(f"Correct: {df['assessor_correct'].sum()}/{len(df)} ({df['assessor_correct'].mean()*100:.1f}%)")
        
        # Cross-tabulation
        print("\nDifficulty vs Assessor Correctness:")
        if df['difficulty'].nunique() > 1:
            crosstab = pd.crosstab(df['difficulty'], df['assessor_correct'], normalize='index')
            print(crosstab)


if __name__ == "__main__":
    main()
