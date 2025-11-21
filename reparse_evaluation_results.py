#!/usr/bin/env python3
"""
Re-parse evaluation results from CSV to fix JSON parsing issues.
"""

import json
import argparse
import pandas as pd
from datetime import datetime


def parse_response(response: str) -> dict:
    """Parse JSON from model response, handling markdown code blocks."""
    try:
        # Remove markdown code blocks if present
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0].strip()
        elif '```' in response:
            response = response.split('```')[1].split('```')[0].strip()
        
        # Look for JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            return result
        else:
            return {
                "plausibility": "unknown",
                "difficulty": "unknown",
                "impact": "unknown",
                "reasoning": response
            }
    except (json.JSONDecodeError, IndexError) as e:
        return {
            "plausibility": "unknown",
            "difficulty": "unknown",
            "impact": "unknown",
            "reasoning": response,
            "parse_error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Re-parse evaluation results")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, help="Output CSV file (default: overwrites input)")
    
    args = parser.parse_args()
    
    # Load CSV
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")
    
    # Re-parse full_response column
    print("Re-parsing responses...")
    parsed_results = []
    for response in df['full_response']:
        parsed = parse_response(str(response))
        parsed_results.append(parsed)
    
    # Update columns
    df['plausibility'] = [r['plausibility'] for r in parsed_results]
    df['difficulty'] = [r['difficulty'] for r in parsed_results]
    df['impact'] = [r['impact'] for r in parsed_results]
    df['reasoning'] = [r['reasoning'] for r in parsed_results]
    
    # Save
    output_path = args.output if args.output else args.input
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n=== SUMMARY STATISTICS ===")
    print(f"\nTotal samples: {len(df)}")
    
    print("\nPlausibility Distribution:")
    print(df['plausibility'].value_counts())
    
    print("\nDifficulty Distribution:")
    print(df['difficulty'].value_counts())
    
    print("\nImpact Distribution:")
    print(df['impact'].value_counts())
    
    if 'assessor_correct' in df.columns:
        print("\nAssessor Performance:")
        correct_rate = df['assessor_correct'].mean()
        print(f"Overall: {df['assessor_correct'].sum()}/{len(df)} ({correct_rate*100:.1f}%)")
        
        # By difficulty
        print("\nAccuracy by Difficulty:")
        difficulty_stats = df.groupby('difficulty')['assessor_correct'].agg(['mean', 'count'])
        print(difficulty_stats)
        
        # By plausibility
        print("\nAccuracy by Plausibility:")
        plausibility_stats = df.groupby('plausibility')['assessor_correct'].agg(['mean', 'count'])
        print(plausibility_stats)
        
        # By impact
        print("\nAccuracy by Impact:")
        impact_stats = df.groupby('impact')['assessor_correct'].agg(['mean', 'count'])
        print(impact_stats)
        
        # Cross-tab: difficulty vs plausibility
        print("\nDifficulty vs Plausibility:")
        crosstab = pd.crosstab(df['difficulty'], df['plausibility'])
        print(crosstab)


if __name__ == "__main__":
    main()
