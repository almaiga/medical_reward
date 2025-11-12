#!/usr/bin/env python3
"""
Medical Error Detection Inference Script

Tests different Qwen3-4B model versions on MEDEC test data:
- Base model (Qwen/Qwen2.5-3B-Instruct or similar)
- Abliterated model (if available)
- Fine-tuned model (from SFT)
- Fine-tuned + Self-play model (from GRPO)

Uses CoT prompting with few-shot examples for error detection.
"""

import os
import json
import argparse
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from local or HF path."""
    print(f"Loading model from: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded successfully")
    return model, tokenizer


def load_test_data(dataset_name: str = "all") -> pd.DataFrame:
    """
    Load MEDEC test data.
    
    Args:
        dataset_name: "ms", "uw", or "all"
    """
    dfs = []
    
    if dataset_name in ["ms", "all"]:
        ms_path = "data_copy/MEDEC/MEDEC-MS/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv"
        if os.path.exists(ms_path):
            df_ms = pd.read_csv(ms_path)
            df_ms['dataset'] = 'MS'
            dfs.append(df_ms)
            print(f"✅ Loaded MS test set: {len(df_ms)} examples")
    
    if dataset_name in ["uw", "all"]:
        uw_path = "data_copy/MEDEC/MEDEC-UW/MEDEC-UW-TestSet-with-GroundTruth-and-ErrorType.csv"
        if os.path.exists(uw_path):
            df_uw = pd.read_csv(uw_path)
            df_uw['dataset'] = 'UW'
            dfs.append(df_uw)
            print(f"✅ Loaded UW test set: {len(df_uw)} examples")
    
    if not dfs:
        raise FileNotFoundError(f"No test data found for dataset: {dataset_name}")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Total test examples: {len(df)}")
    
    return df


def get_few_shot_examples() -> List[Dict[str, str]]:
    """
    Return few-shot examples for error detection.
    These are hand-crafted examples showing the CoT reasoning process.
    """
    return [
        {
            "note": "Patient presents with chest pain. Prescribed aspirin 325mg daily and nitroglycerin as needed. Follow up in 2 weeks.",
            "reasoning": "Let me analyze this medical note for potential errors:\n1. Chest pain assessment - no mention of ECG, troponin, or cardiac workup\n2. Aspirin dosage - 325mg is appropriate for cardiac events\n3. Nitroglycerin - appropriate for angina\n4. Follow-up timing - 2 weeks may be too long for new chest pain without ruling out MI\n\nThe main concern is lack of immediate cardiac workup for chest pain, which is a critical safety issue.",
            "label": "Harmful",
            "explanation": "Missing critical cardiac workup for chest pain presentation"
        },
        {
            "note": "45-year-old male with hypertension. Current BP 138/88. Continue lisinopril 10mg daily. Patient reports good medication adherence. No side effects noted.",
            "reasoning": "Analyzing this note:\n1. Diagnosis clear - hypertension\n2. Blood pressure - 138/88 is slightly elevated but controlled\n3. Medication - lisinopril 10mg is appropriate dosage\n4. Adherence - patient compliant\n5. No adverse effects reported\n\nThis appears to be routine hypertension management with appropriate medication and monitoring.",
            "label": "Safe",
            "explanation": "Appropriate hypertension management with no errors detected"
        },
        {
            "note": "Diabetic patient with HbA1c of 8.2%. Increase metformin to 2000mg twice daily. Advised on diet and exercise.",
            "reasoning": "Checking this note:\n1. HbA1c 8.2% - indicates suboptimal control\n2. Metformin dosage - MAXIMUM dose is 2000mg TOTAL per day, not twice daily\n3. This would be 4000mg/day which exceeds safe limits\n4. Risk of lactic acidosis and GI side effects\n\nThis is a dangerous dosing error that could harm the patient.",
            "label": "Harmful",
            "explanation": "Metformin overdose - prescribed 4000mg/day exceeds maximum safe dose of 2000mg/day"
        }
    ]


def build_error_detection_prompt(
    note: str,
    use_few_shot: bool = True,
    use_cot: bool = True
) -> List[Dict[str, str]]:
    """
    Build prompt for error detection with optional few-shot and CoT.
    
    Returns messages list for chat template.
    """
    system_prompt = """You are an expert medical safety auditor. Your task is to carefully analyze medical notes for potential errors, safety issues, or harmful content.

Classify each note as:
- "Safe": No errors or safety concerns detected
- "Harmful": Contains medical errors, safety issues, or harmful content

Always explain your reasoning."""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add few-shot examples
    if use_few_shot:
        examples = get_few_shot_examples()
        for ex in examples:
            user_msg = f"Analyze this medical note:\n\n{ex['note']}"
            
            if use_cot:
                assistant_msg = f"<think>\n{ex['reasoning']}\n</think>\n\n<output>\nLabel: {ex['label']}\nExplanation: {ex['explanation']}\n</output>"
            else:
                assistant_msg = f"Label: {ex['label']}\nExplanation: {ex['explanation']}"
            
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add the actual query
    query = f"Analyze this medical note:\n\n{note}"
    messages.append({"role": "user", "content": query})
    
    return messages


def parse_response(response: str) -> Tuple[str, str, str]:
    """
    Parse model response to extract thinking, label, and explanation.
    
    Returns: (thinking, label, explanation)
    """
    thinking = ""
    label = "Unknown"
    explanation = ""
    
    # Extract thinking if present
    if "<think>" in response and "</think>" in response:
        start = response.find("<think>") + 7
        end = response.find("</think>")
        thinking = response[start:end].strip()
        response = response[end + 8:].strip()
    
    # Extract output if present
    if "<output>" in response:
        start = response.find("<output>") + 8
        if "</output>" in response:
            end = response.find("</output>")
            response = response[start:end].strip()
        else:
            response = response[start:].strip()
    
    # Extract label
    for line in response.split('\n'):
        line_lower = line.lower().strip()
        if line_lower.startswith('label:'):
            label_text = line.split(':', 1)[1].strip()
            # Normalize label
            if 'safe' in label_text.lower() and 'harmful' not in label_text.lower():
                label = 'Safe'
            elif 'harmful' in label_text.lower():
                label = 'Harmful'
            elif 'concerning' in label_text.lower():
                label = 'Concerning'
        elif line_lower.startswith('explanation:'):
            explanation = line.split(':', 1)[1].strip()
            # Get rest of explanation if multi-line
            remaining = response.split(line, 1)[1].strip()
            if remaining:
                explanation = explanation + " " + remaining
            break
    
    # If no structured format, try to infer from text
    if label == "Unknown":
        response_lower = response.lower()
        if 'no error' in response_lower or 'appears safe' in response_lower:
            label = 'Safe'
        elif 'error' in response_lower or 'harmful' in response_lower or 'dangerous' in response_lower:
            label = 'Harmful'
    
    return thinking, label, explanation


def run_inference(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    use_few_shot: bool = True,
    use_cot: bool = True,
    max_samples: int = None,
    temperature: float = 0.3,
    max_new_tokens: int = 512
) -> List[Dict]:
    """
    Run inference on test data.
    """
    results = []
    
    # Limit samples if specified
    if max_samples:
        test_df = test_df.head(max_samples)
    
    model.eval()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running inference"):
        note = row['Text']
        ground_truth = row['Error Flag']  # 0 = Safe, 1 = Has Error
        error_type = row.get('Error Type', '')
        
        # Build prompt
        messages = build_error_detection_prompt(note, use_few_shot, use_cot)
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse response
        thinking, predicted_label, explanation = parse_response(response)
        
        # Convert ground truth to label
        gt_label = "Harmful" if ground_truth == 1 else "Safe"
        
        # Check if prediction is correct
        correct = (predicted_label == gt_label)
        
        results.append({
            'text_id': row.get('Text ID', f'sample_{idx}'),
            'dataset': row.get('dataset', 'unknown'),
            'note': note,
            'ground_truth_flag': int(ground_truth),
            'ground_truth_label': gt_label,
            'error_type': error_type,
            'predicted_label': predicted_label,
            'explanation': explanation,
            'thinking': thinking,
            'correct': correct,
            'full_response': response
        })
    
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate accuracy, precision, recall, F1."""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    # Binary classification metrics (Safe vs Harmful)
    tp = sum(1 for r in results if r['predicted_label'] == 'Harmful' and r['ground_truth_label'] == 'Harmful')
    fp = sum(1 for r in results if r['predicted_label'] == 'Harmful' and r['ground_truth_label'] == 'Safe')
    tn = sum(1 for r in results if r['predicted_label'] == 'Safe' and r['ground_truth_label'] == 'Safe')
    fn = sum(1 for r in results if r['predicted_label'] == 'Safe' and r['ground_truth_label'] == 'Harmful')
    
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Medical Error Detection Inference")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model (local or HuggingFace)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Name for this model in results (default: use model_path)")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["ms", "uw", "all"],
                       help="Which test dataset to use")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to test (default: all)")
    
    # Prompting arguments
    parser.add_argument("--no_few_shot", action="store_true",
                       help="Disable few-shot examples")
    parser.add_argument("--no_cot", action="store_true",
                       help="Disable chain-of-thought reasoning")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Sampling temperature (default: 0.3)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum tokens to generate (default: 512)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/inference",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set model name
    model_name = args.model_name or args.model_path.replace('/', '_')
    
    print(f"\n{'='*60}")
    print(f"🔬 Medical Error Detection Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Few-shot: {not args.no_few_shot}")
    print(f"CoT: {not args.no_cot}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load test data
    test_df = load_test_data(args.dataset)
    
    # Run inference
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        use_few_shot=not args.no_few_shot,
        use_cot=not args.no_cot,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 Results Summary")
    print(f"{'='*60}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['confusion_matrix']['true_positive']}")
    print(f"  FP: {metrics['confusion_matrix']['false_positive']}")
    print(f"  TN: {metrics['confusion_matrix']['true_negative']}")
    print(f"  FN: {metrics['confusion_matrix']['false_negative']}")
    print(f"{'='*60}\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = os.path.join(
        args.output_dir,
        f"{model_name}_{args.dataset}_{timestamp}_results.jsonl"
    )
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"✅ Detailed results saved to: {results_file}")
    
    # Save summary
    summary = {
        'model_path': args.model_path,
        'model_name': model_name,
        'dataset': args.dataset,
        'few_shot': not args.no_few_shot,
        'cot': not args.no_cot,
        'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens,
        'timestamp': timestamp,
        'metrics': metrics
    }
    
    summary_file = os.path.join(
        args.output_dir,
        f"{model_name}_{args.dataset}_{timestamp}_summary.json"
    )
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
