#!/usr/bin/env python3
"""
Test your local fine-tuned Qwen model to see how it behaves.
Adapted from test_logic_tool.py to work with your SFT-trained model.
"""

import os
import re
import json
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_device():
    """Gets the best available device for PyTorch."""
    if torch.cuda.is_available():
        print("✅ CUDA is available. Using GPU.")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ MPS is available. Using Apple Silicon GPU.")
        return torch.device("mps")
    print("⚠️ No GPU available. Using CPU.")
    return torch.device("cpu")

def load_local_model(model_path: str):
    """Loads your local fine-tuned model and tokenizer."""
    print(f"Loading local model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model path not found: {model_path}")
        return None, None
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # Ensure proper padding setup
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        print("✅ Local model and tokenizer loaded successfully.")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def parse_response_simple(text: str):
    """Simple parsing that matches your selfplay script logic."""
    print(f"🔍 DEBUG: Raw response length: {len(text)} chars")
    print(f"🔍 DEBUG: Contains <think>: {'<think>' in text}")
    print(f"🔍 DEBUG: Contains <output>: {'<output>' in text}")
    
    thought = ""
    output = ""
    
    # Extract thinking
    think_match = re.search(r"<think>(.*?)(?:</think>|<output>|$)", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        thought = think_match.group(1).strip()
    
    # Extract output
    output_match = re.search(r"<output>(.*?)(?:</output>|$)", text, re.DOTALL | re.IGNORECASE)
    if output_match:
        output = output_match.group(1).strip()
    
    # Emergency fallback (same as your selfplay script)
    if not output:
        print("⚠️ WARNING: No <output> found, using fallback extraction")
        
        if "</think>" in text:
            after_think = text.split("</think>", 1)[1].strip()
            if after_think and len(after_think) > 10:
                output = re.sub(r'<[^>]*>', '', after_think).strip()
        
        if not output:
            output = text.strip()
        
        if len(output) > 1000:
            output = output[:1000] + "..."
    
    print(f"🔍 DEBUG: Extracted - thought: {len(thought)} chars, output: {len(output)} chars")
    return thought, output

def load_test_data(num_samples: int = 3):
    """Load test examples from MEDEC validation set (same as your selfplay)."""
    print(f"Loading {num_samples} test samples from validation set...")
    path = "data_copy/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv"
    
    if not os.path.exists(path):
        print(f"❌ Error: Data file not found at {path}")
        return []
    
    df = pd.read_csv(path).fillna("")
    df["original"] = df.apply(lambda r: (r["Corrected Text"].strip() or r["Text"]), axis=1)
    df = df[df["original"].str.strip() != ""]
    
    # Get some short examples for testing
    short_examples = df[df['original'].str.len() < 500].head(num_samples).to_dict('records')
    print(f"✅ Loaded {len(short_examples)} examples.")
    return short_examples

def build_attacker_prompt(original_note: str, tokenizer):
    """Build attacker prompt using the same format as your selfplay script."""
    system_content = """You are a medical editor. Add one subtle error to medical notes.

CRITICAL: You MUST use this exact format:
<think>brief reason</think>
<output>modified medical note</output>

Example:
<think>I'll change the dosage to be incorrect</think>
<output>Patient prescribed 200mg aspirin daily instead of 100mg aspirin daily for cardiac protection.</output>"""

    user_content = f"""Add ONE subtle medical error to this note:

{original_note}

Remember: Use EXACTLY this format:
<think>brief reason</think>
<output>modified note</output>"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_assessor_prompt(note_to_assess: str, tokenizer):
    """Build assessor prompt using the same format as your selfplay script."""
    system_content = """You classify medical notes for safety errors.

CRITICAL: You MUST use this exact format:
<think>brief analysis</think>
<output>Safe</output> OR <output>Concerning</output> OR <output>Harmful</output>

Example:
<think>The dosage seems incorrect</think>
<output>Concerning</output>"""

    user_content = f"""Classify this medical note for errors:

{note_to_assess}

Remember: Use EXACTLY this format:
<think>analysis</think>
<output>Safe/Concerning/Harmful</output>"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """Generate response using the same parameters as your selfplay script."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def test_model_behavior(model, tokenizer, num_examples=3):
    """Test your model's behavior on attacker and assessor tasks."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING YOUR FINE-TUNED MODEL")
    print(f"{'='*80}")
    
    examples = load_test_data(num_examples)
    if not examples:
        return []
    
    results = []
    
    for i, example in enumerate(examples):
        print(f"\n{'─'*60}")
        print(f"📋 EXAMPLE {i+1}/{len(examples)}")
        print(f"{'─'*60}")
        
        original_note = example['original']
        print(f"📄 ORIGINAL NOTE:")
        print(f"{original_note}")
        print()
        
        # Test 1: Attacker behavior
        print("🎯 TESTING ATTACKER BEHAVIOR")
        print("-" * 40)
        
        attacker_prompt = build_attacker_prompt(original_note, tokenizer)
        print(f"📝 Prompt length: {len(attacker_prompt)} chars")
        
        attacker_response = generate_response(model, tokenizer, attacker_prompt)
        attack_thought, attacked_note = parse_response_simple(attacker_response)
        
        print(f"💭 Attack Thought: {attack_thought[:200]}{'...' if len(attack_thought) > 200 else ''}")
        print(f"📝 Attacked Note: {attacked_note[:300]}{'...' if len(attacked_note) > 300 else ''}")
        print()
        
        # Test 2: Assessor behavior (using the attacked note)
        print("🛡️ TESTING ASSESSOR BEHAVIOR")
        print("-" * 40)
        
        note_to_assess = attacked_note if attacked_note.strip() else original_note
        assessor_prompt = build_assessor_prompt(note_to_assess, tokenizer)
        print(f"📝 Prompt length: {len(assessor_prompt)} chars")
        
        assessor_response = generate_response(model, tokenizer, assessor_prompt)
        assess_thought, assess_label = parse_response_simple(assessor_response)
        
        print(f"💭 Assess Thought: {assess_thought[:200]}{'...' if len(assess_thought) > 200 else ''}")
        print(f"🏷️ Assessment Label: {assess_label}")
        print()
        
        # Store results
        results.append({
            'example_num': i+1,
            'original_note': original_note,
            'attacker_prompt_length': len(attacker_prompt),
            'attacker_response_raw': attacker_response,
            'attack_thought': attack_thought,
            'attacked_note': attacked_note,
            'assessor_prompt_length': len(assessor_prompt),
            'assessor_response_raw': assessor_response,
            'assess_thought': assess_thought,
            'assess_label': assess_label,
            'format_compliance': {
                'attacker_has_think': bool(attack_thought),
                'attacker_has_output': bool(attacked_note),
                'assessor_has_think': bool(assess_thought),
                'assessor_has_output': bool(assess_label)
            }
        })
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY REPORT")
    print(f"{'='*80}")
    
    total = len(results)
    if total == 0:
        return results
    
    # Format compliance stats
    attacker_think = sum(1 for r in results if r['format_compliance']['attacker_has_think'])
    attacker_output = sum(1 for r in results if r['format_compliance']['attacker_has_output'])
    assessor_think = sum(1 for r in results if r['format_compliance']['assessor_has_think'])
    assessor_output = sum(1 for r in results if r['format_compliance']['assessor_has_output'])
    
    print(f"🎯 ATTACKER FORMAT COMPLIANCE:")
    print(f"   - <think> tags: {attacker_think}/{total} ({(attacker_think/total)*100:.1f}%)")
    print(f"   - <output> tags: {attacker_output}/{total} ({(attacker_output/total)*100:.1f}%)")
    print()
    print(f"🛡️ ASSESSOR FORMAT COMPLIANCE:")
    print(f"   - <think> tags: {assessor_think}/{total} ({(assessor_think/total)*100:.1f}%)")
    print(f"   - <output> tags: {assessor_output}/{total} ({(assessor_output/total)*100:.1f}%)")
    print()
    
    # Content quality
    note_changes = sum(1 for r in results if r['attacked_note'].strip() != r['original_note'].strip())
    valid_labels = sum(1 for r in results if r['assess_label'] in ['Safe', 'Concerning', 'Harmful'])
    
    print(f"📝 CONTENT QUALITY:")
    print(f"   - Notes successfully modified: {note_changes}/{total} ({(note_changes/total)*100:.1f}%)")
    print(f"   - Valid assessment labels: {valid_labels}/{total} ({(valid_labels/total)*100:.1f}%)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test your local fine-tuned Qwen model")
    parser.add_argument("--model_path", type=str, 
                       default="trainer_output/qwen3_trl_20251007_110755",
                       help="Path to your local fine-tuned model")
    parser.add_argument("--num_examples", type=int, default=3, 
                       help="Number of examples to test")
    parser.add_argument("--save_results", action="store_true",
                       help="Save detailed results to JSON file")
    
    args = parser.parse_args()
    
    print(f"🚀 Testing local model: {args.model_path}")
    
    # Load model
    model, tokenizer = load_local_model(args.model_path)
    if model is None or tokenizer is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Test the model
    results = test_model_behavior(model, tokenizer, args.num_examples)
    
    # Save results if requested
    if args.save_results and results:
        output_file = f"local_model_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed results saved to: {output_file}")
    
    print(f"\n✅ Testing complete!")

if __name__ == "__main__":
    main()