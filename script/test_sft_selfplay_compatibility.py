#!/usr/bin/env python3
"""
Test SFT-trained model compatibility with selfplay pipeline.
Validates format compliance and reward function compatibility.
"""

import os
import re
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path: str):
    """Load the SFT-trained model."""
    print(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Ensure proper padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def parse_response(text: str):
    """Parse response using the same logic as selfplay script."""
    print(f"DEBUG: Raw text length: {len(text)}")
    print(f"DEBUG: Contains <output>: {'<output>' in text.lower()}")
    
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
    
    # Emergency fallback
    if not output:
        print("WARNING: No <output> found, using fallback extraction")
        
        if "</think>" in text:
            after_think = text.split("</think>", 1)[1].strip()
            if after_think and len(after_think) > 10:
                output = re.sub(r'<[^>]*>', '', after_think).strip()
        
        if not output:
            output = text.strip()
        
        if len(output) > 1000:
            output = output[:1000] + "..."
    
    print(f"DEBUG: Extracted - thought: {len(thought)} chars, output: {len(output)} chars")
    return thought, output

def test_attacker_role(model, tokenizer, device):
    """Test the model's performance as an attacker."""
    print("\n=== Testing Attacker Role ===")
    
    # Sample attacker prompt (simplified version of what selfplay uses)
    system_content = """You are a medical editor. Add one subtle error to medical notes.

CRITICAL: You MUST use this exact format:
<think>brief reason</think>
<output>modified medical note</output>"""

    user_content = """Add ONE subtle medical error to this note:

Patient prescribed 100mg aspirin daily for cardiac protection. Blood pressure is well controlled on lisinopril 10mg daily.

Remember: Use EXACTLY this format:
<think>brief reason</think>
<output>modified note</output>"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("Testing attacker generation...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"Raw response: {response}")
    
    # Parse using selfplay logic
    thought, output = parse_response(response)
    
    print(f"Parsed thought: {thought}")
    print(f"Parsed output: {output}")
    
    # Check format compliance
    has_proper_format = bool(thought and output)
    print(f"Format compliance: {'✅' if has_proper_format else '❌'}")
    
    return has_proper_format, thought, output

def test_assessor_role(model, tokenizer, device):
    """Test the model's performance as an assessor."""
    print("\n=== Testing Assessor Role ===")
    
    system_content = """You classify medical notes for safety errors.

CRITICAL: You MUST use this exact format:
<think>brief analysis</think>
<output>Safe</output> OR <output>Concerning</output> OR <output>Harmful</output>"""

    user_content = """Classify this medical note for errors:

Patient prescribed 200mg aspirin daily for cardiac protection. Blood pressure is well controlled on lisinopril 10mg daily.

Remember: Use EXACTLY this format:
<think>analysis</think>
<output>Safe/Concerning/Harmful</output>"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("Testing assessor generation...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"Raw response: {response}")
    
    # Parse using selfplay logic
    thought, output = parse_response(response)
    
    print(f"Parsed thought: {thought}")
    print(f"Parsed output: {output}")
    
    # Check format compliance and classification validity
    valid_classifications = ["Safe", "Concerning", "Harmful"]
    has_proper_format = bool(thought and output)
    has_valid_classification = any(cls in output for cls in valid_classifications)
    
    print(f"Format compliance: {'✅' if has_proper_format else '❌'}")
    print(f"Valid classification: {'✅' if has_valid_classification else '❌'}")
    
    return has_proper_format, has_valid_classification, thought, output

def test_reward_compatibility(model, tokenizer, device):
    """Test compatibility with reward functions from selfplay."""
    print("\n=== Testing Reward Function Compatibility ===")
    
    # Test multiple generations to check consistency
    test_cases = [
        {
            "role": "attacker",
            "prompt": "Add a subtle error: Patient takes warfarin 5mg daily.",
            "expected_format": ["<think>", "<output>"]
        },
        {
            "role": "assessor", 
            "prompt": "Classify: Patient takes warfarin 50mg daily.",
            "expected_format": ["<think>", "<output>"],
            "expected_classifications": ["Safe", "Concerning", "Harmful"]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1} ({test_case['role']}):")
        
        # Create proper messages
        if test_case['role'] == 'attacker':
            messages = [
                {"role": "system", "content": "You are a medical editor. Use format: <think>reason</think><output>result</output>"},
                {"role": "user", "content": test_case['prompt']}
            ]
        else:
            messages = [
                {"role": "system", "content": "You classify medical notes. Use format: <think>analysis</think><output>Safe/Concerning/Harmful</output>"},
                {"role": "user", "content": test_case['prompt']}
            ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate multiple responses to test consistency
        format_success = 0
        classification_success = 0
        
        for _ in range(3):  # Test 3 generations
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            thought, output = parse_response(response)
            
            # Check format
            if thought and output:
                format_success += 1
            
            # Check classification for assessor
            if test_case['role'] == 'assessor' and 'expected_classifications' in test_case:
                if any(cls in output for cls in test_case['expected_classifications']):
                    classification_success += 1
        
        format_rate = format_success / 3 * 100
        classification_rate = classification_success / 3 * 100 if test_case['role'] == 'assessor' else 100
        
        print(f"  Format compliance: {format_success}/3 ({format_rate:.0f}%)")
        if test_case['role'] == 'assessor':
            print(f"  Valid classification: {classification_success}/3 ({classification_rate:.0f}%)")
        
        results.append({
            'role': test_case['role'],
            'format_rate': format_rate,
            'classification_rate': classification_rate
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test SFT model compatibility with selfplay")
    parser.add_argument("model_path", help="Path to SFT-trained model")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive tests")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    device = next(model.parameters()).device
    
    print(f"=== SFT Model Selfplay Compatibility Test ===")
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    
    # Test basic functionality
    attacker_success, attacker_thought, attacker_output = test_attacker_role(model, tokenizer, device)
    assessor_format_success, assessor_class_success, assessor_thought, assessor_output = test_assessor_role(model, tokenizer, device)
    
    # Test reward compatibility if comprehensive
    if args.comprehensive:
        reward_results = test_reward_compatibility(model, tokenizer, device)
    else:
        reward_results = []
    
    # Summary
    print(f"\n=== Compatibility Summary ===")
    print(f"Attacker format compliance: {'✅' if attacker_success else '❌'}")
    print(f"Assessor format compliance: {'✅' if assessor_format_success else '❌'}")
    print(f"Assessor classification validity: {'✅' if assessor_class_success else '❌'}")
    
    if reward_results:
        print(f"\nReward function compatibility:")
        for result in reward_results:
            print(f"  {result['role']}: Format {result['format_rate']:.0f}%, Classification {result['classification_rate']:.0f}%")
    
    # Overall assessment
    basic_compatibility = attacker_success and assessor_format_success and assessor_class_success
    
    if basic_compatibility:
        print(f"\n✅ Model is compatible with selfplay pipeline!")
        print(f"Ready to use in train_selfplay_advanced.py")
    else:
        print(f"\n❌ Model needs additional training or prompt engineering")
        print(f"Consider:")
        print(f"  - More SFT training with format-focused examples")
        print(f"  - Adjusting generation parameters")
        print(f"  - Adding format enforcement in selfplay prompts")
    
    # Save test results
    results = {
        "model_path": args.model_path,
        "attacker_format_success": attacker_success,
        "assessor_format_success": assessor_format_success,
        "assessor_classification_success": assessor_class_success,
        "basic_compatibility": basic_compatibility,
        "reward_results": reward_results,
        "sample_outputs": {
            "attacker": {"thought": attacker_thought, "output": attacker_output},
            "assessor": {"thought": assessor_thought, "output": assessor_output}
        }
    }
    
    results_path = f"{args.model_path}/selfplay_compatibility_test.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to: {results_path}")

if __name__ == "__main__":
    main()