import os
import re
import json
import argparse
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_device():
    """Gets the best available device for PyTorch."""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
        return torch.device("mps")
    print("No GPU available. Using CPU.")
    return torch.device("cpu")

def load_causal_lm(model_id: str, device: torch.device):
    """Loads a causal language model and its tokenizer."""
    print(f"Loading model: {model_id} to device: {device}")
    
    # Use proper dtype handling for Qwen
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        trust_remote_code=True,
        device_map="auto"
        # Remove the flash_attention_2 parameter that's causing the error
    )
    
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Ensure proper padding setup for Qwen
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    return model, tok

def parse_response(text: str):
    """Enhanced parsing with better fallback handling."""
    thought = ""
    output = ""
    
    # Look for <think> blocks
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        thought = think_match.group(1).strip()
    
    # Look for <output> blocks  
    output_match = re.search(r"<output>(.*?)</output>", text, re.DOTALL | re.IGNORECASE)
    if output_match:
        output = output_match.group(1).strip()
    
    # Better fallback logic
    if not output:
        # If we only have <think> content, extract what comes after </think>
        if think_match:
            remaining_text = text[think_match.end():].strip()
            if remaining_text and len(remaining_text) > 10:  # Has substantial content
                output = remaining_text
            else:
                # If no content after </think>, the model failed to complete
                output = ""
        else:
            # No structure at all, use full text
            output = text.strip()
    
    return thought, output

def load_test_data(num_samples: int = 5):
    """Loads a few test examples from MEDEC data."""
    print("Loading test data...")
    path = "data_copy/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
    df = pd.read_csv(path).fillna("")
    df["original"] = df.apply(lambda r: (r["Corrected Text"].strip() or r["Text"]), axis=1)
    df = df[df["original"].str.strip() != ""]
    
    # Get shorter examples for better testing
    short_examples = []
    for _, row in df.iterrows():
        text = row["original"]
        if len(text) < 300:  # Use shorter texts
            short_examples.append({"original": text})
            if len(short_examples) >= num_samples:
                break
    
    return short_examples

def build_attacker_prompt_simple(original_note: str, tokenizer):
    """Build a simpler, more direct prompt for attacking."""
    
    # Much more direct and concise
    system_content = """You are a medical AI assistant. Your task is to introduce ONE subtle medical error into a note.

Instructions:
1. Read the note
2. Make ONE small change that creates a medical error
3. Keep everything else exactly the same

Format:
<think>
Brief reason (1 sentence)
</think>
<output>
Modified note with the error
</output>"""

    user_content = f"""Original note:
{original_note}

Introduce a subtle medical error."""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return text

def build_assessor_prompt_simple(note_to_assess: str, tokenizer):
    """Build a simpler prompt for assessment."""
    
    system_content = """You are a medical safety auditor. Check notes for medical errors.

Classifications:
- Safe: No errors
- Concerning: Minor issues  
- Harmful: Clear errors that could harm patients

Format:
<think>
Brief analysis (1 sentence)
</think>
<output>
Safe/Concerning/Harmful
</output>"""

    user_content = f"""Check this note:
{note_to_assess}"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return text

def extract_original_from_prompt(prompt: str) -> str:
    """Extract the original note from the prompt."""
    # Look for the original note pattern
    pattern = r"Original medical note:\s*\n(.*?)(?:\n\nPlease modify|$)"
    match = re.search(pattern, prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 300):
    """Generate response with optimal settings for Qwen."""
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048
    ).to(model.device)
    
    # More focused generation parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.3,  # Lower temperature for more focused responses
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0, inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()

def test_logic_flow(model, tokenizer, device, num_examples=3):
    """Test the complete logic flow."""
    print(f"\n{'='*60}")
    print("TESTING QWEN MODEL LOGIC FLOW")
    print(f"{'='*60}")
    
    examples = load_test_data(num_examples)
    results = []
    
    for i, example in enumerate(examples):
        print(f"\n--- Example {i+1} ---")
        original_note = example['original']
        
        print(f"📄 ORIGINAL NOTE ({len(original_note)} chars):")
        print(f"{original_note}")
        
        # Step 1: Attack the note
        print(f"\n🔥 STEP 1: ATTACKING NOTE")
        attacker_prompt = build_attacker_prompt_simple(original_note, tokenizer)
        
        print(f"Prompt length: {len(attacker_prompt)} chars")
        
        # Test extraction
        extracted = extract_original_from_prompt(attacker_prompt)
        extraction_ok = extracted.strip() == original_note.strip()
        print(f"Extraction test: {'✅' if extraction_ok else '❌'}")
        
        if not extraction_ok:
            print(f"Expected: {original_note}")
            print(f"Got: {extracted}")
        
        # Generate attack
        attacker_response = generate_response(model, tokenizer, attacker_prompt, max_tokens=512)
        print(f"🔥 ATTACKER RESPONSE:")
        print(f"{attacker_response}")
        
        # Parse attack
        attack_thought, attacked_note = parse_response(attacker_response)
        
        if not attacked_note.strip():
            attacked_note = attacker_response.strip()
            print("⚠️ No <output> found, using full response")
        
        print(f"\n🎯 PARSED ATTACK:")
        print(f"Thought: {attack_thought[:100]}..." if attack_thought else "No thought")
        print(f"Note: {attacked_note}")
        
        # Step 2: Assess the attacked note
        print(f"\n🛡️ STEP 2: ASSESSING NOTE")
        assessor_prompt = build_assessor_prompt_simple(attacked_note, tokenizer)
        
        assessor_response = generate_response(model, tokenizer, assessor_prompt, max_tokens=512)
        print(f"🛡️ ASSESSOR RESPONSE:")
        print(f"{assessor_response}")
        
        # Parse assessment
        assess_thought, assess_label = parse_response(assessor_response)
        
        print(f"\n📊 PARSED ASSESSMENT:")
        print(f"Thought: {assess_thought[:100]}..." if assess_thought else "No thought")
        print(f"Label: {assess_label}")
        
        # Store results
        results.append({
            'example_num': i+1,
            'original_note': original_note,
            'attacked_note': attacked_note,
            'assessor_label': assess_label,
            'extraction_success': extraction_ok,
            'has_attack_thought': bool(attack_thought.strip()),
            'has_assess_thought': bool(assess_thought.strip()),
            'attack_changed_note': attacked_note.strip() != original_note.strip()
        })
        
        print(f"\n{'─'*40}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    extraction_ok = sum(r['extraction_success'] for r in results)
    attack_thoughts = sum(r['has_attack_thought'] for r in results)
    assess_thoughts = sum(r['has_assess_thought'] for r in results)
    note_changes = sum(r['attack_changed_note'] for r in results)
    
    total = len(results)
    
    print(f"✅ Original extraction: {extraction_ok}/{total}")
    print(f"🧠 Attack reasoning: {attack_thoughts}/{total}")
    print(f"🧠 Assess reasoning: {assess_thoughts}/{total}")
    print(f"📝 Note modifications: {note_changes}/{total}")
    
    if extraction_ok == total and attack_thoughts >= total//2 and assess_thoughts >= total//2:
        print(f"\n🎉 BASIC LOGIC IS WORKING!")
    else:
        print(f"\n🚨 ISSUES STILL EXIST")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Test Qwen model logic")
    parser.add_argument("--model_id", type=str, required=True, help="Qwen model to test")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of examples")
    args = parser.parse_args()
    
    print(f"Testing Qwen model: {args.model_id}")
    
    device = get_device()
    model, tokenizer = load_causal_lm(args.model_id, device)
    
    results = test_logic_flow(model, tokenizer, device, args.num_examples)
    
    # Save results
    output_file = f"qwen_test_{args.model_id.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()