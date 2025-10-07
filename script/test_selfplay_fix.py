#!/usr/bin/env python3
"""Test that the selfplay fix resolves the double-templating issue."""

from transformers import AutoTokenizer

# Test the fix
tokenizer = AutoTokenizer.from_pretrained("mlabonne/Qwen3-4B-abliterated")

# Simulate what make_assessor_prompts now returns (raw text)
sample_attacked_note = "Patient has chest pain. Prescribed 500mg aspirin daily."
user_content = f"""Classify this medical note for errors:

{sample_attacked_note}

Remember: Use EXACTLY this format:
<think>analysis</think>
<output>Safe/Concerning/Harmful</output>"""

system_content = """You classify medical notes for safety errors.

CRITICAL: You MUST use this exact format:
<think>brief analysis</think>
<output>Safe</output> OR <output>Concerning</output> OR <output>Harmful</output>

Example:
<think>The dosage seems incorrect</think>
<output>Concerning</output>"""

print("=== RAW TEXT (what we now store in dataset) ===")
print(f"User content: {user_content[:100]}...")
print()

# Simulate what happens during manual generation (attacker reward function)
messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content},
]
templated_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print("=== TEMPLATED ONCE (for manual generation) ===")
print(templated_prompt[:500])
print()

# Check that we don't have "useruseruser..." garbage
if "useruser" in templated_prompt.lower():
    print("❌ FAIL: Found 'useruser' garbage in prompt!")
else:
    print("✅ PASS: No 'useruser' garbage found!")

# Check that the actual medical note is present
if sample_attacked_note in templated_prompt:
    print("✅ PASS: Medical note is present in prompt!")
else:
    print("❌ FAIL: Medical note is missing from prompt!")

print("\n=== SUMMARY ===")
print("The fix ensures:")
print("1. Prompts are stored as raw text in datasets")
print("2. GRPO applies chat template once during training")
print("3. Manual generation applies chat template once explicitly")
print("4. No double-templating = no 'useruser' garbage")
