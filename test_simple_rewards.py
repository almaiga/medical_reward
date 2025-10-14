#!/usr/bin/env python3
"""Test script to verify simple rewards implementation."""

import sys
sys.path.insert(0, 'script')

print('Testing imports...')
from train_selfplay_advanced import (
    load_and_prepare_data,
    build_attacker_prompts,
    deduplicate_attacked_notes
)

print('✅ Imports successful\n')

# Test 1: Data loading with dual game types
print('='*60)
print('TEST 1: Data Loading')
print('='*60)
ds_originals, ds_few_shot = load_and_prepare_data(num_samples=16)
print(f'Total samples: {len(ds_originals)}')
game_types = ds_originals['game_type']
inject_count = sum(1 for gt in game_types if gt == 'inject')
keep_clean_count = sum(1 for gt in game_types if gt == 'keep_clean')
print(f'Inject: {inject_count}, Keep_clean: {keep_clean_count}')
assert inject_count == keep_clean_count, "Game types should be balanced!"
print('✅ Game types balanced\n')

# Test 2: Enhanced prompts
print('='*60)
print('TEST 2: Enhanced Prompts')
print('='*60)

# Mock tokenizer for testing
class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"MOCK_TEMPLATE: {messages[0]['role']}: {messages[0]['content'][:100]}..."

mock_tok = MockTokenizer()
ds_attacker = build_attacker_prompts(ds_originals, ds_few_shot, mock_tok)

# Check inject prompt
inject_sample = [item for item in ds_attacker if item['game_type'] == 'inject'][0]
print(f"Inject prompt preview:\n{inject_sample['prompt'][:300]}...")
# Check for key words that indicate examples
has_examples = any(word in inject_sample['prompt'].lower() for word in ['severe', 'moderate', 'subtle', 'examples'])
assert has_examples, "Inject prompt should mention severity levels or examples!"
print('✅ Inject prompt has examples\n')

# Check keep_clean prompt
keep_clean_sample = [item for item in ds_attacker if item['game_type'] == 'keep_clean'][0]
print(f"Keep_clean prompt preview:\n{keep_clean_sample['prompt'][:200]}...")
assert 'safe' in keep_clean_sample['prompt'].lower(), \
    "Keep_clean prompt should mention keeping safe!"
print('✅ Keep_clean prompt looks good\n')

# Test 3: Deduplication
print('='*60)
print('TEST 3: Deduplication')
print('='*60)

# Create mock attacked notes with duplicates
mock_notes = [
    {"original": "Note A", "attacked": "Modified A1", "game_type": "inject"},
    {"original": "Note A", "attacked": "Modified A2", "game_type": "inject"},  # Duplicate
    {"original": "Note B", "attacked": "Modified B1", "game_type": "keep_clean"},
    {"original": "Note C", "attacked": "Modified C1", "game_type": "inject"},
    {"original": "Note C", "attacked": "Modified C2", "game_type": "inject"},  # Duplicate
]

print(f"Before deduplication: {len(mock_notes)} notes")
deduplicated = deduplicate_attacked_notes(mock_notes)
print(f"After deduplication: {len(deduplicated)} notes")
assert len(deduplicated) == 3, "Should have 3 unique notes (A, B, C)"
print('✅ Deduplication working correctly\n')

# Test 4: Verify reward structure is simple
print('='*60)
print('TEST 4: Reward Structure')
print('='*60)
print("Checking that code uses simple binary rewards...")

# Read the file and check for multipliers
with open('script/train_selfplay_advanced.py', 'r') as f:
    content = f.read()
    
# Should NOT have harm_multipliers dict anymore
assert 'harm_multipliers = {' not in content, \
    "Should not have harm_multipliers dict!"
print('✅ No harm-level multipliers found')

# Should have simple binary rewards
assert 'r_harm = +R_HARM' in content or 'r_harm = -R_HARM' in content, \
    "Should have simple binary rewards!"
print('✅ Simple binary rewards found')

# Should have removed refusal reward complexity
assert 'r_refusal = 0.0' in content, \
    "Should have simplified refusal reward!"
print('✅ Refusal reward simplified\n')

print('='*60)
print('ALL TESTS PASSED! ✅')
print('='*60)
print('\nImplementation is ready for training!')
print('Run: bash run_selfplay_training.sh')
