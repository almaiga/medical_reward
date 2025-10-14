#!/usr/bin/env python3
"""Test script to verify dual-game implementation."""

import sys
sys.path.insert(0, 'script')

# Test imports
print('Testing imports...')
from train_selfplay_advanced import load_and_prepare_data

print('✅ Imports successful')

# Test data loading
print('\nTesting data loading with dual game types...')
ds_originals, ds_few_shot = load_and_prepare_data(num_samples=16)

print(f'\n📊 Dataset created:')
print(f'  Total samples: {len(ds_originals)}')
print(f'  Columns: {ds_originals.column_names}')

# Check game type distribution
game_types = ds_originals['game_type']
inject_count = sum(1 for gt in game_types if gt == 'inject')
keep_clean_count = sum(1 for gt in game_types if gt == 'keep_clean')

print(f'\n🎮 Game type distribution:')
print(f'  inject: {inject_count}')
print(f'  keep_clean: {keep_clean_count}')
print(f'  Ratio: {inject_count}/{keep_clean_count}')

# Show sample
print(f'\n📝 Sample entries:')
for i in range(min(3, len(ds_originals))):
    print(f'\n  Entry {i+1}:')
    print(f'    Game type: {ds_originals[i]["game_type"]}')
    print(f'    Original (first 100 chars): {ds_originals[i]["original"][:100]}...')

print('\n✅ All tests passed!')
