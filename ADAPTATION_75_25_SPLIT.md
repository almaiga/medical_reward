# Adaptation Data: 75/25 AI-Style vs Human-Style Split

## Overview

Updated `script/generate_game_format_adaptation.py` to implement a 75/25 split for attacker-safe examples, optimized for deployment scenarios where models primarily encounter AI-generated notes.

## Rationale

- **Primary Use Case**: Models will be deployed with AI-generated clinical notes (75%)
- **Robustness**: Models still need to handle real-world messiness (25%)
- **Balance**: Optimizes for common case while maintaining generalization

## Implementation

### Attacker Safe Examples Split

**75% Clean AI-Style:**
- Well-formatted, structured notes
- Consistent terminology
- No abbreviations or shorthand
- Output unchanged from corrected text
- Reasoning: Verification of medical accuracy

**25% Messy Human-Style:**
- Realistic clinical documentation variations
- Abbreviations (e.g., "Temp" vs "temperature", "BP" vs "blood pressure")
- Formatting variations (spacing, symbols, slashes)
- Reordering of information
- Shorthand (e.g., "pt" vs "patient", "exam" vs "examination")
- Output: Same corrected text (messiness applied conceptually)
- Reasoning: Focus on medical content over presentation

### Code Changes

1. **Updated `create_attacker_safe_example()` function:**
   - Added `messy_ratio=0.25` parameter
   - Random selection: 75% clean, 25% messy
   - Different system prompts for each style
   - Different reasoning generation (GPT or template)
   - Added `style` metadata field

2. **Enhanced statistics output:**
   - Shows clean vs messy distribution
   - Validates 75/25 split in generated data

3. **Updated key features documentation:**
   - Highlights 75/25 split in output

## Usage

Generate adaptation data with the split:

```bash
# Template-based (fast, free)
python3 script/generate_game_format_adaptation.py \
  --note_ids_file data/splits/adaptation_correct.json

# GPT-augmented (higher quality)
python3 script/generate_game_format_adaptation.py \
  --note_ids_file data/splits/adaptation_correct.json \
  --use_gpt
```

## Expected Output

For 299 adaptation notes (Error Flag=1):
- **Total examples**: 1,196 (299 × 4)
- **Attacker safe**: 299 examples
  - Clean AI-style: ~224 examples (75%)
  - Messy human-style: ~75 examples (25%)
- **Other examples**: 897 (attacker harmful, assessor harmful, assessor safe)

## Benefits

1. **Deployment-Optimized**: Trains primarily on the note style models will encounter
2. **Robust**: Still handles real-world messiness when it occurs
3. **Efficient**: Focuses training on the most common scenario
4. **Realistic**: Reflects actual deployment distribution

## Next Steps

1. Generate the 299 adaptation examples with proper stratification
2. Verify clean/messy distribution matches 75/25 target
3. Run adaptation training (1 epoch)
4. Proceed to GRPO self-play training
