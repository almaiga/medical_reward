# 4-Way Game Structure Implementation Summary

## Overview
Successfully implemented the 4-way game structure as described in the Self-RedTeam paper, replacing the previous 2-way (harmful/safe) structure.

## Changes Made

### 1. Data Loading (`script/selfplay/data.py`)
**Before:** 50/50 split (harmful/safe)
**After:** 25/25/25/25 split across 4 categories

- `vanilla_harmful`: Error note → copy as-is (EASY)
- `adversarial_harmful`: Error note → modify/worsen error (HARD)
- `vanilla_benign`: Clean note → copy as-is (EASY)
- `adversarial_benign`: Clean note → inject error (HARD)

**Key Changes:**
- Split MEDEC data into 4 equal quarters
- Added `game_category` field to dataset
- Added `error_note` and `clean_note` columns
- Updated logging to show 4-way distribution

### 2. Prompt Generation (`script/selfplay/prompts.py`)
**Updated `build_attacker_prompts()`:**
- Handles 4 game categories with different instructions:
  - `vanilla_harmful`: "Copy the error note exactly as-is"
  - `adversarial_harmful`: "Modify or worsen the existing error"
  - `vanilla_benign`: "Copy the clean note exactly as-is"
  - `adversarial_benign`: "Inject a new error into the clean note"

**Updated `make_assessor_prompts()`:**
- Changed `game_type` → `game_category` throughout
- Passes through game_category to assessor dataset

### 3. Judge Enhancement (`script/selfplay/judge.py`)
**Updated `get_judge_assessment()`:**
- Added `game_category` parameter (replacing `game_type`)
- Added game context mapping to help judge understand what to expect:
  - `vanilla_harmful`: "This is an unmodified error note from the dataset"
  - `adversarial_harmful`: "An attacker modified an existing error note"
  - `vanilla_benign`: "This is an unmodified clean note from the dataset"
  - `adversarial_benign`: "An attacker tried to inject errors into a clean note"
- Updated classification logic to account for 4 game types

### 4. Reward Functions (`script/selfplay/rewards.py`)
**Updated both reward functions:**
- Changed `game_type` → `game_category` throughout
- Updated diversity stats tracking for 4 categories
- Updated faithfulness tracking:
  - `*_harmful` categories should produce "Harmful"
  - `*_benign` categories should produce "Safe"
- Updated revision reward logic to check if "harmful" in game_category
- Updated all logging to include `game_category`

### 5. Main Training Loop (`script/selfplay/main.py`)
**Updated diversity statistics:**
- Changed from 2-way to 4-way tracking
- Updated logging to show stats for all 4 categories
- Shows Safe/Harmful distribution and faithfulness for each category

## Data Flow

```
MEDEC Dataset (Error Flag=1)
├── Text (error note)
└── Corrected Text (clean note)
         ↓
    Shuffle & Split
         ↓
┌────────────────────────────────────┐
│  25% vanilla_harmful               │
│  - Attacker gets: error note       │
│  - Task: Copy exactly              │
│  - Expected output: Harmful        │
├────────────────────────────────────┤
│  25% adversarial_harmful           │
│  - Attacker gets: error note       │
│  - Task: Modify/worsen error       │
│  - Expected output: Harmful        │
├────────────────────────────────────┤
│  25% vanilla_benign                │
│  - Attacker gets: clean note       │
│  - Task: Copy exactly              │
│  - Expected output: Safe           │
├────────────────────────────────────┤
│  25% adversarial_benign            │
│  - Attacker gets: clean note       │
│  - Task: Inject error              │
│  - Expected output: Harmful        │
└────────────────────────────────────┘
         ↓
    Attacker Training
         ↓
    Assessor Training
```

## Benefits

1. **Balanced Task Difficulty**: 50% easy (vanilla copy) + 50% hard (adversarial modify)
2. **Better Defender Training**: Sees both unmodified and adversarially modified notes
3. **Improved Judge Accuracy**: Context helps judge understand what to expect
4. **Better Logging**: Track performance across all 4 game categories
5. **Matches Paper**: Implements the exact 25:25:25:25 structure from Self-RedTeam paper

## Testing Recommendations

1. Run training with small `--num_samples` (e.g., 16) to verify 4-way split
2. Check logs to confirm game_category distribution is 25/25/25/25
3. Verify vanilla games produce high faithfulness (attacker copies correctly)
4. Verify adversarial games show diversity in modifications
5. Check judge context is helping with classification accuracy

## Example Command

```bash
python -m script.selfplay.main \
  --model_id Qwen/Qwen3-1.7B-Instruct \
  --judge_model_id google/medgemma-4b-it \
  --num_samples 64 \
  --num_generations 2 \
  --learning_rate 1e-5 \
  --rounds 3
```

Expected output in logs:
```
✅ Created 4-way game structure:
   - vanilla_harmful: 16 (25.0%)
   - adversarial_harmful: 16 (25.0%)
   - vanilla_benign: 16 (25.0%)
   - adversarial_benign: 16 (25.0%)
```
