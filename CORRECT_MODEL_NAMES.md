# Correct Model Names for This Project

## Models Used

### Policy Model (Attacker & Assessor)
```
Abdine/qwen3-4b-medical-selfplay-sft
```
- Based on Qwen 3 4B
- Fine-tuned for medical self-play
- Used for both attacker and assessor roles

### Judge Model (Harm & Plausibility Assessment)
```
google/medgemma-4b-it
```
- MedGemma 4B instruction-tuned
- Used for judging harm and plausibility
- Medical domain expert model

## Usage in Scripts

### test_plausibility_simple.py
```bash
# Uses medgemma-4b-it by default
python test_plausibility_simple.py
```

### test_plausibility_reward.py
```bash
# Uses both models by default
python test_plausibility_reward.py --num-samples 5

# Or specify explicitly:
python test_plausibility_reward.py \
    --policy-model Abdine/qwen3-4b-medical-selfplay-sft \
    --judge-model google/medgemma-4b-it \
    --num-samples 5
```

### Training Scripts
```bash
# In script/selfplay/main.py or training scripts
POLICY_MODEL = "Abdine/qwen3-4b-medical-selfplay-sft"
JUDGE_MODEL = "google/medgemma-4b-it"
```

## Important Notes

- **DO NOT use**: `Qwen/Qwen2.5-3B-Instruct` (wrong model)
- **DO NOT use**: `google/medgemma-2b-it` (wrong size)
- **USE**: The models listed above

## Model Locations

If using local models:
```python
# Policy model
policy_model_path = "trainer_output/qwen3-4b-medical-selfplay-sft"

# Judge model (if downloaded locally)
judge_model_path = "models/medgemma-4b-it"
```

## Verification

To verify you're using the correct models:
```python
print(f"Policy model: {policy_model.config._name_or_path}")
print(f"Judge model: {judge_model.config._name_or_path}")
```

Should output:
```
Policy model: Abdine/qwen3-4b-medical-selfplay-sft
Judge model: google/medgemma-4b-it
```
