# Model Paths Reference

## Server Models (Used in Training)

These are the models available on your server and used in the actual training pipeline:

### Policy Model (Attacker/Assessor)
```bash
trainer_output/qwen3-4b-medical-selfplay-sft
```
- This is your SFT-trained model
- Used as the base for GRPO selfplay training
- Located in `run_selfplay_training.sh` line 13

### Judge Model (Evaluation)
```bash
google/medgemma-4b-it
```
- Medical specialist model from Google
- Used for harm and plausibility assessment
- Located in `run_selfplay_training.sh` line 21

## Test Scripts Updated

All test scripts now use the correct server model paths:

### test_plausibility_simple_v2.py
```python
--policy-model "trainer_output/qwen3-4b-medical-selfplay-sft"
--judge-model "google/medgemma-4b-it"
```

### test_plausibility_reward.py
```python
--policy-model "trainer_output/qwen3-4b-medical-selfplay-sft"
--judge-model "google/medgemma-4b-it"
```

## Usage on Server

### Quick Test (10 samples)
```bash
python test_plausibility_simple_v2.py --num-samples 10
```

### Full Test (100 samples)
```bash
python test_plausibility_simple_v2.py --num-samples 100
```

### Training
```bash
./run_selfplay_training.sh
```

## Model Requirements

The policy model should exist at:
```bash
/workspace/medical_reward/trainer_output/qwen3-4b-medical-selfplay-sft
```

If it doesn't exist, you need to run SFT training first:
```bash
./run_sft_training.sh
```

## Notes

- The test scripts will automatically use the server model paths
- No need to specify `--policy-model` unless testing a different model
- The judge model (`google/medgemma-4b-it`) will be downloaded from HuggingFace if not cached
- Make sure you're authenticated with HuggingFace: `huggingface-cli login`
