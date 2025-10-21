# Quick Start: Game Format Adaptation with GPT-5 🚀

## What You're About to Do

Bridge your educational SFT model to GRPO training using **GPT-5** to generate high-quality strategic reasoning.

## Prerequisites

1. ✅ Educational SFT model trained (1,592 examples)
2. ✅ OpenAI API key with GPT-5 access
3. ✅ `requests` library installed (`pip install requests`)

## Step-by-Step

### 1. Set Your API Key

```bash
export OPENAI_API_KEY='your-gpt-5-api-key-here'
```

### 2. Generate Data with GPT-5 (~13 minutes, ~$0.10-0.15)

```bash
bash run_generate_game_adaptation.sh
```

**What happens:**
- Loads ms-train-733 to 857 (125 rows)
- Calls **GPT-5 Responses API** 250 times
- Generates 500 high-quality examples
- Saves to `data/adaptation/game_format_adaptation.jsonl`

**Progress bar shows:**
```
GPT calls: 100%|████████████| 250/250 [13:24<00:00,  3.21s/call]
```

### 3. Update Model Path

Edit `run_sft_game_adaptation.sh` and set your model:

```bash
MODEL_ID="trainer_output/qwen3_sft_YYYYMMDD_HHMMSS"  # Your educational model
```

### 4. Run Adaptation Training (~30 minutes)

```bash
bash run_sft_game_adaptation.sh
```

**Training settings:**
- 1 epoch (preserve base knowledge)
- Learning rate: 1e-5 (low to avoid forgetting)
- Batch size: 4
- Time: ~30 minutes

### 5. Run GRPO Training

```bash
python3 script/train_selfplay_advanced.py \
    --model_id trainer_output/qwen3_game_adapted_TIMESTAMP \
    --num_samples 16 \
    --rounds 3
```

## Why GPT-5?

| Feature | Template | GPT-4o-mini | **GPT-5** |
|---------|----------|-------------|-----------|
| Quality | Good | Better | **Best** |
| Strategic depth | Basic | Good | **Excellent** |
| Natural language | Formulaic | Natural | **Most natural** |
| Cost | Free | ~$0.05 | **~$0.10-0.15** |
| Time | Instant | ~13 min | **~13 min** |

**GPT-5 advantages:**
- ✅ Deepest strategic reasoning
- ✅ Most sophisticated thinking patterns
- ✅ Best medical knowledge integration
- ✅ Highest quality adaptation data

## Example Output

**Attacker Harmful (GPT-5 generated):**
```
<think>I'll substitute S. pneumoniae with H. influenzae. While both are 
respiratory pathogens, H. influenzae typically causes bronchitis rather than 
lobar pneumonia. The assessor needs to verify organism-phenotype coherence 
and recognize that the consolidation pattern doesn't match H. influenzae's 
typical presentation.</think>
<output>[note with H. influenzae]</output>
```

**Assessor Harmful (GPT-5 generated):**
```
<think>Checking pathogen identification: H. influenzae is documented, but 
the clinical presentation shows lobar consolidation which is characteristic 
of S. pneumoniae, not H. influenzae. This organism-phenotype mismatch could 
lead to suboptimal antibiotic selection and treatment delays.</think>
<output>Harmful</output>
```

## Troubleshooting

**"OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY='your-key'
```

**"requests library not installed"**
```bash
pip install requests
```

**"Model not found"**
- Update `MODEL_ID` in `run_sft_game_adaptation.sh`
- Point to your educational SFT model directory

## Files Created

- ✅ `script/generate_game_format_adaptation.py` - GPT-5 data generator
- ✅ `run_generate_game_adaptation.sh` - Data generation runner
- ✅ `run_sft_game_adaptation.sh` - Adaptation training runner
- ✅ `data/adaptation/game_format_adaptation.jsonl` - Generated data (after step 2)

## Total Time & Cost

- **Data generation**: ~13 minutes, ~$0.10-0.15
- **Adaptation training**: ~30 minutes, free
- **Total**: ~43 minutes, ~$0.10-0.15

## Ready? Let's Go! 🎮

```bash
# 1. Set API key
export OPENAI_API_KEY='your-key'

# 2. Generate data
bash run_generate_game_adaptation.sh

# 3. Update model path in run_sft_game_adaptation.sh

# 4. Train
bash run_sft_game_adaptation.sh

# 5. GRPO!
```

---

**Questions?** Check `GAME_FORMAT_ADAPTATION_READY.md` for detailed documentation.
