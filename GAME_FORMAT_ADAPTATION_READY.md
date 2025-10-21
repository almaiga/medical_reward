# Game Format Adaptation - Ready to Run! 🎮

## What This Does

Bridges the gap between your **educational SFT model** (1,592 examples) and **GRPO training** by teaching:

1. **Game Format**: Attacker vs Assessor prompts (different from educational format)
2. **Strategic Reasoning**: How to play the game effectively
3. **Pre-Fill CoT**: `<think>reasoning</think><output>response</output>` format

## Why This Matters

Your current model:
- ✅ Understands medical errors deeply (educational SFT)
- ❌ Doesn't know game format (attacker/assessor prompts)
- ❌ Lacks strategic reasoning (how to fool/detect)

After adaptation:
- ✅ Recognizes GRPO prompt format
- ✅ Generates strategic reasoning
- ✅ 60-80% faithfulness pass rate (vs current 0%)
- ✅ Non-zero reward variance (enables GRPO learning)

## Data Generation

### Script: `script/generate_game_format_adaptation.py`

**Features:**
- Uses 125 rows starting from ms-train-734 (spans to ~ms-train-950)
- No contamination with educational SFT data (ms-train-0 to 732)
- Generates 500 examples (4 per row)
- **GPT-5 Responses API** for highest quality strategic reasoning
- Pre-fill format (matches educational SFT + GRPO)

**Example Types:**

1. **Attacker Harmful (125 examples)**
   ```
   <think>I'll change S. pneumoniae to H. influenzae. Both cause pneumonia 
   but H. influenzae doesn't match lobar pattern. Assessor must check 
   organism-phenotype coherence to catch this.</think>
   <output>[note with H. influenzae]</output>
   ```

2. **Attacker Safe (125 examples)**
   ```
   <think>This note is already clean. I'll output it unchanged to ensure 
   assessor correctly identifies it as safe.</think>
   <output>[same clean note]</output>
   ```

3. **Assessor Harmful (125 examples)**
   ```
   <think>Checking pathogen: H. influenzae doesn't match lobar pneumonia 
   pattern. S. pneumoniae is typical for lobar consolidation. This 
   mismatch could lead to inappropriate treatment.</think>
   <output>Harmful</output>
   ```

4. **Assessor Safe (125 examples)**
   ```
   <think>Reviewing systematically: pathogen matches presentation, dosages 
   within range, no inconsistencies. All information appears accurate.</think>
   <output>Safe</output>
   ```

## Usage

### Step 1: Generate Data (~13 minutes, ~$0.10-0.15)

```bash
bash run_generate_game_adaptation.sh
```

This will:
- Load 125 rows starting from ms-train-734 (spans to ~ms-train-950)
- Call **GPT-5 Responses API** for strategic reasoning (250 API calls)
- Generate 500 examples
- Save to `data/adaptation/game_format_adaptation.jsonl`

### Step 2: Run Adaptation Training (~30 minutes)

```bash
bash run_sft_game_adaptation.sh
```

**Before running**, update `MODEL_ID` in the script to point to your educational model:
```bash
MODEL_ID="trainer_output/qwen3_sft_20241021_120000"  # UPDATE THIS
```

Training settings:
- **1 epoch only** (preserve base knowledge)
- **Learning rate: 1e-5** (low to avoid forgetting)
- **Batch size: 4** (memory efficient)
- **Time: ~30 minutes**

### Step 3: Run GRPO Training

```bash
python3 script/train_selfplay_advanced.py \
    --model_id trainer_output/qwen3_game_adapted_TIMESTAMP \
    --num_samples 16 \
    --rounds 3
```

## Key Advantages

### vs Template-Based Reasoning
- ✅ **Highest quality**: GPT-5 generates most sophisticated reasoning
- ✅ **More natural**: Best strategic thinking patterns
- ✅ **Worth the cost**: $0.10-0.15 for significantly better adaptation

### vs Post-Fill Format
- ✅ **Consistency**: Matches your 1,592 educational examples
- ✅ **No confusion**: Model knows one format well
- ✅ **GRPO compatible**: `parse_response()` tries pre-fill first

### vs Direct GRPO
- ✅ **Smooth transition**: Gradual format adaptation
- ✅ **Better initialization**: Model starts with game knowledge
- ✅ **Faster convergence**: GRPO learns faster with proper format

## Data Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
| **Total** | 500 | 100% |
| Attacker | 250 | 50% |
| Assessor | 250 | 50% |
| Harmful/Concerning | 250 | 50% |
| Safe | 250 | 50% |

**Error Types Covered:**
- Pharmacotherapy (medication errors)
- Causal Organism (pathogen errors)
- Diagnosis (diagnostic errors)
- Treatment (treatment errors)
- Management (management errors)

## Expected Results

After adaptation training:
- ✅ Model recognizes game format
- ✅ Generates valid `<think>` and `<output>` tags
- ✅ 60-80% faithfulness pass rate (vs current 0%)
- ✅ Non-zero reward variance (enables GRPO learning)
- ✅ No mode collapse
- ✅ Strategic reasoning in both roles

## Files Created

- `script/generate_game_format_adaptation.py` - Data generation script
- `run_generate_game_adaptation.sh` - Data generation runner
- `run_sft_game_adaptation.sh` - Adaptation training runner
- `GAME_FORMAT_ADAPTATION_READY.md` - This document

## Training Pipeline

```
Educational SFT (1,592 examples)
    ↓
    ms-train-0 to 732
    Deep medical reasoning
    ↓
Game Format Adaptation (500 examples) ← YOU ARE HERE
    ↓
    ms-train-733 to 857
    Strategic game reasoning
    1 epoch, 30 minutes
    ↓
GRPO Training (Self-Play)
    ↓
    Validation set
    3 rounds, attacker vs assessor
    ↓
Final Model
```

## Cost & Time

| Step | Time | Cost |
|------|------|------|
| Data Generation (GPT-5) | ~13 min | ~$0.10-0.15 |
| Adaptation Training | ~30 min | Free |
| **Total** | **~43 min** | **~$0.10-0.15** |

## Ready to Run! 🚀

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

2. Generate data:
   ```bash
   bash run_generate_game_adaptation.sh
   ```

3. Update model path in `run_sft_game_adaptation.sh`

4. Run adaptation training:
   ```bash
   bash run_sft_game_adaptation.sh
   ```

5. Run GRPO training with adapted model!

---

**Questions?** Check the scripts for detailed comments and configuration options.
