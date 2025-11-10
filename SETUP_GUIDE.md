# Quick Setup Guide

## One-Command Setup (SSH Server)

After cloning the repository on your SSH server, run:

```bash
bash setup_environment.sh
```

The script will prompt you for your Hugging Face token on first run.

**Important**: Everything is installed in the workspace directory to persist across SSH restarts:
- Miniconda3: `workspace/miniconda3/`
- Conda environment: `workspace/envs/medical_reward/`
- HF token: `workspace/.hf_token`
- HF cache: `workspace/.huggingface/`

This script will:
1. Install Miniconda3 to `workspace/miniconda3/`
2. Create the conda environment in `workspace/envs/medical_reward/`
3. Install all required packages from `requirements.txt`
4. Prompt for and save your Hugging Face token in the workspace

## Manual Steps (if needed)

If you prefer to run steps individually:

```bash
# 1. Install Miniconda3 (Linux)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init bash
source ~/.bashrc

# 2. Create environment
conda create -y -n medical_reward python=3.10
conda activate medical_reward

# 3. Install packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. Login to Hugging Face (interactive)
huggingface-cli login
# Token will be saved to ~/.cache/huggingface/token
```

## After Setup

Activate the environment:
```bash
conda activate medical_reward
```

Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Running Training

```bash
# Self-play training
python script/train_selfplay_advanced.py --model_id Qwen/Qwen2.5-0.5B-Instruct --num_samples 16 --rounds 3

# Baseline evaluation
bash run_baselines.sh
```
