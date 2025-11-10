#!/bin/bash
# Quick activation script for the medical_reward environment
# Source this file: source activate_env.sh

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add conda to PATH
export PATH="${WORKSPACE_DIR}/miniconda3/bin:${PATH}"

# Set HF_HOME to workspace
export HF_HOME="${WORKSPACE_DIR}/.huggingface"

# Load HF token if exists
if [ -f "${WORKSPACE_DIR}/.hf_token" ]; then
    export HF_TOKEN=$(cat "${WORKSPACE_DIR}/.hf_token")
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "${WORKSPACE_DIR}/envs/medical_reward"

echo "✓ Environment activated: medical_reward"
echo "✓ HF_HOME: ${HF_HOME}"
