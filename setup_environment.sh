#!/bin/bash

set -e  # Exit on any error

echo "=========================================="
echo "Medical Reward Model - Environment Setup"
echo "=========================================="
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "Detected OS: ${MACHINE}"
echo ""

# Step 1: Install Miniconda3 if not present
echo "Step 1: Checking for Miniconda3..."
if command -v conda &> /dev/null; then
    echo "✓ Conda already installed at: $(which conda)"
else
    echo "Installing Miniconda3..."
    
    if [ "${MACHINE}" = "Linux" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [ "${MACHINE}" = "Mac" ]; then
        # Detect architecture
        if [ "$(uname -m)" = "arm64" ]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    else
        echo "❌ Unsupported OS: ${MACHINE}"
        exit 1
    fi
    
    # Download and install to workspace/miniconda3
    # Download and install to workspace/miniconda3
    INSTALL_DIR="$workspace/miniconda3"
    wget "${MINICONDA_URL}" -O miniconda.sh
    bash miniconda.sh -b -p "${INSTALL_DIR}"
    rm miniconda.sh
    
    # Initialize conda
    eval "$(${INSTALL_DIR}/bin/conda shell.bash hook)"
    ${INSTALL_DIR}/bin/conda init bash
    
    echo "✓ Miniconda3 installed successfully to ${INSTALL_DIR}"
    echo "⚠️  Please run 'source ~/.bashrc' or restart your shell, then run this script again"
    exit 0
fi

echo ""

# Step 2: Create conda environment in workspace
echo "Step 2: Creating conda environment 'medical_reward' in workspace..."
ENV_DIR="$(pwd)/envs/medical_reward"

if [ -d "${ENV_DIR}" ]; then
    echo "⚠️  Environment already exists at ${ENV_DIR}"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${ENV_DIR}"
        conda create -y -p "${ENV_DIR}" python=3.10
        echo "✓ Environment recreated"
    else
        echo "✓ Using existing environment"
    fi
else
    conda create -y -p "${ENV_DIR}" python=3.10
    echo "✓ Environment created at ${ENV_DIR}"
fi

echo ""

# Step 3: Activate environment and install requirements
echo "Step 3: Installing Python packages..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"

# Upgrade pip
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
    echo "✓ Requirements installed"
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

echo ""

# Step 4: Hugging Face authentication (saved in workspace)
echo "Step 4: Configuring Hugging Face authentication..."

# Store HF token in workspace instead of home directory
HF_TOKEN_PATH="$(pwd)/.hf_token"
export HF_HOME="$(pwd)/.huggingface"
mkdir -p "${HF_HOME}"

if [ -f "${HF_TOKEN_PATH}" ]; then
    echo "✓ Hugging Face token already configured"
    export HF_TOKEN=$(cat "${HF_TOKEN_PATH}")
else
    echo "Hugging Face token not found. Please enter your token:"
    echo "(Get your token from: https://huggingface.co/settings/tokens)"
    read -p "HF Token: " HF_TOKEN
    
    if [ -z "${HF_TOKEN}" ]; then
        echo "⚠️  No token provided. Skipping authentication..."
        echo "You can add it later to: ${HF_TOKEN_PATH}"
    else
        # Save token to workspace
        echo "${HF_TOKEN}" > "${HF_TOKEN_PATH}"
        chmod 600 "${HF_TOKEN_PATH}"
        
        # Install huggingface_hub if needed
        if ! command -v huggingface-cli &> /dev/null; then
            echo "Installing huggingface_hub..."
            pip install -q huggingface_hub
        fi
        
        # Login with token
        echo "${HF_TOKEN}" | huggingface-cli login --token "${HF_TOKEN}"
        echo "✓ Hugging Face token saved to ${HF_TOKEN_PATH}"
    fi
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Everything is stored in the workspace to persist across SSH restarts"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $(pwd)/envs/medical_reward"
echo ""
echo "Or add this to your session startup:"
echo "  export PATH=$(pwd)/miniconda3/bin:\$PATH"
echo "  conda activate $(pwd)/envs/medical_reward"
echo "  export HF_HOME=$(pwd)/.huggingface"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
echo "  python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'"
echo ""
