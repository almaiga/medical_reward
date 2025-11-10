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
    
    # Download and install
    wget "${MINICONDA_URL}" -O miniconda.sh
    bash miniconda.sh -b -p "${HOME}/miniconda3"
    rm miniconda.sh
    
    # Initialize conda
    eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    
    echo "✓ Miniconda3 installed successfully"
    echo "⚠️  Please run 'source ~/.bashrc' or restart your shell, then run this script again"
    exit 0
fi

echo ""

# Step 2: Create conda environment
echo "Step 2: Creating conda environment 'medical_reward'..."
if conda env list | grep -q "^medical_reward "; then
    echo "⚠️  Environment 'medical_reward' already exists"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n medical_reward -y
        conda create -y -n medical_reward python=3.10
        echo "✓ Environment recreated"
    else
        echo "✓ Using existing environment"
    fi
else
    conda create -y -n medical_reward python=3.10
    echo "✓ Environment created"
fi

echo ""

# Step 3: Activate environment and install requirements
echo "Step 3: Installing Python packages..."
eval "$(conda shell.bash hook)"
conda activate medical_reward

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

# Step 4: Hugging Face authentication
echo "Step 4: Configuring Hugging Face authentication..."

# Check if already logged in
HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
if [ -f "${HF_TOKEN_PATH}" ]; then
    echo "✓ Hugging Face token already configured"
else
    echo "Hugging Face token not found. Please enter your token:"
    echo "(Get your token from: https://huggingface.co/settings/tokens)"
    read -p "HF Token: " HF_TOKEN
    
    if [ -z "${HF_TOKEN}" ]; then
        echo "⚠️  No token provided. Skipping authentication..."
        echo "You can login later with: huggingface-cli login"
    else
        # Install huggingface_hub if needed
        if ! command -v huggingface-cli &> /dev/null; then
            echo "Installing huggingface_hub..."
            pip install -q huggingface_hub
        fi
        
        # Login (this saves the token to ~/.cache/huggingface/token)
        echo "${HF_TOKEN}" | huggingface-cli login --token "${HF_TOKEN}"
        echo "✓ Hugging Face authentication configured and saved"
    fi
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate medical_reward"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
echo "  python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'"
echo ""
echo "Hugging Face token configured for model downloads"
echo ""
