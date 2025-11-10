#!/usr/bin/env python3
"""
Create a clean, minimal project structure for medical error detection training.

This keeps only essential files:
- Training scripts (SFT + GRPO self-play)
- Clean training data
- MEDEC dataset
- Setup files (venv, requirements, etc.)
"""

import shutil
from pathlib import Path

def create_clean_project():
    """Create clean project structure."""
    
    print("=" * 70)
    print("CREATE CLEAN PROJECT STRUCTURE")
    print("=" * 70)
    
    # Define clean project root
    clean_root = Path("medical_reward_clean")
    
    if clean_root.exists():
        response = input(f"\n⚠️  {clean_root} already exists. Overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
        shutil.rmtree(clean_root)
    
    clean_root.mkdir()
    print(f"\n✓ Created {clean_root}/")
    
    # 1. Copy training scripts
    print("\n📂 Copying training scripts...")
    
    script_dir = clean_root / "script"
    script_dir.mkdir()
    
    # Essential training scripts
    essential_scripts = [
        "train_selfplay_advanced.py",
        "train_qwen3_sft.py",  # If it exists
    ]
    
    for script in essential_scripts:
        src = Path("script") / script
        if src.exists():
            shutil.copy(src, script_dir / script)
            print(f"  ✓ {script}")
    
    # Copy selfplay package
    selfplay_src = Path("script/selfplay")
    if selfplay_src.exists():
        selfplay_dst = script_dir / "selfplay"
        shutil.copytree(selfplay_src, selfplay_dst)
        print(f"  ✓ selfplay/ (package)")
    
    # 2. Copy clean training data
    print("\n📂 Copying clean training data...")
    
    data_clean_src = Path("data/sft_clean")
    if data_clean_src.exists():
        data_clean_dst = clean_root / "data" / "sft_clean"
        data_clean_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(data_clean_src, data_clean_dst)
        print(f"  ✓ data/sft_clean/")
    else:
        print(f"  ⚠️  data/sft_clean/ not found - run organize_clean_data.py first")
    
    # 3. Copy MEDEC dataset
    print("\n📂 Copying MEDEC dataset...")
    
    medec_src = Path("data_copy/MEDEC")
    if medec_src.exists():
        medec_dst = clean_root / "data" / "MEDEC"
        shutil.copytree(medec_src, medec_dst)
        print(f"  ✓ data/MEDEC/")
    else:
        print(f"  ⚠️  data_copy/MEDEC/ not found")
    
    # 4. Copy setup files
    print("\n📂 Copying setup files...")
    
    setup_files = [
        "requirements.txt",
        "setup_venv.sh",
        "activate_venv.sh",
        "README.md",
    ]
    
    for filename in setup_files:
        src = Path(filename)
        if src.exists():
            shutil.copy(src, clean_root / filename)
            print(f"  ✓ {filename}")
    
    # 5. Create .gitignore
    print("\n📂 Creating .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Training outputs
trainer_output/
*.pt
*.pth
*.safetensors
wandb/
runs/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
"""
    
    with open(clean_root / ".gitignore", 'w') as f:
        f.write(gitignore_content)
    print(f"  ✓ .gitignore")
    
    # 6. Create README
    print("\n📂 Creating README...")
    
    readme_content = """# Medical Error Detection Training

Clean, minimal project for training medical error detection models using self-play GRPO.

## Project Structure

```
medical_reward_clean/
├── script/
│   ├── train_selfplay_advanced.py    # GRPO self-play training
│   ├── train_qwen3_sft.py            # SFT training
│   └── selfplay/                     # Self-play package
├── data/
│   ├── sft_clean/                    # Clean training data
│   │   ├── educational_stratified.jsonl
│   │   ├── adaptation_stratified.jsonl
│   │   └── README.md
│   └── MEDEC/                        # MEDEC dataset
├── requirements.txt
├── setup_venv.sh
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
bash setup_venv.sh

# Activate environment
source venv/bin/activate  # or: bash activate_venv.sh
```

### 2. Train Educational SFT

```bash
python3 script/train_qwen3_sft.py \\
  --data_path data/sft_clean/educational_stratified.jsonl \\
  --epochs 3 \\
  --batch_size 4 \\
  --output_dir trainer_output/qwen3_educational
```

### 3. Train Adaptation (Game Format)

```bash
python3 script/train_qwen3_sft.py \\
  --model_id trainer_output/qwen3_educational \\
  --data_path data/sft_clean/adaptation_stratified.jsonl \\
  --epochs 1 \\
  --batch_size 4 \\
  --learning_rate 1e-5 \\
  --output_dir trainer_output/qwen3_game_adapted
```

### 4. Run GRPO Self-Play Training

```bash
python3 script/train_selfplay_advanced.py \\
  --model_id trainer_output/qwen3_game_adapted \\
  --num_samples 16 \\
  --rounds 3
```

## Training Data

See `data/sft_clean/README.md` for details on the training data:
- **Educational**: 913 notes (75% of MEDEC), all 5 error types
- **Adaptation**: 306 notes → 1,224 examples (25% of MEDEC), game format
- **Stratification**: Proper 75/25 split with no overlap
- **Coverage**: 100% of MEDEC dataset (1,219 notes)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- TRL (Transformer Reinforcement Learning)
- See `requirements.txt` for full list

## Notes

This is a clean, minimal version of the project containing only essential files for training.
The original project may contain additional data generation scripts and experimental code.
"""
    
    with open(clean_root / "README.md", 'w') as f:
        f.write(readme_content)
    print(f"  ✓ README.md")
    
    # 7. Create directory structure summary
    print("\n" + "=" * 70)
    print("✅ CLEAN PROJECT CREATED!")
    print("=" * 70)
    
    print(f"\nLocation: {clean_root.absolute()}/")
    print("\nStructure:")
    print("  script/")
    print("    ├── train_selfplay_advanced.py")
    print("    ├── train_qwen3_sft.py")
    print("    └── selfplay/")
    print("  data/")
    print("    ├── sft_clean/")
    print("    │   ├── educational_stratified.jsonl")
    print("    │   ├── adaptation_stratified.jsonl")
    print("    │   └── README.md")
    print("    └── MEDEC/")
    print("  requirements.txt")
    print("  setup_venv.sh")
    print("  README.md")
    print("  .gitignore")
    
    print("\n📊 Size comparison:")
    
    # Calculate sizes
    def get_dir_size(path):
        total = 0
        for p in Path(path).rglob('*'):
            if p.is_file():
                total += p.stat().st_size
        return total
    
    original_size = get_dir_size(".")
    clean_size = get_dir_size(clean_root)
    
    print(f"  Original project: {original_size / 1024 / 1024:.1f} MB")
    print(f"  Clean project: {clean_size / 1024 / 1024:.1f} MB")
    print(f"  Reduction: {(1 - clean_size/original_size)*100:.1f}%")
    
    print("\n🚀 Next steps:")
    print(f"  1. cd {clean_root}")
    print(f"  2. bash setup_venv.sh")
    print(f"  3. source venv/bin/activate")
    print(f"  4. Start training!")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    create_clean_project()
