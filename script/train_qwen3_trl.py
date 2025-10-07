#!/usr/bin/env python3
"""
Train Qwen3-4B using TRL's SFTTrainer on medical attacker/assessor data.
Fully compatible with Qwen3 and TRL best practices.
"""

import os
import json
import argparse
import torch
import time
from datetime import datetime, timedelta
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

# Optional wandb import - only if explicitly requested
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class ProgressCallback(TrainerCallback):
    """Custom callback to show detailed training progress."""
    
    def __init__(self, total_epochs, total_steps):
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.start_time = None
        self.epoch_start_time = None
        self.step_times = []
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"🚀 Training started at {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 Total steps: {self.total_steps}")
        print(f"📈 Total epochs: {self.total_epochs}")
        print(f"⏱️  Logging every {args.logging_steps} steps")
        print("-" * 80)
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        current_epoch = int(state.epoch) + 1
        print(f"\n📅 EPOCH {current_epoch}/{self.total_epochs} STARTED")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        current_step = state.global_step
        current_epoch = state.epoch
        
        # Calculate progress
        step_progress = (current_step / self.total_steps) * 100
        epoch_progress = ((current_step % (self.total_steps // self.total_epochs)) / 
                         (self.total_steps // self.total_epochs)) * 100
        
        # Estimate remaining time
        if current_step > 0:
            avg_time_per_step = elapsed / current_step
            remaining_steps = self.total_steps - current_step
            eta_seconds = remaining_steps * avg_time_per_step
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "calculating..."
            
        # Format elapsed time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print(f"📊 Step {current_step:4d}/{self.total_steps} "
              f"({step_progress:5.1f}%) | "
              f"Epoch {current_epoch:.2f} "
              f"({epoch_progress:5.1f}%)")
        
        if 'train_loss' in logs:
            print(f"📉 Loss: {logs['train_loss']:.4f}")
            
        print(f"⏱️  Elapsed: {elapsed_str} | ETA: {eta}")
        
        if 'train_samples_per_second' in logs:
            print(f"🚄 Speed: {logs['train_samples_per_second']:.2f} samples/sec")
            
        print("-" * 60)
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            current_epoch = int(state.epoch)
            print(f"✅ EPOCH {current_epoch} COMPLETED in {timedelta(seconds=int(epoch_time))}")
            
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print("\n" + "=" * 80)
        print(f"🎉 TRAINING COMPLETED!")
        print(f"⏱️  Total time: {timedelta(seconds=int(total_time))}")
        print(f"📊 Total steps: {state.global_step}")
        print(f"🏁 Final loss: {state.log_history[-1].get('train_loss', 'N/A')}")
        print(f"⏰ Finished at: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)


def load_sft_data(data_path: str) -> Dataset:
    """Load SFT training data in TRL-compatible format."""
    print(f"Loading SFT data from {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("Processing training examples...")
    for line in tqdm(lines, desc="Loading data", unit="examples"):
        item = json.loads(line.strip())
        # TRL expects 'messages' field for conversational data
        data.append(item)
    
    print(f"✅ Loaded {len(data)} training examples")
    
    # Count by role
    attacker_count = sum(1 for item in data 
                        if item['metadata']['role'] == 'attacker')
    assessor_count = sum(1 for item in data 
                        if item['metadata']['role'] == 'assessor')
    
    print(f"  - Attacker examples: {attacker_count}")
    print(f"  - Assessor examples: {assessor_count}")
    
    return Dataset.from_list(data)


def setup_model_and_tokenizer(model_id: str):
    """Setup Qwen3 model and tokenizer for TRL training."""
    print(f"🔄 Loading model: {model_id}")
    
    # Load tokenizer with proper settings for Qwen3
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        padding_side="right"  # Required for training
    )
    
    # Qwen3 tokenizer setup - ensure proper padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✅ Tokenizer loaded and configured")
    
    # Load model with optimal settings for Qwen3
    print("🧠 Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() 
                    and torch.cuda.is_bf16_supported() else torch.float16),
        trust_remote_code=True,
        device_map="auto"
    )
    print("✅ Model loaded successfully")
    
    return model, tokenizer


def create_sft_config(output_dir: str, args) -> SFTConfig:
    """Create SFTConfig following TRL and Qwen3 best practices."""
    
    return SFTConfig(
        output_dir=output_dir,
        
        # Training schedule
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        
        # Learning rate and optimization - optimized for Qwen3
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,  # Lower warmup for Qwen3
        weight_decay=0.1,   # Higher weight decay
        
        # Precision - bf16 preferred for Qwen3
        bf16=(torch.cuda.is_available() and 
              torch.cuda.is_bf16_supported()),
        fp16=not (torch.cuda.is_available() and 
                 torch.cuda.is_bf16_supported()),
        
        # Memory optimization
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        
        # TRL-specific settings
        max_length=args.max_seq_length,
        packing=False,  # Don't pack for better format compliance
        
        # For Qwen3 - set proper EOS token
        eos_token="<|im_end|>",  # Qwen3's EOS token
        
        # Logging and saving - more frequent for server monitoring
        logging_steps=5,  # Log every 5 steps for better visibility
        save_steps=100,   # Save more frequently
        save_total_limit=3,
        eval_strategy="no",
        
        # Progress tracking
        disable_tqdm=False,  # Keep tqdm progress bars
        log_level="info",    # More detailed logging
        
        # Other settings
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=(f"qwen3-sft-{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
                 if args.use_wandb else None),
        
        # Stability
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        
        # Optimizer
        optim="adamw_torch",
        group_by_length=False,  # Better for conversation data
        
        # Dataset settings
        dataset_text_field="messages",  # TRL will use messages field
        dataset_kwargs={"skip_prepare_dataset": False},
    )


def validate_model_format(model, tokenizer, sample_prompts):
    """Test the trained model's format compliance."""
    print("\n=== Testing Format Compliance ===")
    
    model.eval()
    device = next(model.parameters()).device
    
    for i, prompt in enumerate(sample_prompts[:2]):  # Test 2 examples
        print(f"\nTest {i+1}:")
        print(f"Prompt: {prompt[:100]}...")
        
        # Create proper messages format
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        print(f"Response: {response}")
        
        # Check format compliance
        has_think = "<think>" in response and "</think>" in response
        has_output = "<output>" in response
        
        print(f"Format check - Think: {'✅' if has_think else '❌'}, "
              f"Output: {'✅' if has_output else '❌'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-4B with TRL SFT for medical selfplay"
    )
    
    # Model and data
    parser.add_argument("--model_id", type=str, 
                       default="mlabonne/Qwen3-4B-abliterated", 
                       help="Model to fine-tune (abliterated for red teaming)")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to SFT training data (JSONL)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="Per-device batch size")
    parser.add_argument("--grad_accumulation", type=int, default=8, 
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                       help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, 
                       help="Maximum sequence length")
    
    # Output and logging
    parser.add_argument("--output_dir", type=str, 
                       default="trainer_output/qwen3_trl",
                       help="Output directory")
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Use Weights & Biases logging")
    
    # Testing
    parser.add_argument("--test_format", action="store_true", 
                       help="Test format compliance after training")
    
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🤖 QWEN SFT TRAINING WITH TRL")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Model: {args.model_id}")
    print(f"📊 Data: {args.data_path}")
    print(f"📁 Output: {output_dir}")
    if torch.cuda.is_available():
        print(f"🔧 Device: {torch.cuda.get_device_name()}")
        print(f"💾 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    else:
        print(f"🔧 Device: CPU")
    print("=" * 80)
    
    # Initialize wandb if requested and available
    if args.use_wandb:
        if HAS_WANDB:
            wandb.init(
                project="qwen3-medical-sft",
                name=f"trl-sft-{timestamp}",
                config=vars(args)
            )
        else:
            print("⚠️  wandb not installed, skipping experiment tracking")
            args.use_wandb = False
    
    # Load data
    print("📊 Step 1/5: Loading training data...")
    dataset = load_sft_data(args.data_path)
    
    # Setup model and tokenizer
    print("🤖 Step 2/5: Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args.model_id)
    
    # Create SFT configuration
    print("⚙️  Step 3/5: Creating training configuration...")
    sft_config = create_sft_config(output_dir, args)
    print(f"✅ Training config created:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Max length: {args.max_seq_length}")
    
    # Calculate training steps for progress tracking
    print("🏗️  Step 4/5: Initializing trainer...")
    total_steps = (len(dataset) // (args.batch_size * args.grad_accumulation)) * args.epochs
    
    # Create progress callback
    progress_callback = ProgressCallback(args.epochs, total_steps)
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,  # TRL uses processing_class
        callbacks=[progress_callback],  # Add our custom progress callback
    )
    
    print("✅ Trainer initialized")
    print(f"📈 Calculated training steps: {total_steps}")
    print(f"📊 Steps per epoch: {total_steps // args.epochs}")
    print(f"🔄 Effective batch size: {args.batch_size * args.grad_accumulation}")
    
    # Train
    print("🚀 Step 5/5: Starting training...")
    print(f"📈 Training {len(dataset)} examples for {args.epochs} epochs")
    print("💡 Progress will be shown by TRL's built-in progress bars")
    print("-" * 60)
    
    trainer.train()
    
    print("-" * 60)
    print("✅ Training completed!")
    
    # Save final model
    print("💾 Saving model and tokenizer...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to: {output_dir}")
    
    # Test format compliance if requested
    if args.test_format:
        # Create sample prompts for testing
        sample_prompts = [
            "Classify this medical note: Patient has chest pain and shortness of breath.",
            "Add a subtle error to this note: Patient prescribed aspirin 100mg daily."
        ]
        validate_model_format(model, tokenizer, sample_prompts)
    
    # Save training info
    training_info = {
        "model_id": args.model_id,
        "data_path": args.data_path,
        "output_dir": output_dir,
        "training_args": vars(args),
        "dataset_size": len(dataset),
        "timestamp": timestamp,
        "framework": "TRL",
        "config_type": "SFTConfig"
    }
    
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("\n" + "=" * 80)
    print("🎉 SFT TRAINING PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"📁 Model saved to: {output_dir}")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Dataset size: {len(dataset)} examples")
    print(f"🔄 Training epochs: {args.epochs}")
    print("✅ Ready for selfplay training!")
    print("=" * 80)
    
    if args.use_wandb and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()