#!/usr/bin/env python3
"""
Train Qwen3-4B using SFT on medical attacker/assessor data.
Prepares the model for selfplay training with proper format compliance.
"""

import os
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer
import wandb

def load_sft_data(data_path: str) -> Dataset:
    """Load and process SFT training data in the correct format."""
    print(f"Loading SFT data from {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # Your data format: {"messages": [...], "metadata": {...}}
            data.append(item)
    
    print(f"Loaded {len(data)} training examples")
    
    # Count by role
    attacker_count = sum(1 for item in data if item['metadata']['role'] == 'attacker')
    assessor_count = sum(1 for item in data if item['metadata']['role'] == 'assessor')
    
    print(f"  - Attacker examples: {attacker_count}")
    print(f"  - Assessor examples: {assessor_count}")
    
    return Dataset.from_list(data)

def format_conversation(example):
    """Format conversation for Qwen3 training - use existing messages as-is."""
    # Your data already has the correct format with user/assistant messages
    # No need to add system messages since they're already properly formatted
    messages = example['messages']
    role = example['metadata']['role']
    error_type = example['metadata']['error_type']
    
    return {"messages": messages, "role": role, "error_type": error_type}

def prepare_training_data(dataset: Dataset, tokenizer) -> Dataset:
    """Prepare data for SFT training following Swift's approach."""
    
    def tokenize_conversation(example):
        # Use the messages directly - they're already in the correct format
        messages = example['messages']
        
        # Apply chat template (this is the key part for Qwen3)
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Tokenize with proper settings for Qwen3
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None
        )
        
        # For SFT, labels = input_ids (standard approach)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return {
            **tokenized,
            "role": example['metadata']['role'],
            "error_type": example['metadata']['error_type']
        }
    
    return dataset.map(tokenize_conversation, remove_columns=dataset.column_names)

def setup_model_and_tokenizer(model_id: str):
    """Setup Qwen3 model and tokenizer following Swift best practices."""
    print(f"Loading model: {model_id}")
    
    # Load tokenizer with proper settings for Qwen3
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        padding_side="right"  # Required for training
    )
    
    # Qwen3 tokenizer setup - ensure proper padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with optimal settings for Qwen3
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def create_training_args(output_dir: str, args) -> TrainingArguments:
    """Create training arguments following Swift's Qwen3 best practices."""
    
    return TrainingArguments(
        output_dir=output_dir,
        
        # Training schedule - Swift recommended settings
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        
        # Learning rate and optimization - optimized for Qwen3
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,  # Swift recommends lower warmup for Qwen3
        weight_decay=0.1,   # Swift uses higher weight decay
        
        # Precision - bf16 preferred for Qwen3
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        
        # Memory optimization
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        
        # Logging and saving
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="no",
        
        # Swift recommended settings
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"qwen3-sft-{datetime.now().strftime('%Y%m%d_%H%M%S')}" if args.use_wandb else None,
        
        # Stability
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        
        # Additional Swift recommendations
        optim="adamw_torch",
        group_by_length=False,  # Better for conversation data
    )

def validate_model_format(model, tokenizer, sample_prompts):
    """Test the trained model's format compliance."""
    print("\n=== Testing Format Compliance ===")
    
    model.eval()
    device = next(model.parameters()).device
    
    for i, prompt in enumerate(sample_prompts[:2]):  # Test 2 examples
        print(f"\nTest {i+1}:")
        print(f"Prompt: {prompt[:100]}...")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
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
        
        print(f"Format check - Think: {'✅' if has_think else '❌'}, Output: {'✅' if has_output else '❌'}")

def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-4B with SFT for medical selfplay")
    
    # Model and data
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Instruct", 
                       help="Model to fine-tune")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to SFT training data (JSONL)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    
    # Output and logging
    parser.add_argument("--output_dir", type=str, default="trainer_output/qwen3_sft",
                       help="Output directory")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    # Testing
    parser.add_argument("--test_format", action="store_true", 
                       help="Test format compliance after training")
    
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Qwen3-4B SFT Training ===")
    print(f"Model: {args.model_id}")
    print(f"Data: {args.data_path}")
    print(f"Output: {output_dir}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="qwen3-medical-sft",
            name=f"sft-{timestamp}",
            config=vars(args)
        )
    
    # Load and prepare data
    dataset = load_sft_data(args.data_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_id)
    
    # Prepare training data
    print("Preparing training data...")
    train_dataset = prepare_training_data(dataset, tokenizer)
    
    # Create training arguments
    training_args = create_training_args(output_dir, args)
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Create trainer following Swift's approach
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=args.max_seq_length,
        packing=False,  # Don't pack sequences for better format compliance
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
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
        "timestamp": timestamp
    }
    
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {output_dir}")
    print(f"Ready for selfplay training!")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()