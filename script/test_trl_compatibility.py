#!/usr/bin/env python3
"""
Test TRL compatibility with our Qwen3 data format.
"""

import json
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig


def test_data_format(data_path: str):
    """Test if our data format works with TRL."""
    print(f"Testing TRL compatibility with: {data_path}")
    
    # Load a few examples
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Just test first 5 examples
                break
            item = json.loads(line.strip())
            data.append(item)
    
    dataset = Dataset.from_list(data)
    print(f"Loaded {len(dataset)} test examples")
    
    # Check data structure
    print("\nData structure:")
    example = dataset[0]
    print(f"Keys: {list(example.keys())}")
    print(f"Messages type: {type(example['messages'])}")
    print(f"Messages length: {len(example['messages'])}")
    print(f"First message: {example['messages'][0]}")
    
    # Test with tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",  # Use available Qwen model for testing
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test chat template application
    messages = example['messages']
    try:
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        print(f"\n✅ Chat template works!")
        print(f"Formatted length: {len(formatted)} chars")
        print(f"Preview: {formatted[:200]}...")
        
        # Test tokenization
        tokens = tokenizer(formatted, return_tensors="pt")
        print(f"✅ Tokenization works!")
        print(f"Token count: {tokens['input_ids'].shape[1]}")
        
    except Exception as e:
        print(f"❌ Chat template failed: {e}")
        return False
    
    # Test SFTConfig creation
    try:
        config = SFTConfig(
            output_dir="test_output",
            max_length=1024,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            eos_token="<|im_end|>",
            dataset_text_field="messages"
        )
        print(f"✅ SFTConfig creation works!")
        
    except Exception as e:
        print(f"❌ SFTConfig failed: {e}")
        return False
    
    print(f"\n🎉 TRL compatibility test passed!")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TRL compatibility")
    parser.add_argument("data_path", help="Path to SFT data file")
    
    args = parser.parse_args()
    
    success = test_data_format(args.data_path)
    
    if success:
        print("\n✅ Your data is compatible with TRL!")
        print("You can proceed with training using script/train_qwen3_trl.py")
    else:
        print("\n❌ Data format issues found!")
        print("Please check your data format.")


if __name__ == "__main__":
    main()