"""Test script to verify tokenizer patching fixes the add_special_tokens issue."""

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def patch_tokenizer_for_grpo(tokenizer):
    """Monkey-patch tokenizer to force add_special_tokens=True for GRPO training."""
    original_call = tokenizer.__call__
    
    def patched_call(*args, add_special_tokens=True, **kwargs):
        if not add_special_tokens:
            print("✅ Intercepted add_special_tokens=False, forcing True")
            add_special_tokens = True
        return original_call(*args, add_special_tokens=add_special_tokens, **kwargs)
    
    tokenizer.__call__ = patched_call
    return tokenizer


def test_tokenizer_patch():
    """Test that the patch correctly intercepts add_special_tokens=False."""
    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nTokenizer special tokens:")
    print(f"  EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    if hasattr(tokenizer, "bos_token") and tokenizer.bos_token:
        print(f"  BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")

    # Test text
    test_text = "Hello, world!"

    print("\n" + "=" * 60)
    print("TEST 1: Normal tokenizer with add_special_tokens=True")
    print("=" * 60)
    result1 = tokenizer(test_text, add_special_tokens=True)
    print(f"Input IDs: {result1.input_ids}")
    print(f"Decoded: {tokenizer.decode(result1.input_ids)}")

    print("\n" + "=" * 60)
    print("TEST 2: Normal tokenizer with add_special_tokens=False")
    print("=" * 60)
    result2 = tokenizer(test_text, add_special_tokens=False)
    print(f"Input IDs: {result2.input_ids}")
    print(f"Decoded: {tokenizer.decode(result2.input_ids)}")

    print("\n" + "=" * 60)
    print("TEST 3: Patched tokenizer with add_special_tokens=False")
    print("=" * 60)
    patched = patch_tokenizer_for_grpo(tokenizer)
    result3 = patched(test_text, add_special_tokens=False)
    print(f"Input IDs: {result3.input_ids}")
    print(f"Decoded: {patched.decode(result3.input_ids)}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if result1.input_ids == result3.input_ids:
        print("✅ SUCCESS: Patched tokenizer with False == Normal with True")
        print("   The patch correctly forces add_special_tokens=True")
    else:
        print("❌ FAILURE: Token IDs don't match")

    if result2.input_ids != result3.input_ids:
        print("✅ SUCCESS: Patched tokenizer != Normal with False")
        print("   The patch is actually changing behavior")
    else:
        print("❌ FAILURE: Patch has no effect")

    # Test that other attributes work
    print("\n" + "=" * 60)
    print("TEST 4: Verify patched tokenizer preserves attributes")
    print("=" * 60)
    print(f"patched.eos_token: {patched.eos_token}")
    print(f"patched.pad_token: {patched.pad_token}")
    print(f"patched.vocab_size: {patched.vocab_size}")
    print("✅ Attributes preserved")

    # Test chat template
    print("\n" + "=" * 60)
    print("TEST 5: Verify chat template works with patched tokenizer")
    print("=" * 60)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    chat_prompt = patched.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Chat template output:\n{chat_prompt[:200]}...")
    print("✅ Chat template works")
    
    # Test isinstance check (critical for GRPO)
    print("\n" + "=" * 60)
    print("TEST 6: Verify isinstance check (critical for GRPO)")
    print("=" * 60)
    print(f"isinstance(patched, PreTrainedTokenizerBase): {isinstance(patched, PreTrainedTokenizerBase)}")
    if isinstance(patched, PreTrainedTokenizerBase):
        print("✅ SUCCESS: Patched tokenizer passes isinstance check")
        print("   This means GRPO will accept it!")
    else:
        print("❌ FAILURE: Patched tokenizer fails isinstance check")


if __name__ == "__main__":
    test_tokenizer_patch()
