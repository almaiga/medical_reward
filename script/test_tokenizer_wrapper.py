"""Test script to verify TokenizerWrapper fixes add_special_tokens."""

from transformers import AutoTokenizer


class TokenizerWrapper:
    """Wrapper to force add_special_tokens=True for GRPO training."""

    def __init__(self, tokenizer):
        self._wrapped = tokenizer

    def __call__(self, *args, add_special_tokens=True, **kwargs):
        if not add_special_tokens:
            print("✅ Intercepted add_special_tokens=False, forcing True")
            add_special_tokens = True
        return self._wrapped(*args, add_special_tokens=add_special_tokens, **kwargs)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def test_tokenizer_wrapper():
    """Test that the wrapper correctly intercepts add_special_tokens=False."""
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
    print("TEST 3: Wrapped tokenizer with add_special_tokens=False")
    print("=" * 60)
    wrapped = TokenizerWrapper(tokenizer)
    result3 = wrapped(test_text, add_special_tokens=False)
    print(f"Input IDs: {result3.input_ids}")
    print(f"Decoded: {wrapped.decode(result3.input_ids)}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if result1.input_ids == result3.input_ids:
        print("✅ SUCCESS: Wrapped tokenizer with False == Normal with True")
        print("   The wrapper correctly forces add_special_tokens=True")
    else:
        print("❌ FAILURE: Token IDs don't match")

    if result2.input_ids != result3.input_ids:
        print("✅ SUCCESS: Wrapped tokenizer != Normal with False")
        print("   The wrapper is actually changing behavior")
    else:
        print("❌ FAILURE: Wrapper has no effect")

    # Test that other attributes work
    print("\n" + "=" * 60)
    print("TEST 4: Verify wrapper delegates other attributes")
    print("=" * 60)
    print(f"wrapped.eos_token: {wrapped.eos_token}")
    print(f"wrapped.pad_token: {wrapped.pad_token}")
    print(f"wrapped.vocab_size: {wrapped.vocab_size}")
    print("✅ Attribute delegation works")

    # Test chat template
    print("\n" + "=" * 60)
    print("TEST 5: Verify chat template works with wrapper")
    print("=" * 60)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    chat_prompt = wrapped.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Chat template output:\n{chat_prompt[:200]}...")
    print("✅ Chat template works")


if __name__ == "__main__":
    test_tokenizer_wrapper()
