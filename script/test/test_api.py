#!/usr/bin/env python3
"""
Test script to verify API connectivity and latest OpenAI API usage.
"""

import os
import sys

def test_openai_api():
    """Test OpenAI API with latest version."""
    try:
        import openai
        print("✅ OpenAI package imported successfully")
        
        # Check if API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set")
            return False
        
        print(f"✅ API key found: {api_key[:10]}...")
        
        # Test API call with latest format
        client = openai.OpenAI(api_key=api_key)
        
        # Use GPT-5 Responses API
        response = client.responses.create(
            model="gpt-5",
            input="Write something about usefulness of AI in Healthcare",
            reasoning={"effort": "low"},
            max_output_tokens=300
        )
        
        result = response.output_text
        print("✅ API call successful")
        print(result)
        return True
        
    except ImportError:
        print("❌ OpenAI package not installed. Run: pip install openai")
        return False
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

def test_claude_api():
    """Test Claude API."""
    try:
        import anthropic
        print("✅ Anthropic package imported successfully")
        
        # Check if API key is set
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not set")
            return False
        
        print(f"✅ API key found: {api_key[:10]}...")
        
        # Test API call
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            temperature=0.1,
            messages=[{"role": "user", "content": "Say 'API test successful'"}]
        )
        
        result = response.content[0].text
        print(f"✅ API call successful: {result}")
        return True
        
    except ImportError:
        print("❌ Anthropic package not installed. Run: pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

def main():
    print("=== API Connectivity Test ===\n")
    
    openai_works = test_openai_api()
    print()
    claude_works = test_claude_api()
    
    print("\n=== Summary ===")
    if openai_works:
        print("✅ OpenAI API ready for use")
    if claude_works:
        print("✅ Claude API ready for use")
    
    if not openai_works and not claude_works:
        print("❌ No working APIs found. Please set up at least one API key.")
        sys.exit(1)
    else:
        print("🎉 Ready to generate SFT data!")

if __name__ == "__main__":
    main()