#!/usr/bin/env python3
"""
Direct test of GPT-5 Responses API.
"""

import os
import requests
import json


def test_gpt5_direct():
    """Test GPT-5 Responses API directly."""
    print("🧪 Testing GPT-5 Responses API directly")
    print("=" * 40)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = {
        "model": "gpt-5",
        "input": "Say 'Hello from GPT-5' in exactly 3 words.",
        "reasoning": {"effort": "minimal"},
        "text": {"verbosity": "low"},
        "max_output_tokens": 50,
    }

    print(f"Making request to: https://api.openai.com/v1/responses")
    print(f"Data: {json.dumps(data, indent=2)}")

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=data,
            timeout=30,
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"Response: {json.dumps(result, indent=2)}")

            # Extract text from GPT-5 response structure
            output_text = ""
            if "output" in result:
                for output_item in result["output"]:
                    if output_item.get("type") == "message":
                        content = output_item.get("content", [])
                        if content and len(content) > 0:
                            output_text = content[0].get("text", "")
                            break

            if output_text:
                print(f"✅ Output Text: {output_text}")
            else:
                print("⚠️  No text found in response")

        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")


if __name__ == "__main__":
    test_gpt5_direct()
