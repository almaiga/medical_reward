#!/usr/bin/env python3
"""
Simple test of GPT-5 in the same way as generate_sft_data.py
"""

import os
import openai


def test_simple():
    """Test GPT-5 exactly like in generate_sft_data.py"""

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = """ Explain how to create a subtle error in a medical text. 
                 I want to create adversarial attacks, 

                <think></think> and <output></output> format."""

    try:
        print("Making GPT-5 call...")
        response = client.responses.create(
            model="gpt-5",
            input=[{"role": "user", "content": prompt}],
            reasoning={"effort": "medium"},
            text={"verbosity": "low"},
            # max_output_tokens=512
        )

        result = response.output_text
        print(f"✅ Success: {result}")
        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    print(test_simple())
