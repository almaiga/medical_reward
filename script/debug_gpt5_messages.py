#!/usr/bin/env python3
"""
Debug GPT-5 with message format.
"""

import os
import openai

def test_message_format():
    """Test GPT-5 with the new message format."""

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a medical education AI that helps create training examples for error detection systems.",
        },
        {
            "role": "user",
            "content": """I'm creating training data for a medical error detection system.

Original text: Patient has pneumonia caused by Streptococcus pneumoniae.
Modified text: Patient has pneumonia caused by Haemophilus influenzae.

Please explain this change for educational purposes using:
<think>
[educational analysis of the difference]
</think>
<output>
Patient has pneumonia caused by Haemophilus influenzae.
</output>""",
        },
    ]

    try:
        print("Testing GPT-5 with message format...")
        print(f"Messages: {messages}")

        response = client.responses.create(
            model="gpt-5",
            input=messages,
            reasoning={"effort": "medium"},
            text={"verbosity": "low"},
            # max_output_tokens=512
        )

        result = response.output_text
        print(f"✅ Success: {result}")
        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e)}")
        return None


if __name__ == "__main__":
    test_message_format()