#!/usr/bin/env python3
"""Display JSONL data in a readable format."""

import json
import sys


def display_jsonl(file_path):
    """Read and display JSONL file with proper formatting."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Display each entry
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                print(f"\n{'='*80}")
                print(
                    f"Entry {i} - Round {data.get('round', '?')} - "
                    f"Phase: {data.get('phase', '?')}"
                )
                print(f"{'='*80}")
                print(f"Timestamp: {data.get('timestamp', 'N/A')}")

                orig = data.get("original_note", "N/A")
                print(f"\nOriginal Note:")
                print(orig)

                attacked = data.get("attacked_note", "N/A")
                print(f"\nAttacked Note:")
                print(attacked)

                assessor = data.get("assessor_response", {})
                print("\nAssessor Response:")
                if isinstance(assessor, dict):
                    if "thought" in assessor:
                        print(f"  Thought: {assessor['thought']}")
                    if "label" in assessor:
                        print(f"  Label: {assessor['label']}")
                else:
                    print(f"  {assessor}")

                judge = data.get("judge_assessment", {})
                print("\nJudge Assessment:")
                if isinstance(judge, dict):
                    for key, value in judge.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {judge}")

                rewards = data.get("rewards", {})
                print("\nRewards:")
                if isinstance(rewards, dict):
                    for key, value in rewards.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {rewards}")

            except json.JSONDecodeError as e:
                print(f"\nEntry {i}: JSON decode error - {e}")
                print(f"Raw: {line[:100]}...")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/Trainer Output Interactions.jsonl"

    display_jsonl(file_path)
