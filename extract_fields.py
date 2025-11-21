import json
import re

input_file = "results/inference/Medical Selfplay Results Nov 18 2025.jsonl"
output_file = "results/inference/Medical Selfplay Results Nov 18 2025_simplified.jsonl"

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        try:
            data = json.loads(line)
            final_content = data.get("final_content", "")
            
            # Extract predicted label from final_content
            predicted_label = None
            if final_content:
                if re.search(r'\bSafe\b', final_content, re.IGNORECASE):
                    predicted_label = "Safe"
                elif re.search(r'\bHarmful\b', final_content, re.IGNORECASE):
                    predicted_label = "Harmful"
            
            simplified = {
                "text_id": data.get("text_id"),
                "ground_truth_label": data.get("ground_truth_label"),
                "predicted_label": predicted_label,
                "final_content": final_content
            }
            outfile.write(json.dumps(simplified) + '\n')
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {e}")
            continue

print(f"Created simplified file: {output_file}")
