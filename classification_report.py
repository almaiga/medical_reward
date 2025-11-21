import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

input_file = "results/inference/Medical Selfplay Results Nov 18 2025_simplified.jsonl"

ground_truth = []
predictions = []

with open(input_file, 'r') as infile:
    for line in infile:
        data = json.loads(line)
        gt = data.get("ground_truth_label")
        pred = data.get("predicted_label")
        
        if gt is not None and pred is not None:
            ground_truth.append(gt)
            predictions.append(pred)

print("Classification Report:")
print("=" * 60)
print(classification_report(ground_truth, predictions))

print("\nConfusion Matrix:")
print("=" * 60)
print(confusion_matrix(ground_truth, predictions))

print("\nAccuracy:", accuracy_score(ground_truth, predictions))
print(f"\nTotal samples evaluated: {len(ground_truth)}")
