import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def main():
    parser = argparse.ArgumentParser(description="Compute classification performance from results CSV.")
    parser.add_argument("csv_file", type=str, help="Path to the results CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    y_true = df["gt_label"]
    y_pred = df["model_label"]

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=["CORRECT", "INCORRECT"])
    print("Confusion Matrix (rows: true, columns: predicted):")
    print(pd.DataFrame(cm, index=["True_CORRECT", "True_INCORRECT"], columns=["Pred_CORRECT", "Pred_INCORRECT"]))

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=["CORRECT", "INCORRECT"]))

if __name__ == "__main__":
    main()