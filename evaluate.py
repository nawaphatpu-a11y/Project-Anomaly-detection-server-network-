"""
evaluate.py
Evaluate prediction results against the injected ground-truth labels (Step 5: Evaluation)

Usage:
    python evaluate.py --infile data/processed/predictions.csv
"""

import argparse
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

MODEL_COLS = {
    "pred_isoforest": "Isolation Forest (main)",
    "pred_lof": "LOF (benchmark)",
    "pred_zscore": "Z-score (benchmark)",
}


def evaluate_one(y_true, y_pred, label: str) -> dict:
    # FIX: ระบุ labels=[0, 1] ตรงๆ กัน confusion_matrix คืนมาไม่ครบ 2x2
    # (ถ้าข้อมูลชุดเล็กมากจนโมเดลไม่ทายว่ามี anomaly เลยสักแถว .ravel() จะ error
    #  เพราะได้ matrix ขนาด 1x1 แทนที่จะเป็น 2x2)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "model": label, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection predictions")
    # FIX: default ให้ตรงกับ default --outfile ของ train_model.py
    parser.add_argument("--infile", type=str, default="data/processed/predictions.csv")
    parser.add_argument("--outfile", type=str, default="data/processed/evaluation_report.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.infile)
    y_true = df["is_anomaly"]

    results = []
    for col, label in MODEL_COLS.items():
        results.append(evaluate_one(y_true, df[col], label))

    report = pd.DataFrame(results).set_index("model")
    report[["precision", "recall", "f1"]] = report[["precision", "recall", "f1"]].round(3)

    print(report.to_string())

    out_dir = os.path.dirname(args.outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)  # FIX: กัน FileNotFoundError ถ้ายังไม่มีโฟลเดอร์ปลายทาง

    report.to_csv(args.outfile)
    print(f"\nSaved -> {args.outfile}")


if __name__ == "__main__":
    main()
