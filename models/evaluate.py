"""
evaluate.py
Evaluate prediction results against the injected ground-truth labels (Step 5: Evaluation)

Usage:
    python evaluate.py --infile data/processed/predictions.csv
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from diagnose import diagnose_frame
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

MODEL_COLS = {
    "pred_isoforest": "Isolation Forest (main)",
    "pred_lof": "LOF (benchmark)",
    "pred_zscore": "Z-score (benchmark)",
    "pred_iqr": "IQR (benchmark)",
}


def evaluate_one(y_true, y_pred, label: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "model": label, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def plot_true_vs_predicted(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)

    for ax, col, title in [
        (axes[0], "is_anomaly", "Ground Truth"),
        (axes[1], "pred_isoforest", "Isolation Forest prediction"),
    ]:
        normal = df[df[col] == 0]
        anomaly = df[df[col] == 1]
        ax.scatter(normal["cpu_usage"], normal["response_time"], s=8, alpha=0.4, label="Normal", color="#4C78A8")
        ax.scatter(anomaly["cpu_usage"], anomaly["response_time"], s=14, alpha=0.85, label="Anomaly", color="#E45756")
        ax.set_title(title)
        ax.set_xlabel("cpu_usage (%)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("response_time (ms)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection predictions")
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
        os.makedirs(out_dir, exist_ok=True)

    report.to_csv(args.outfile)
    print(f"\nSaved -> {args.outfile}")

    type_recall = (df[df["is_anomaly"] == 1]
                   .groupby("anomaly_type")[list(MODEL_COLS)].mean()
                   .rename(columns=MODEL_COLS).round(3))
    type_recall_path = os.path.join(out_dir or ".", "recall_by_anomaly_type.csv")
    type_recall.to_csv(type_recall_path)
    print(f"Saved -> {type_recall_path}")

    if {"cpu_usage_role_z", "response_time_dev_from_roll"}.issubset(df.columns):
        causes = ["CPU_SPIKE", "NETWORK_SURGE", "RESPONSE_DEGRADATION", "UNCLASSIFIED"]
        flagged = diagnose_frame(df[df["pred_isoforest"] == 1])
        rows = {}
        for group, subset, target in [
            *[(t, flagged[(flagged["is_anomaly"] == 1) & (flagged["anomaly_type"] == t)], t) for t in causes[:3]],
            ("FALSE_ALARM", flagged[flagged["is_anomaly"] == 0], "UNCLASSIFIED"),
        ]:
            counts = subset["cause_type"].value_counts().reindex(causes, fill_value=0)
            rows[group] = {"n": len(subset), **counts.to_dict(),
                           "correct_rate": round(counts[target] / len(subset), 3) if len(subset) else float("nan")}
        cause_report = pd.DataFrame(rows).T
        cause_report[["n", *causes]] = cause_report[["n", *causes]].astype(int)
        cause_report.index.name = "true_type"
        cause_path = os.path.join(out_dir or ".", "cause_diagnosis_report.csv")
        cause_report.to_csv(cause_path)
        print("\nCause diagnosis on rows flagged by Isolation Forest")
        print(cause_report.to_string())
        print(f"Saved -> {cause_path}")

    plot_path = os.path.join(out_dir or ".", "evaluation_scatter.png")
    plot_true_vs_predicted(df, plot_path)
    print(f"Saved visualization -> {plot_path}")


if __name__ == "__main__":
    main()
