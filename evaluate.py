"""
evaluate.py
Evaluate prediction results against the injected ground-truth labels (Step 5: Evaluation)

Usage:
    python evaluate.py --infile data/processed/predictions.csv
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")  # ไม่มีหน้าจอ (รันเป็น script) กัน error หา display ไม่เจอ
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

MODEL_COLS = {
    "pred_isoforest": "Isolation Forest (main)",
    "pred_lof": "LOF (benchmark)",
    "pred_zscore": "Z-score (benchmark)",
    "pred_iqr": "IQR (benchmark)",
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


def plot_true_vs_predicted(df: pd.DataFrame, out_path: str):
    # Visualization-based Evaluation ตามแผน Step 5 -- ของเดิมมีแต่ตัวเลขในตาราง ไม่มี
    # กราฟเก็บเป็นไฟล์ให้ตรวจสอบด้วยตา จึงเพิ่มมา: scatter 2 แผงเทียบกัน ซ้าย = ความจริง
    # (is_anomaly ที่ generate_data.py ใส่ไว้), ขวา = สิ่งที่ Isolation Forest ทาย ถ้าสอง
    # แผงหน้าตาคล้ายกัน แปลว่าโมเดล flag จุดที่ควร flag ได้ตรงกับความจริงจริงๆ
    #
    # FIX: label ในกราฟใช้ภาษาอังกฤษล้วน -- matplotlib เขียนไฟล์ด้วยฟอนต์ default
    # (DejaVu Sans) ที่ไม่มีตัวอักษรไทย ถ้าใส่ label ไทยจะได้ไฟล์ที่ตัวอักษรหายเป็นกล่องๆ
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

    plot_path = os.path.join(out_dir or ".", "evaluation_scatter.png")
    plot_true_vs_predicted(df, plot_path)
    print(f"Saved visualization -> {plot_path}")


if __name__ == "__main__":
    main()
