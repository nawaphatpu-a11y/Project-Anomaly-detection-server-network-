"""
ประเมินโมเดลที่เซฟไว้กับชุดข้อมูลทดสอบอิสระ (สร้างจาก AI คนละตัว)

Usage:
    python evaluate_independent.py --infile independent_test_set_v2.csv
"""

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from etl import (fix_data_errors, fix_timestamp_errors, handle_missing, add_time_features,
                  add_engineered_features, add_trend_feature, add_severity,
                  add_role_relative_features)
from evaluate import plot_true_vs_predicted

MODEL_PATH = "models/isolation_forest.joblib"
SCALE_PARAMS_PATH = "data/processed/scale_params.json"


def apply_min_max(df: pd.DataFrame, cols: list, scale_info: dict) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        col_min, col_max = scale_info[col]["min"], scale_info[col]["max"]
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"{col}_norm"] = 0.0
        df[f"{col}_norm"] = df[f"{col}_norm"].clip(0.0, 1.0)
    return df


def score_with_bundle(df: pd.DataFrame, bundle: dict, scale_info: dict):
    df = df.copy()
    if bundle.get("role_baselines"):
        df = add_role_relative_features(df, bundle["role_baselines"])
    df = apply_min_max(df, list(scale_info.keys()), scale_info)
    X = df[bundle["feature_cols"]].fillna(0)
    return (bundle["model"].predict(X) == -1).astype(int)


def metrics_row(label: str, y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "dataset": label, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved model on an independently-generated test set")
    parser.add_argument("--infile", type=str, default="independent_test_set.csv")
    parser.add_argument("--outfile", type=str, default="data/processed/evaluation_report_independent.csv")
    args = parser.parse_args()

    print(f"Loading independent test set: {args.infile}")
    df = pd.read_csv(args.infile)
    print(f"Loaded: {len(df):,} rows")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    window = bundle["trend_window"]

    with open(SCALE_PARAMS_PATH, "r", encoding="utf-8") as f:
        scale_info = json.load(f)

    df = fix_data_errors(df)
    df = fix_timestamp_errors(df)
    df = handle_missing(df)
    df = add_time_features(df)
    df = add_engineered_features(df)
    df = add_trend_feature(df, window=window)
    df = add_severity(df)
    df["pred_isoforest"] = score_with_bundle(df, bundle, scale_info)
    if bundle.get("role_baselines"):
        df = add_role_relative_features(df, bundle["role_baselines"])
    df = apply_min_max(df, list(scale_info.keys()), scale_info)

    report = pd.DataFrame([metrics_row("Independent test set (ChatGPT-generated)",
                                       df["is_anomaly"], df["pred_isoforest"])]).set_index("dataset")

    print(report.to_string())

    out_dir = os.path.dirname(args.outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    report.to_csv(args.outfile)
    print(f"\nSaved -> {args.outfile}")

    plot_path = os.path.join(out_dir or ".", "evaluation_scatter_independent.png")
    plot_true_vs_predicted(df, plot_path)
    print(f"Saved visualization -> {plot_path}")


if __name__ == "__main__":
    main()
