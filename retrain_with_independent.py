"""
ทดลอง retrain ด้วยข้อมูลเดิมรวมกับชุดอิสระส่วนแรก แล้ววัดผลกับส่วนที่เหลือ

Usage:
    python retrain_with_independent.py --independent-infile independent_test_set_v2.csv
"""

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from etl import (fix_data_errors, fix_timestamp_errors, handle_missing, add_time_features,
                  add_engineered_features, add_trend_feature, add_severity,
                  compute_role_baselines, add_role_relative_features)
from train_model import NORMALIZE_COLS, ISO_FEATURE_COLS, fit_min_max, apply_min_max, time_based_split


def main():
    parser = argparse.ArgumentParser(description="Retrain with original + part of an independent test set")
    parser.add_argument("--metrics-clean", type=str, default="data/processed/metrics_clean.csv")
    parser.add_argument("--independent-infile", type=str, default="independent_test_set_v2.csv")
    parser.add_argument("--seen-ratio", type=float, default=0.5,
                         help="สัดส่วนของ independent set (ตามเวลา, ช่วงแรก) ที่เอาไป retrain")
    parser.add_argument("--contamination", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="models/isolation_forest_retrained.joblib")
    parser.add_argument("--outfile", type=str, default="data/processed/evaluation_report_independent_after.csv")
    args = parser.parse_args()

    print(f"Loading independent set: {args.independent_infile}")
    indep = pd.read_csv(args.independent_infile)
    indep = fix_data_errors(indep)
    indep = fix_timestamp_errors(indep)
    indep = handle_missing(indep)
    indep = add_time_features(indep)
    indep = add_engineered_features(indep)
    indep = add_trend_feature(indep)
    indep = add_severity(indep)

    seen, holdout = time_based_split(indep, test_ratio=1 - args.seen_ratio)
    print(f"Independent split: seen(retrain)={len(seen):,} rows | holdout(final test, never trained on)={len(holdout):,} rows")

    print(f"Loading original training data: {args.metrics_clean}")
    original = pd.read_csv(args.metrics_clean)
    combined = pd.concat([original, seen], ignore_index=True)
    print(f"Combined training data: {len(combined):,} rows ({len(original):,} original + {len(seen):,} from independent set)")

    scale_info = fit_min_max(combined, NORMALIZE_COLS)
    combined = apply_min_max(combined, NORMALIZE_COLS, scale_info)
    holdout = apply_min_max(holdout, NORMALIZE_COLS, scale_info)

    role_baselines = compute_role_baselines(combined)
    combined = add_role_relative_features(combined, role_baselines)
    holdout = add_role_relative_features(holdout, role_baselines)

    X_train = combined[ISO_FEATURE_COLS].fillna(0)
    model = IsolationForest(contamination=args.contamination, n_estimators=200,
                             random_state=args.seed, n_jobs=-1)
    model.fit(X_train)

    X_holdout = holdout[ISO_FEATURE_COLS].fillna(0)
    pred = (model.predict(X_holdout) == -1).astype(int)
    y_true = holdout["is_anomaly"]

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)

    report = pd.DataFrame([{
        "dataset": "Independent holdout (after retrain with combined data)",
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
    }]).set_index("dataset")
    print(report.to_string())

    out_dir = os.path.dirname(args.outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    report.to_csv(args.outfile)
    print(f"\nSaved -> {args.outfile}")

    model_dir = os.path.dirname(args.model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    server_roles = {}
    if "role" in combined.columns:
        server_roles = combined.drop_duplicates("server_id").set_index("server_id")["role"].to_dict()

    joblib.dump({
        "model": model,
        "feature_cols": ISO_FEATURE_COLS,
        "trend_window": 12,
        "server_ids": sorted(combined["server_id"].unique().tolist()),
        "server_roles": server_roles,
        "role_baselines": role_baselines,
    }, args.model_out)
    print(f"Saved retrained model -> {args.model_out} (ยังไม่ได้ใช้แทนโมเดล deploy เดิม)")

    scale_path = os.path.join(model_dir or ".", "scale_params_retrained.json")
    with open(scale_path, "w", encoding="utf-8") as f:
        json.dump(scale_info, f, ensure_ascii=False, indent=2)
    print(f"Saved scale params -> {scale_path}")


if __name__ == "__main__":
    main()
