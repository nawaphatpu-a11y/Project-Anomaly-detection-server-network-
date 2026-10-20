"""
เทรน Isolation Forest (โมเดลหลัก) และ LOF, Z-score, IQR (benchmark) โดยแบ่ง train/test ตามเวลาแยกทีละ server

Usage:
    python train_model.py --infile data/processed/metrics_clean.csv --outfile data/processed/predictions.csv
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from detection import zscore_stats
from etl import ROLE_METRICS, compute_role_baselines, add_role_relative_features

NORMALIZE_COLS = [
    "cpu_usage", "memory_usage", "network_in", "network_out",
    "response_time", "network_total", "cpu_memory_ratio",
    "response_time_dev_from_roll",
]

BASE_FEATURE_COLS = [
    "cpu_usage_norm", "memory_usage_norm", "network_in_norm", "network_out_norm",
    "response_time_norm", "network_total_norm", "cpu_memory_ratio_norm",
    "hour", "day_of_week", "is_peak_hour",
]

TREND_COL_RAW = "response_time_dev_from_roll"
TREND_COL_NORM = "response_time_dev_from_roll_norm"

ROLE_Z_COLS = [f"{m}_role_z" for m in ROLE_METRICS]

ISO_FEATURE_COLS = BASE_FEATURE_COLS + [TREND_COL_RAW]
LOF_FEATURE_COLS = BASE_FEATURE_COLS + [TREND_COL_NORM]
ZSCORE_FEATURE_COLS = BASE_FEATURE_COLS + [TREND_COL_RAW]
IQR_FEATURE_COLS = ZSCORE_FEATURE_COLS

ALL_FEATURE_COLS = BASE_FEATURE_COLS + [TREND_COL_RAW, TREND_COL_NORM] + ROLE_Z_COLS


def fit_min_max(source_df: pd.DataFrame, cols: list) -> dict:
    scale_info = {}
    for col in cols:
        col_min, col_max = source_df[col].min(), source_df[col].max()
        scale_info[col] = {"min": float(col_min), "max": float(col_max)}
    return scale_info


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


def time_based_split(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """แบ่ง train/test ตามเวลา แยกทีละ server_id เพื่อกัน data leakage ข้าม episode"""
    df = df.sort_values(["server_id", "timestamp"])
    train_parts, test_parts = [], []
    for _, group in df.groupby("server_id", sort=False):
        cut = int(len(group) * (1 - test_ratio))
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)


def train_isolation_forest(X_train: pd.DataFrame, X_test: pd.DataFrame,
                            contamination: float, seed: int) -> np.ndarray:
    model = IsolationForest(contamination=contamination, n_estimators=200,
                             random_state=seed, n_jobs=-1)
    model.fit(X_train)
    pred = model.predict(X_test)
    return (pred == -1).astype(int)


def train_lof(X_train: pd.DataFrame, X_test: pd.DataFrame, contamination: float) -> np.ndarray:
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination,
                                novelty=True, n_jobs=-1)
    model.fit(X_train.to_numpy())
    pred = model.predict(X_test.to_numpy())
    return (pred == -1).astype(int)


def zscore_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 3.0) -> np.ndarray:
    mean, std = X_train.mean(), X_train.std(ddof=0).replace(0, 1)
    z = (X_test - mean) / std
    return (z.abs() > threshold).any(axis=1).astype(int)


def iqr_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame, k: float = 1.5) -> np.ndarray:
    q1, q3 = X_train.quantile(0.25), X_train.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return ((X_test < lower) | (X_test > upper)).any(axis=1).astype(int)


REPLAY_COLUMNS = ["server_id", "role", "timestamp", "cpu_usage", "memory_usage", "network_in", "network_out",
                  "response_time", "is_anomaly", "anomaly_type"]


def build_replay_sample(test_df: pd.DataFrame, rows_per_server: int = 400) -> pd.DataFrame:
    parts = [group.sort_values("timestamp").head(rows_per_server) for _, group in test_df.groupby("server_id")]
    return pd.concat(parts)[REPLAY_COLUMNS].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection models with a held-out test split")
    parser.add_argument("--infile", type=str, default="data/processed/metrics_clean.csv")
    parser.add_argument("--outfile", type=str, default="data/processed/predictions.csv")
    parser.add_argument("--contamination", type=float, default=0.03)
    parser.add_argument("--test-ratio", type=float, default=0.2,
                         help="สัดส่วนข้อมูล 'ใหม่สุด' ต่อ server ที่กันไว้เป็น test (ไม่ใช้ตอน fit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="models/isolation_forest.joblib",
                         help="ที่เก็บโมเดล Isolation Forest ตัวสุดท้าย (เทรนจากข้อมูลทั้งหมด) สำหรับ live scoring")
    args = parser.parse_args()

    print(f"Loading: {args.infile}")
    df = pd.read_csv(args.infile)

    if df["timestamp"].pipe(pd.to_datetime).dt.normalize().nunique() < 10:
        print("[warn] ข้อมูลครอบคลุมน้อยกว่า 10 วัน — day_of_week อาจทำให้ LOF ประเมินผิดพลาดได้ "
              "แนะนำ generate_data.py --days อย่างน้อย 14 วันขึ้นไป")

    train_df, test_df = time_based_split(df, args.test_ratio)
    print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows (test = {args.test_ratio:.0%} ล่าสุดของแต่ละ server)")

    eval_scale_info = fit_min_max(train_df, NORMALIZE_COLS)
    train_df = apply_min_max(train_df, NORMALIZE_COLS, eval_scale_info)
    test_df = apply_min_max(test_df, NORMALIZE_COLS, eval_scale_info)

    eval_role_baselines = compute_role_baselines(train_df)
    train_df = add_role_relative_features(train_df, eval_role_baselines)
    test_df = add_role_relative_features(test_df, eval_role_baselines)

    print("Training Isolation Forest (main prototype model)...")
    Xtr, Xte = train_df[ISO_FEATURE_COLS].fillna(0), test_df[ISO_FEATURE_COLS].fillna(0)
    test_df["pred_isoforest"] = train_isolation_forest(Xtr, Xte, args.contamination, args.seed)

    print("Training LOF (benchmark only)...")
    Xtr, Xte = train_df[LOF_FEATURE_COLS].fillna(0), test_df[LOF_FEATURE_COLS].fillna(0)
    test_df["pred_lof"] = train_lof(Xtr, Xte, args.contamination)

    print("Computing Z-score baseline (benchmark only)...")
    Xtr, Xte = train_df[ZSCORE_FEATURE_COLS].fillna(0), test_df[ZSCORE_FEATURE_COLS].fillna(0)
    test_df["pred_zscore"] = zscore_baseline(Xtr, Xte)

    print("Computing IQR baseline (benchmark only)...")
    Xtr, Xte = train_df[IQR_FEATURE_COLS].fillna(0), test_df[IQR_FEATURE_COLS].fillna(0)
    test_df["pred_iqr"] = iqr_baseline(Xtr, Xte)

    keep_cols = ["server_id", "timestamp", "is_anomaly", "anomaly_type", "severity_level",
                 "cpu_usage", "response_time",
                 "pred_isoforest", "pred_lof", "pred_zscore", "pred_iqr"] + ALL_FEATURE_COLS

    out_dir = os.path.dirname(args.outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    test_df[keep_cols].to_csv(args.outfile, index=False)

    print(f"Saved predictions (test set only) -> {args.outfile}")

    replay_path = os.path.join(out_dir or ".", "replay_sample.csv")
    replay = build_replay_sample(test_df)
    replay.to_csv(replay_path, index=False)
    print(f"Saved replay sample for the live dashboard -> {replay_path} "
          f"({len(replay):,} rows, {replay['is_anomaly'].mean():.1%} anomalies)")
    for col in ["pred_isoforest", "pred_lof", "pred_zscore", "pred_iqr"]:
        print(f"  {col}: flagged {int(test_df[col].sum()):,} rows as anomaly (out of {len(test_df):,} test rows)")

    print("Training final Isolation Forest on ALL data (for live deployment)...")
    deploy_scale_info = fit_min_max(df, NORMALIZE_COLS)
    df = apply_min_max(df, NORMALIZE_COLS, deploy_scale_info)
    deploy_role_baselines = compute_role_baselines(df)
    df = add_role_relative_features(df, deploy_role_baselines)

    X_all = df[ISO_FEATURE_COLS].fillna(0)
    final_model = IsolationForest(contamination=args.contamination, n_estimators=200,
                                   random_state=args.seed, n_jobs=-1)
    final_model.fit(X_all)

    model_dir = os.path.dirname(args.model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    server_roles = df.drop_duplicates("server_id").set_index("server_id")["role"].to_dict()

    joblib.dump({
        "model": final_model,
        "feature_cols": ISO_FEATURE_COLS,
        "trend_window": 12,
        "server_ids": sorted(df["server_id"].unique().tolist()),
        "server_roles": server_roles,
        "role_baselines": deploy_role_baselines,
        "zscore_stats": zscore_stats(df[ZSCORE_FEATURE_COLS].fillna(0), ZSCORE_FEATURE_COLS),
    }, args.model_out)
    print(f"Saved deployment model -> {args.model_out}")

    scale_path = os.path.join(out_dir or ".", "scale_params.json")
    with open(scale_path, "w", encoding="utf-8") as f:
        json.dump(deploy_scale_info, f, ensure_ascii=False, indent=2)
    print(f"Saved scale params (full-data, for live scoring) -> {scale_path}")


if __name__ == "__main__":
    main()
