"""
Re-generate the training data with several seeds and score every model on the independent test set.

Usage:
    python scripts/robustness_sweep.py --seeds 42 7 11 23 5 --workdir data/sweep
"""

import argparse
import os
import subprocess
import sys

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import etl
import evaluate_independent as E
import train_model as T


def prepare_independent(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = etl.fix_data_errors(df)
    df = etl.fix_timestamp_errors(df)
    df = etl.handle_missing(df)
    df = etl.add_time_features(df)
    df = etl.add_engineered_features(df)
    return etl.add_trend_feature(df, window=12)


def ensure_clean_data(seed: int, workdir: str) -> str:
    folder = os.path.join(workdir, f"seed_{seed}")
    clean = os.path.join(folder, "clean.csv")
    if os.path.exists(clean):
        return clean
    os.makedirs(folder, exist_ok=True)
    raw = os.path.join(folder, "raw.csv")
    subprocess.run([sys.executable, os.path.join(ROOT, "generate_data.py"), "--seed", str(seed), "--outfile", raw],
                   check=True, capture_output=True, cwd=ROOT)
    subprocess.run([sys.executable, os.path.join(ROOT, "etl.py"), "--infile", raw, "--outfile", clean],
                   check=True, capture_output=True, cwd=ROOT)
    return clean


def fit_forest(train: pd.DataFrame, cols: list, seed: int) -> IsolationForest:
    return IsolationForest(contamination=0.03, n_estimators=200, random_state=seed, n_jobs=-1).fit(train[cols].fillna(0))


def scores(y_true, y_pred) -> dict:
    return {"precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3)}


def run_seed(clean: pd.DataFrame, independent: pd.DataFrame) -> list:
    rows = []
    split_train, split_test = T.time_based_split(clean, 0.2)
    split_scale = T.fit_min_max(split_train, T.NORMALIZE_COLS)
    split_train = T.apply_min_max(split_train, T.NORMALIZE_COLS, split_scale)
    split_test = T.apply_min_max(split_test, T.NORMALIZE_COLS, split_scale)
    split_baselines = etl.compute_role_baselines(split_train)
    split_train = etl.add_role_relative_features(split_train, split_baselines)
    split_test = etl.add_role_relative_features(split_test, split_baselines)

    scale = T.fit_min_max(clean, T.NORMALIZE_COLS)
    full = T.apply_min_max(clean, T.NORMALIZE_COLS, scale)
    baselines = etl.compute_role_baselines(full)
    full = etl.add_role_relative_features(full, baselines)
    ind = E.apply_min_max(independent, list(scale.keys()), scale)
    ind = etl.add_role_relative_features(ind, baselines)

    for name, cols in (("Isolation Forest", T.ISO_FEATURE_COLS),
                       ("Isolation Forest + role_z", T.ISO_FEATURE_COLS + T.ROLE_Z_COLS)):
        in_dist = (fit_forest(split_train, cols, 42).predict(split_test[cols].fillna(0)) == -1).astype(int)
        pred = (fit_forest(full, cols, 42).predict(ind[cols].fillna(0)) == -1).astype(int)
        rows.append({"model": name, "test_split_f1": round(f1_score(split_test["is_anomaly"], in_dist), 3),
                     **scores(ind["is_anomaly"], pred)})

    z_in = T.zscore_baseline(split_train[T.ZSCORE_FEATURE_COLS].fillna(0), split_test[T.ZSCORE_FEATURE_COLS].fillna(0))
    z_ind = T.zscore_baseline(full[T.ZSCORE_FEATURE_COLS].fillna(0), ind[T.ZSCORE_FEATURE_COLS].fillna(0))
    rows.append({"model": "Z-score", "test_split_f1": round(f1_score(split_test["is_anomaly"], z_in), 3),
                 **scores(ind["is_anomaly"], z_ind)})
    q_in = T.iqr_baseline(split_train[T.IQR_FEATURE_COLS].fillna(0), split_test[T.IQR_FEATURE_COLS].fillna(0))
    q_ind = T.iqr_baseline(full[T.IQR_FEATURE_COLS].fillna(0), ind[T.IQR_FEATURE_COLS].fillna(0))
    rows.append({"model": "IQR", "test_split_f1": round(f1_score(split_test["is_anomaly"], q_in), 3),
                 **scores(ind["is_anomaly"], q_ind)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Score every model on the independent set across several training seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 11, 23, 5])
    parser.add_argument("--workdir", type=str, default="data/sweep")
    parser.add_argument("--independent", type=str, default=os.path.join(ROOT, "independent_test_set_v2.csv"))
    parser.add_argument("--outfile", type=str, default=os.path.join(ROOT, "data/processed/robustness_sweep.csv"))
    args = parser.parse_args()

    independent = prepare_independent(args.independent)
    all_rows = []
    for seed in args.seeds:
        clean = pd.read_csv(ensure_clean_data(seed, args.workdir))
        for row in run_seed(clean, independent):
            all_rows.append({"seed": seed, **row})
        print(f"seed {seed} done", flush=True)

    result = pd.DataFrame(all_rows)
    summary = result.groupby("model")[["test_split_f1", "precision", "recall", "f1"]].agg(["mean", "min", "max"]).round(3)
    print(summary.to_string())
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    result.to_csv(args.outfile, index=False)
    print(f"Saved -> {args.outfile}")


if __name__ == "__main__":
    main()
