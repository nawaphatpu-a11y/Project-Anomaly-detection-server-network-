"""
ทำความสะอาดข้อมูลดิบและสร้าง feature

Usage:
    python etl.py --infile data/raw/metrics.csv --outfile data/processed/metrics_clean.csv
"""

import argparse
import os
import numpy as np
import pandas as pd

SEVERITY_MAP = {
    "CPU_SPIKE": "High",
    "NETWORK_SURGE": "High",
    "RESPONSE_DEGRADATION": "Medium",
    "Other": "Low",
    "NONE": "None",
}


def add_severity(df: pd.DataFrame, type_col: str = "anomaly_type") -> pd.DataFrame:
    df["severity_level"] = df[type_col].map(SEVERITY_MAP).fillna("None")
    return df


def fix_data_errors(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    invalid_cpu = (df["cpu_usage"] < 0) | (df["cpu_usage"] > 100)
    invalid_mem = (df["memory_usage"] < 0) | (df["memory_usage"] > 100)
    invalid_resp = df["response_time"] < 0
    invalid_net_in = df["network_in"] < 0
    invalid_net_out = df["network_out"] < 0

    n_invalid = (invalid_cpu | invalid_mem | invalid_resp | invalid_net_in | invalid_net_out).sum()

    df.loc[invalid_cpu, "cpu_usage"] = np.nan
    df.loc[invalid_mem, "memory_usage"] = np.nan
    df.loc[invalid_resp, "response_time"] = np.nan
    df.loc[invalid_net_in, "network_in"] = np.nan
    df.loc[invalid_net_out, "network_out"] = np.nan

    print(f"[fix_data_errors] found {n_invalid:,} impossible values (set to NaN) out of {before:,} rows")
    return df


def fix_timestamp_errors(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    parsed = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    n_bad = int(parsed.isna().sum())

    df = df.loc[parsed.notna()].copy()
    df["timestamp"] = parsed.loc[parsed.notna()]

    print(f"[fix_timestamp_errors] found {n_bad:,} rows with malformed timestamp -> removed (out of {before:,} rows)")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["cpu_usage", "memory_usage", "network_in", "network_out", "response_time"]
    n_missing_before = df[metric_cols].isna().sum().sum()

    df = df.sort_values(["server_id", "timestamp"])
    df[metric_cols] = df.groupby("server_id")[metric_cols].transform(lambda s: s.ffill().bfill())

    n_missing_after = df[metric_cols].isna().sum().sum()
    print(f"[handle_missing] missing before: {n_missing_before:,} -> after: {n_missing_after:,}")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_peak_hour"] = df["hour"].between(9, 17).astype(int)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df["network_total"] = df["network_in"] + df["network_out"]
    df["cpu_memory_ratio"] = df["cpu_usage"] / df["memory_usage"].replace(0, np.nan)
    df["cpu_memory_ratio"] = df["cpu_memory_ratio"].fillna(df["cpu_memory_ratio"].median())
    return df


ROLE_METRICS = ["cpu_usage", "memory_usage", "network_in", "network_out", "response_time"]


def compute_role_baselines(df: pd.DataFrame) -> dict:
    def stats(g: pd.DataFrame) -> dict:
        out = {}
        for m in ROLE_METRICS:
            med = float(g[m].median())
            mad = float((g[m] - med).abs().median())
            out[m] = {"median": med, "mad": mad if mad > 0 else 1.0}
        return out

    baselines = {role: stats(g) for role, g in df.groupby("role")}
    baselines["__global__"] = stats(df)
    return baselines


def add_role_relative_features(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    df = df.copy()
    roles = df["role"] if "role" in df.columns else pd.Series("__global__", index=df.index)
    for m in ROLE_METRICS:
        med = roles.map(lambda r: baselines.get(r, baselines["__global__"])[m]["median"])
        mad = roles.map(lambda r: baselines.get(r, baselines["__global__"])[m]["mad"])
        df[f"{m}_role_z"] = (df[m] - med) / mad
    return df


def add_trend_feature(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    roll_mean = df.groupby("server_id")["response_time"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).mean()
    )
    df["response_time_dev_from_roll"] = (df["response_time"] - roll_mean).fillna(0)
    return df


def main():
    parser = argparse.ArgumentParser(description="ETL: clean raw metrics data")
    parser.add_argument("--infile", type=str, default="data/raw/metrics.csv")
    parser.add_argument("--outfile", type=str, default="data/processed/metrics_clean.csv")
    args = parser.parse_args()

    print(f"Loading: {args.infile}")
    df = pd.read_csv(args.infile)
    print(f"Loaded: {len(df):,} rows")

    df = fix_data_errors(df)
    df = fix_timestamp_errors(df)
    df = handle_missing(df)
    df = add_time_features(df)
    df = add_engineered_features(df)
    df = add_trend_feature(df)
    df = add_severity(df)


    out_dir = os.path.dirname(args.outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.outfile, index=False)

    print(f"Saved cleaned file -> {args.outfile}")
    print(f"True Anomaly rows preserved: {(df['is_anomaly'] == 1).sum():,} (none removed)")


if __name__ == "__main__":
    main()
