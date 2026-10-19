"""
สร้างข้อมูล metric ของเซิร์ฟเวอร์จำลอง พร้อมแทรกความผิดปกติและข้อมูลผิดพลาด

Usage:
    python generate_data.py --servers 15 --days 30 --interval 5 --outfile data/raw/metrics.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SERVER_ROLES = ["Web Server", "Database Server", "App Server"]


SERVER_ROLES = ["Web Server", "Database Server", "App Server"]
SERVER_LOCATIONS = ["Bangkok-DC1", "Bangkok-DC2", "Nonthaburi-DC1"]


ROLE_BASELINES = {
    "Web Server":      {"cpu": (22, 36), "mem": (25, 40), "net_in": (120, 220), "net_out": (150, 260), "resp": (40, 90)},
    "Database Server": {"cpu": (30, 45), "mem": (52, 68), "net_in": (50, 100),  "net_out": (40, 90),   "resp": (90, 180)},
    "App Server":      {"cpu": (40, 58), "mem": (40, 55), "net_in": (70, 130),  "net_out": (70, 130),  "resp": (60, 120)},
}


def make_server_profiles(n_servers: int, rng: np.random.Generator) -> pd.DataFrame:
    roles = np.array([SERVER_ROLES[i % len(SERVER_ROLES)] for i in range(n_servers)])
    roles = rng.permutation(roles)
    locations = rng.choice(SERVER_LOCATIONS, size=n_servers)

    def draw(metric: str) -> np.ndarray:
        return np.array([rng.uniform(*ROLE_BASELINES[r][metric]) for r in roles])

    profiles = pd.DataFrame({
        "server_id": [f"SRV-{i+1:03d}" for i in range(n_servers)],
        "hostname": [f"host-{i+1:03d}" for i in range(n_servers)],
        "role": roles,
        "location": locations,
        "base_cpu": draw("cpu"),
        "base_mem": draw("mem"),
        "base_net_in": draw("net_in"),
        "base_net_out": draw("net_out"),
        "base_resp": draw("resp"),
    })
    return profiles


def is_peak_hour(hour: int) -> bool:
    return 9 <= hour < 18


def generate_baseline(n_rows: int, hours: np.ndarray, profile: pd.Series, rng: np.random.Generator) -> dict:
    peak_mask = np.isin(hours, list(range(9, 18))).astype(float)

    cpu = profile["base_cpu"] + peak_mask * rng.uniform(5, 15) + rng.normal(0, 4, n_rows)
    mem = profile["base_mem"] + peak_mask * rng.uniform(3, 10) + rng.normal(0, 3, n_rows)
    net_in = profile["base_net_in"] + peak_mask * rng.uniform(20, 60) + rng.normal(0, 10, n_rows)
    net_out = profile["base_net_out"] + peak_mask * rng.uniform(20, 60) + rng.normal(0, 10, n_rows)
    resp = profile["base_resp"] + peak_mask * rng.uniform(10, 30) + rng.normal(0, 8, n_rows)

    cpu = np.clip(cpu, 1, 99)
    mem = np.clip(mem, 1, 99)
    net_in = np.clip(net_in, 1, None)
    net_out = np.clip(net_out, 1, None)
    resp = np.clip(resp, 1, None)

    return {"cpu_usage": cpu, "memory_usage": mem, "network_in": net_in,
            "network_out": net_out, "response_time": resp}


def inject_true_anomalies(df: pd.DataFrame, rng: np.random.Generator, target_ratio: float = 0.03):
    n = len(df)
    target_anomaly_rows = int(n * target_ratio)
    anomaly_types = ["CPU_SPIKE", "NETWORK_SURGE", "RESPONSE_DEGRADATION"]
    rows_done = 0
    server_ids = df["server_id"].unique()

    max_attempts = target_anomaly_rows * 20 + 1000
    attempts = 0

    while rows_done < target_anomaly_rows and attempts < max_attempts:
        attempts += 1
        server = rng.choice(server_ids)
        server_rows = df.index[df["server_id"] == server]
        if len(server_rows) < 30:
            continue

        start = rng.integers(0, len(server_rows) - 20)
        episode_len = rng.integers(6, 20)
        idx = server_rows[start:start + episode_len]
        atype = rng.choice(anomaly_types)

        if atype == "CPU_SPIKE":
            df.loc[idx, "cpu_usage"] = rng.uniform(92, 100, size=len(idx))
            df.loc[idx, "memory_usage"] += rng.uniform(10, 20, size=len(idx))
            df.loc[idx, "memory_usage"] = df.loc[idx, "memory_usage"].clip(upper=99)
        elif atype == "NETWORK_SURGE":
            df.loc[idx, "network_in"] *= rng.uniform(5, 10)
            df.loc[idx, "network_out"] *= rng.uniform(5, 10)
        elif atype == "RESPONSE_DEGRADATION":
            ramp = np.linspace(1.2, 4.0, num=len(idx))
            df.loc[idx, "response_time"] *= ramp

        df.loc[idx, "is_anomaly"] = 1
        df.loc[idx, "anomaly_type"] = atype
        rows_done += len(idx)

    if attempts >= max_attempts:
        print(f"[inject_true_anomalies] warning: hit max_attempts, got {rows_done:,}/{target_anomaly_rows:,} target rows")

    return df


def inject_data_errors(df: pd.DataFrame, rng: np.random.Generator, error_ratio: float = 0.015):
    normal_idx = df.index[df["is_anomaly"] == 0]
    n_errors = int(len(df) * error_ratio)
    n_errors = min(n_errors, len(normal_idx))
    error_idx = rng.choice(normal_idx, size=n_errors, replace=False)

    for i in error_idx:
        choice = rng.integers(0, 4)
        if choice == 0:
            df.loc[i, "cpu_usage"] = rng.uniform(101, 150)
        elif choice == 1:
            df.loc[i, "response_time"] = -abs(rng.uniform(1, 50))
        elif choice == 2:
            df.loc[i, "memory_usage"] = np.nan
        else:
            df.loc[i, "network_in"] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate simulated server/network monitoring data")
    parser.add_argument("--servers", type=int, default=15, help="number of simulated servers")
    parser.add_argument("--days", type=int, default=30, help="number of past days to simulate")
    parser.add_argument("--interval", type=int, default=5, help="reading interval in minutes")
    parser.add_argument("--anomaly-ratio", type=float, default=0.03, help="target ratio of True Anomaly rows")
    parser.add_argument("--error-ratio", type=float, default=0.015, help="target ratio of Data Error rows")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--start", type=str, default="2026-01-05", help="timestamp of the first reading (fixed for reproducible output)")
    parser.add_argument("--outfile", type=str, default="data/raw/metrics.csv", help="output file path")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    readings_per_day = int(24 * 60 / args.interval)
    n_rows_per_server = readings_per_day * args.days
    total_rows = n_rows_per_server * args.servers
    print(f"Generating data: {args.servers} servers x {n_rows_per_server} readings = {total_rows:,} rows")

    profiles = make_server_profiles(args.servers, rng)

    start_time = datetime.fromisoformat(args.start)
    timestamps = [start_time + timedelta(minutes=args.interval * i) for i in range(n_rows_per_server)]
    hours = np.array([t.hour for t in timestamps])

    all_frames = []
    for _, profile in profiles.iterrows():
        baseline = generate_baseline(n_rows_per_server, hours, profile, rng)
        frame = pd.DataFrame({
            "server_id": profile["server_id"],
            "hostname": profile["hostname"],
            "role": profile["role"],
            "location": profile["location"],
            "timestamp": timestamps,
            **baseline,
        })
        frame["is_anomaly"] = 0
        frame["anomaly_type"] = "NONE"
        all_frames.append(frame)

    df = pd.concat(all_frames, ignore_index=True)
    df = df.sort_values(["server_id", "timestamp"]).reset_index(drop=True)

    df = inject_true_anomalies(df, rng, target_ratio=args.anomaly_ratio)
    df = inject_data_errors(df, rng, error_ratio=args.error_ratio)

    out_dir = os.path.dirname(args.outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.outfile, index=False)

    n_anom = (df["is_anomaly"] == 1).sum()
    n_missing = df[["cpu_usage", "memory_usage", "network_in", "network_out", "response_time"]].isna().any(axis=1).sum()
    print(f"Done -> {args.outfile}")
    print(f"  Total rows      : {len(df):,}")
    print(f"  True Anomaly    : {n_anom:,} ({n_anom/len(df)*100:.2f}%)")
    print(f"  Rows w/ Missing : {n_missing:,}")


if __name__ == "__main__":
    main()
