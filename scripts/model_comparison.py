"""
Extra comparison between Isolation Forest and Z-score:
threshold sensitivity and an OR/AND ensemble on the independent set, plus a contextual-anomaly stress test.

Usage:
    python scripts/model_comparison.py --seeds 42 7 11 23 5 --workdir data/sweep
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robustness_sweep as R

E, T, etl, ROOT = R.E, R.T, R.etl, R.ROOT

IF_CONTAMINATION = (0.01, 0.02, 0.03, 0.05)
Z_THRESHOLDS = (2.5, 3.0, 3.5, 4.0)
STRESS_LEVELS = {
    "moderate": {"cpu_usage": 12, "memory_usage": 6, "network_in": 40, "network_out": 40, "response_time": 15},
    "strong": {"cpu_usage": 20, "memory_usage": 10, "network_in": 70, "network_out": 70, "response_time": 25},
}
OFF_PEAK_HOURS = (8, 18)
STRESS_ROWS = 300


def forest_predict(train, test, contamination):
    model = IsolationForest(contamination=contamination, n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(train[T.ISO_FEATURE_COLS].fillna(0))
    return (model.predict(test[T.ISO_FEATURE_COLS].fillna(0)) == -1).astype(int)


def zscore_predict(train, test, threshold):
    return T.zscore_baseline(train[T.ZSCORE_FEATURE_COLS].fillna(0), test[T.ZSCORE_FEATURE_COLS].fillna(0), threshold)


def result(experiment, seed, model, setting, y_true, y_pred):
    normal = (np.asarray(y_true) == 0)
    return {"experiment": experiment, "seed": seed, "model": model, "setting": setting,
            **R.scores(y_true, y_pred), "false_alarm_rate": round(float(np.asarray(y_pred)[normal].mean()), 4)}


def independent_experiments(seed, clean, independent):
    scale = T.fit_min_max(clean, T.NORMALIZE_COLS)
    full = T.apply_min_max(clean, T.NORMALIZE_COLS, scale)
    ind = E.apply_min_max(independent, list(scale.keys()), scale)
    y = ind["is_anomaly"]
    rows, forests, zscores = [], {}, {}
    for contamination in IF_CONTAMINATION:
        forests[contamination] = forest_predict(full, ind, contamination)
        rows.append(result("independent", seed, "Isolation Forest", f"contamination={contamination}", y, forests[contamination]))
    for threshold in Z_THRESHOLDS:
        zscores[threshold] = zscore_predict(full, ind, threshold)
        rows.append(result("independent", seed, "Z-score", f"threshold={threshold}", y, zscores[threshold]))
    rows.append(result("independent", seed, "IF OR Z-score", "defaults", y, forests[0.03] | zscores[3.0]))
    rows.append(result("independent", seed, "IF AND Z-score", "defaults", y, forests[0.03] & zscores[3.0]))
    return rows


def stress_experiments(seed, clean):
    train_raw, test_raw = T.time_based_split(clean, 0.2)
    scale = T.fit_min_max(train_raw, T.NORMALIZE_COLS)
    train = T.apply_min_max(train_raw, T.NORMALIZE_COLS, scale)
    model = IsolationForest(contamination=0.03, n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(train[T.ISO_FEATURE_COLS].fillna(0))
    rng = np.random.default_rng(seed)
    rows = []
    for level, bumps in STRESS_LEVELS.items():
        test = test_raw.copy()
        off_peak = test.index[(~test["hour"].between(*OFF_PEAK_HOURS)) & (test["is_anomaly"] == 0)]
        chosen = rng.choice(off_peak, size=min(STRESS_ROWS, len(off_peak) // 10), replace=False)
        test["injected"] = 0
        test.loc[chosen, "injected"] = 1
        for column, bump in bumps.items():
            test.loc[chosen, column] = test.loc[chosen, column] + bump
        test["cpu_usage"] = test["cpu_usage"].clip(upper=99)
        test["memory_usage"] = test["memory_usage"].clip(upper=99)
        test = etl.add_engineered_features(test)
        test = etl.add_trend_feature(test, window=12)
        test = T.apply_min_max(test, T.NORMALIZE_COLS, scale)
        scored = test[test["is_anomaly"] == 0]
        y = scored["injected"]
        iso = (model.predict(scored[T.ISO_FEATURE_COLS].fillna(0)) == -1).astype(int)
        zsc = zscore_predict(train, scored, 3.0)
        rows.append(result("contextual_stress", seed, "Isolation Forest", level, y, iso))
        rows.append(result("contextual_stress", seed, "Z-score", level, y, zsc))
        rows.append(result("contextual_stress", seed, "IF OR Z-score", level, y, iso | zsc))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Isolation Forest vs Z-score: sensitivity, ensemble and stress test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 11, 23, 5])
    parser.add_argument("--workdir", type=str, default="data/sweep")
    parser.add_argument("--independent", type=str, default=os.path.join(ROOT, "independent_test_set_v2.csv"))
    parser.add_argument("--outfile", type=str, default=os.path.join(ROOT, "data/processed/model_comparison.csv"))
    args = parser.parse_args()

    independent = R.prepare_independent(args.independent)
    rows = []
    for seed in args.seeds:
        clean = pd.read_csv(R.ensure_clean_data(seed, args.workdir))
        rows += independent_experiments(seed, clean, independent)
        rows += stress_experiments(seed, clean)
        print(f"seed {seed} done", flush=True)

    new = pd.DataFrame(rows)
    if os.path.exists(args.outfile):
        old = pd.read_csv(args.outfile)
        new = pd.concat([old[~old["seed"].isin(args.seeds)], new])
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    new.sort_values(["experiment", "model", "setting", "seed"]).to_csv(args.outfile, index=False)
    print(f"Saved -> {args.outfile}")


if __name__ == "__main__":
    main()
