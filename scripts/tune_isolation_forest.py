"""
Hyperparameter search for the Isolation Forest using a time-ordered validation split.

Usage:
    python scripts/tune_isolation_forest.py --infile data/processed/metrics_clean.csv
"""

import argparse
import itertools
import os
import sys

import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robustness_sweep as R

E, T, ROOT = R.E, R.T, R.ROOT

GRID = {"contamination": [0.01, 0.02, 0.03, 0.05], "n_estimators": [100, 200], "max_samples": ["auto", 512]}
DEFAULT = {"contamination": 0.03, "n_estimators": 200, "max_samples": "auto"}


def fit_predict(train, test, params):
    model = IsolationForest(random_state=42, n_jobs=-1, **params).fit(train[T.ISO_FEATURE_COLS].fillna(0))
    return (model.predict(test[T.ISO_FEATURE_COLS].fillna(0)) == -1).astype(int)


def prepare(fit_df, others):
    scale = T.fit_min_max(fit_df, T.NORMALIZE_COLS)
    return T.apply_min_max(fit_df, T.NORMALIZE_COLS, scale), [T.apply_min_max(o, T.NORMALIZE_COLS, scale) for o in others], scale


def multi_seed_check(seeds, workdir, independent_path, outfile):
    independent = R.prepare_independent(independent_path)
    rows = []
    for seed in seeds:
        clean = pd.read_csv(R.ensure_clean_data(seed, workdir))
        train_part, test_part = T.time_based_split(clean, 0.2)
        train_norm, (test_norm,), _ = prepare(train_part, [test_part])
        scale = T.fit_min_max(clean, T.NORMALIZE_COLS)
        full = T.apply_min_max(clean, T.NORMALIZE_COLS, scale)
        ind = E.apply_min_max(independent, list(scale.keys()), scale)
        for max_samples in ("auto", 512):
            params = {**DEFAULT, "max_samples": max_samples}
            test_scores = R.scores(test_norm["is_anomaly"], fit_predict(train_norm, test_norm, params))
            ind_scores = R.scores(ind["is_anomaly"], fit_predict(full, ind, params))
            rows.append({"seed": seed, "max_samples": max_samples, "test_f1": test_scores["f1"],
                         "independent_precision": ind_scores["precision"], "independent_recall": ind_scores["recall"],
                         "independent_f1": ind_scores["f1"]})
        print(f"seed {seed} done", flush=True)
    new = pd.DataFrame(rows)
    if os.path.exists(outfile):
        old = pd.read_csv(outfile)
        new = pd.concat([old[~old["seed"].isin(seeds)], new])
    new.sort_values(["max_samples", "seed"]).to_csv(outfile, index=False)
    print(new.groupby("max_samples")[["test_f1", "independent_f1"]].agg(["mean", "min", "max"]).round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description="Tune Isolation Forest hyperparameters on a time-ordered validation split")
    parser.add_argument("--infile", type=str, default=os.path.join(ROOT, "data/processed/metrics_clean.csv"))
    parser.add_argument("--independent", type=str, default=os.path.join(ROOT, "independent_test_set_v2.csv"))
    parser.add_argument("--outfile", type=str, default=os.path.join(ROOT, "data/processed/tuning_results.csv"))
    parser.add_argument("--multiseed", type=int, nargs="+", default=None,
                        help="instead of tuning, compare max_samples=auto and 512 across these training seeds")
    parser.add_argument("--workdir", type=str, default="data/sweep")
    args = parser.parse_args()

    if args.multiseed:
        multi_seed_check(args.multiseed, args.workdir, args.independent,
                         args.outfile.replace(".csv", "_multiseed.csv"))
        return

    clean = pd.read_csv(args.infile)
    train_part, test_part = T.time_based_split(clean, 0.2)
    fit_part, valid_part = T.time_based_split(train_part, 0.25)
    fit_norm, (valid_norm,), _ = prepare(fit_part, [valid_part])

    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        pred = fit_predict(fit_norm, valid_norm, params)
        rows.append({**params, **R.scores(valid_norm["is_anomaly"], pred)})
        print(params, rows[-1]["f1"], flush=True)

    result = pd.DataFrame(rows).rename(columns={"precision": "val_precision", "recall": "val_recall", "f1": "val_f1"})
    best = result.loc[result["val_f1"].idxmax(), list(GRID)].to_dict()
    result["selected"] = (result[list(GRID)] == pd.Series(best)).all(axis=1)

    train_norm, (test_norm,), scale = prepare(train_part, [test_part])
    full_norm = T.apply_min_max(clean, T.NORMALIZE_COLS, T.fit_min_max(clean, T.NORMALIZE_COLS))
    full_scale = T.fit_min_max(clean, T.NORMALIZE_COLS)
    independent = E.apply_min_max(R.prepare_independent(args.independent), list(full_scale.keys()), full_scale)

    final = []
    for name, params in (("default", DEFAULT), ("tuned", best)):
        test_scores = R.scores(test_norm["is_anomaly"], fit_predict(train_norm, test_norm, params))
        indep_scores = R.scores(independent["is_anomaly"], fit_predict(full_norm, independent, params))
        final.append({"config": name, **params,
                      **{f"test_{k}": v for k, v in test_scores.items()},
                      **{f"independent_{k}": v for k, v in indep_scores.items()}})
    final = pd.DataFrame(final)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    result.to_csv(args.outfile, index=False)
    final.to_csv(args.outfile.replace(".csv", "_final.csv"), index=False)
    print(result.sort_values("val_f1", ascending=False).head(6).to_string(index=False))
    print(final.T.to_string())


if __name__ == "__main__":
    main()
