"""Shared helpers for the dashboard: model loading, live reading simulation and feature building."""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from diagnose import diagnose
from etl import SEVERITY_MAP, ROLE_METRICS
MODEL_PATH = "models/isolation_forest.joblib"
SCALE_PARAMS_PATH = "data/processed/scale_params.json"
BUFFER_MAX_ROWS = 3000
ANOMALY_INJECT_PROB = 0.04

def _server_weight(server_id: str, server_ids: list) -> float:
    idx = server_ids.index(server_id) if server_id in server_ids else 0
    return 0.4 + (idx % 6) * 0.36


def _ensure_pipeline_has_run():
    model_path = PROJECT_ROOT / MODEL_PATH
    scale_path = PROJECT_ROOT / SCALE_PARAMS_PATH
    if model_path.exists() and scale_path.exists():
        return

    with st.spinner("No model found. Running the pipeline for the first time (generate_data → etl → train_model)..."):
        for script in ["generate_data.py", "etl.py", "train_model.py"]:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / script)],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            )
            if result.returncode != 0:
                st.error(f"{script} failed:\n{result.stderr[-2000:]}")
                st.stop()


def ensure_evaluation_has_run():
    _ensure_pipeline_has_run()

    eval_path = PROJECT_ROOT / "data/processed/evaluation_report.csv"
    indep_report = PROJECT_ROOT / "data/processed/evaluation_report_independent.csv"
    indep_csv = PROJECT_ROOT / "independent_test_set_v2.csv"

    todo = []
    if not eval_path.exists():
        todo.append("evaluate.py")
    if not indep_csv.exists():
        todo.append("Generate_data_AI.py")
    if not indep_report.exists():
        todo.append(["evaluate_independent.py", "--infile", "independent_test_set_v2.csv"])

    if not todo:
        return

    with st.spinner("Running evaluate.py for the first time (needed by the Model Evaluation page)..."):
        for item in todo:
            cmd = [sys.executable] + (item if isinstance(item, list) else [item])
            result = subprocess.run(
                [cmd[0], str(PROJECT_ROOT / cmd[1])] + cmd[2:],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            )
            if result.returncode != 0:
                st.warning(f"{cmd[1]} failed (this page will be incomplete):\n{result.stderr[-1000:]}")


@st.cache_resource
def load_model_bundle():
    """Load the bundle saved by train_model.py: model, feature_cols, trend_window, server_ids."""
    _ensure_pipeline_has_run()
    try:
        return joblib.load(PROJECT_ROOT / MODEL_PATH), None
    except FileNotFoundError:
        return None, f"Model file not found at {MODEL_PATH} even after running the pipeline automatically. Try running it manually"


@st.cache_resource
def load_scale_params():
    with open(PROJECT_ROOT / SCALE_PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _minmax(value: float, col: str, scale_params: dict) -> float:
    lo, hi = scale_params[col]["min"], scale_params[col]["max"]
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _role_normal_params(role: str, baselines: dict | None) -> dict:
    """Normal (loc, scale) per metric for a role, taken from the role_baselines learned at training time.
    Falls back to fixed defaults when the model bundle has no baselines (older models)."""
    default = {"cpu_usage": (30, 6), "memory_usage": (40, 8), "network_in": (80, 25),
               "network_out": (70, 20), "response_time": (90, 20)}
    if not baselines or role not in baselines:
        return default
    return {m: (baselines[role][m]["median"], max(baselines[role][m]["mad"] * 1.4826, 1e-6))
            for m in ROLE_METRICS}


def _simulate_raw_reading(server_id: str, anomaly_prob: float, role: str = "Unknown",
                          baselines: dict | None = None) -> dict:
    """Simulate one live reading for a server, centered on that server's role baseline."""
    p = _role_normal_params(role, baselines)
    cpu = np.random.normal(*p["cpu_usage"])
    mem = np.random.normal(*p["memory_usage"])
    net_in = abs(np.random.normal(*p["network_in"]))
    net_out = abs(np.random.normal(*p["network_out"]))
    resp = abs(np.random.normal(*p["response_time"]))

    injected_type = None
    if random.random() < anomaly_prob:
        injected_type = random.choice(["CPU_SPIKE", "NETWORK_SURGE", "RESPONSE_DEGRADATION"])

    if injected_type == "CPU_SPIKE":
        cpu = np.random.uniform(92, 100)
        mem = min(mem + np.random.uniform(10, 20), 99.0)
    elif injected_type == "NETWORK_SURGE":
        net_in *= np.random.uniform(5, 10)
        net_out *= np.random.uniform(5, 10)
    elif injected_type == "RESPONSE_DEGRADATION":
        resp *= np.random.uniform(2.5, 4.5)

    return {
        "timestamp": datetime.now(),
        "server_id": server_id,
        "cpu_usage": round(float(np.clip(cpu, 0, 100)), 2),
        "memory_usage": round(float(np.clip(mem, 0, 100)), 2),
        "network_in": round(float(net_in), 2),
        "network_out": round(float(net_out), 2),
        "response_time": round(float(resp), 2),
        "_injected_type": injected_type,
    }


def _role_z_features(reading: dict, baselines: dict | None) -> dict:
    if not baselines:
        return {}
    b = baselines.get(reading.get("role"), baselines["__global__"])
    return {f"{m}_role_z": (reading[m] - b[m]["median"]) / b[m]["mad"] for m in ROLE_METRICS}


def _build_feature_row(reading: dict, scale_params: dict, feature_cols: list,
                        resp_history: deque, baselines: dict | None = None) -> pd.DataFrame:
    """Build features exactly as etl.py does at training time, ordered by feature_cols."""
    network_total = reading["network_in"] + reading["network_out"]
    cpu_memory_ratio = reading["cpu_usage"] / reading["memory_usage"] if reading["memory_usage"] else 0.0

    hour = reading["timestamp"].hour
    day_of_week = reading["timestamp"].weekday()
    is_peak_hour = int(9 <= hour <= 17)

    if len(resp_history) >= 3:
        roll_mean = float(np.mean(resp_history))
    else:
        roll_mean = reading["response_time"]
    response_time_dev_from_roll = reading["response_time"] - roll_mean

    values = {
        "cpu_usage_norm": _minmax(reading["cpu_usage"], "cpu_usage", scale_params),
        "memory_usage_norm": _minmax(reading["memory_usage"], "memory_usage", scale_params),
        "network_in_norm": _minmax(reading["network_in"], "network_in", scale_params),
        "network_out_norm": _minmax(reading["network_out"], "network_out", scale_params),
        "response_time_norm": _minmax(reading["response_time"], "response_time", scale_params),
        "network_total_norm": _minmax(network_total, "network_total", scale_params),
        "cpu_memory_ratio_norm": _minmax(cpu_memory_ratio, "cpu_memory_ratio", scale_params),
        "hour": hour,
        "day_of_week": day_of_week,
        "is_peak_hour": is_peak_hour,
        "response_time_dev_from_roll": response_time_dev_from_roll,
    }
    values.update(_role_z_features(reading, baselines))
    return pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)


def init_buffer():
    if "readings" not in st.session_state:
        st.session_state.readings = pd.DataFrame()
    if "alerts" not in st.session_state:
        st.session_state.alerts = pd.DataFrame()
    if "resp_history" not in st.session_state:
        st.session_state.resp_history = {}


def tick(n_new: int = 5):
    """Called on every autorefresh: simulate n_new readings, score them and update the buffer."""
    init_buffer()
    bundle, err = load_model_bundle()
    if err:
        st.error(err)
        st.stop()
    scale_params = load_scale_params()

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    window = bundle["trend_window"]
    server_ids = bundle["server_ids"]
    server_roles = bundle.get("server_roles", {})
    role_baselines = bundle.get("role_baselines")

    new_rows = []
    for _ in range(n_new):
        server_id = random.choice(server_ids)
        prob = ANOMALY_INJECT_PROB * _server_weight(server_id, server_ids)
        role = server_roles.get(server_id, "Unknown")
        reading = _simulate_raw_reading(server_id, prob, role, role_baselines)
        reading["role"] = role

        hist = st.session_state.resp_history.setdefault(server_id, deque(maxlen=window))
        X = _build_feature_row(reading, scale_params, feature_cols, hist, role_baselines)

        pred = model.predict(X)[0]
        score = model.decision_function(X)[0]

        reading["is_anomaly"] = int(pred == -1)
        reading["anomaly_score"] = round(float(score), 4)
        if reading["is_anomaly"]:
            features = {**X.iloc[0].to_dict(), **_role_z_features(reading, role_baselines)}
            diagnosis = diagnose(features, role=role)
            reading["alert_type"] = "Other" if diagnosis["cause_type"] == "UNCLASSIFIED" else diagnosis["cause_type"]
            reading["probable_cause"] = diagnosis["probable_cause"]
            reading["top_signals"] = diagnosis["top_signals"]
            reading["suggested_check"] = diagnosis["suggested_check"]
            reading["severity_level"] = SEVERITY_MAP.get(reading["alert_type"], "None")
        else:
            reading["alert_type"] = None
            reading["probable_cause"] = None
            reading["top_signals"] = None
            reading["suggested_check"] = None
            reading["severity_level"] = "None"
        reading.pop("_injected_type", None)

        hist.append(reading["response_time"])
        new_rows.append(reading)

    new_df = pd.DataFrame(new_rows)
    st.session_state.readings = pd.concat([st.session_state.readings, new_df], ignore_index=True).tail(BUFFER_MAX_ROWS)

    new_alerts = new_df[new_df["is_anomaly"] == 1]
    if not new_alerts.empty:
        st.session_state.alerts = pd.concat([st.session_state.alerts, new_alerts], ignore_index=True).tail(BUFFER_MAX_ROWS)

    return new_df
