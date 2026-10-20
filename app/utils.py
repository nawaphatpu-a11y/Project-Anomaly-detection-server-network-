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
from streamlit_autorefresh import st_autorefresh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from detection import SEVERITY_RANK, detection_source, group_incidents, zscore_flag
from diagnose import cause_title, decode_signals, diagnose, format_signals, suggested_check
from etl import SEVERITY_MAP, ROLE_METRICS
from i18n import get_lang, t
from replay import ReplayFeed
MODEL_PATH = "models/isolation_forest.joblib"
SCALE_PARAMS_PATH = "data/processed/scale_params.json"
BUFFER_MAX_ROWS = 3000
HISTORY_PER_SERVER = 400
REPLAY_PATH = "data/processed/replay_sample.csv"
ANOMALY_INJECT_PROB = 0.04

def _server_weight(server_id: str, server_ids: list) -> float:
    idx = server_ids.index(server_id) if server_id in server_ids else 0
    return 0.4 + (idx % 6) * 0.36


def _ensure_pipeline_has_run():
    model_path = PROJECT_ROOT / MODEL_PATH
    scale_path = PROJECT_ROOT / SCALE_PARAMS_PATH
    if model_path.exists() and scale_path.exists():
        return

    with st.spinner(t("msg.pipeline")):
        for script in ["generate_data.py", "etl.py", "train_model.py"]:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / script)],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            )
            if result.returncode != 0:
                st.error(t("msg.script_failed", script=script, detail=result.stderr[-2000:]))
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

    with st.spinner(t("msg.evaluate")):
        for item in todo:
            cmd = [sys.executable] + (item if isinstance(item, list) else [item])
            result = subprocess.run(
                [cmd[0], str(PROJECT_ROOT / cmd[1])] + cmd[2:],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            )
            if result.returncode != 0:
                st.warning(t("msg.evaluate_failed", script=cmd[1], detail=result.stderr[-1000:]))


@st.cache_resource
def load_model_bundle():
    """Load the bundle saved by train_model.py: model, feature_cols, trend_window, server_ids."""
    _ensure_pipeline_has_run()
    try:
        return joblib.load(PROJECT_ROOT / MODEL_PATH), None
    except FileNotFoundError:
        return None, t("msg.model_missing", path=MODEL_PATH)


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


def _feature_values(reading: dict, scale_params: dict, resp_history: deque, baselines: dict | None = None) -> dict:
    network_total = reading["network_in"] + reading["network_out"]
    cpu_memory_ratio = reading["cpu_usage"] / reading["memory_usage"] if reading["memory_usage"] else 0.0

    clock = reading.get("source_timestamp", reading["timestamp"])
    hour = clock.hour
    day_of_week = clock.weekday()
    is_peak_hour = int(9 <= hour <= 17)

    if len(resp_history) >= 3:
        roll_mean = float(np.mean(resp_history))
    else:
        roll_mean = reading["response_time"]

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
        "response_time_dev_from_roll": reading["response_time"] - roll_mean,
    }
    values.update(_role_z_features(reading, baselines))
    return values


def init_buffer():
    if "readings" not in st.session_state:
        st.session_state.readings = pd.DataFrame()
    if "alerts" not in st.session_state:
        st.session_state.alerts = pd.DataFrame()
    if "resp_history" not in st.session_state:
        st.session_state.resp_history = {}
    for counter in ("total_readings", "total_anomalies", "round", "server_cursor"):
        st.session_state.setdefault(counter, 0)
    st.session_state.setdefault("replay_cursors", {})
    st.session_state.setdefault("reading_counts", {})
    st.session_state.setdefault("feedback", {})


@st.cache_resource
def load_replay_feed():
    path = PROJECT_ROOT / REPLAY_PATH
    return ReplayFeed(pd.read_csv(path)) if path.exists() else None


def _replay_reading(feed: ReplayFeed, server_id: str, cursors: dict):
    record, wrapped = feed.next_reading(server_id, cursors)
    injected = record["anomaly_type"] if record["is_anomaly"] else None
    reading = {
        "timestamp": datetime.now(),
        "server_id": server_id,
        "cpu_usage": float(record["cpu_usage"]),
        "memory_usage": float(record["memory_usage"]),
        "network_in": float(record["network_in"]),
        "network_out": float(record["network_out"]),
        "response_time": float(record["response_time"]),
        "_injected_type": injected,
        "source_timestamp": pd.Timestamp(record["timestamp"]),
    }
    return reading, wrapped


def tick(n_new: int = 5):
    """Called on every autorefresh: take n_new readings, score them in one batch and update the buffer."""
    init_buffer()
    bundle, err = load_model_bundle()
    if err:
        st.error(err)
        st.stop()
    scale_params = load_scale_params()
    feed = load_replay_feed()

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    window = bundle["trend_window"]
    server_ids = bundle["server_ids"]
    server_roles = bundle.get("server_roles", {})
    role_baselines = bundle.get("role_baselines")

    st.session_state["round"] += 1
    readings, feature_values = [], []
    for _ in range(n_new):
        server_id = server_ids[st.session_state["server_cursor"] % len(server_ids)]
        st.session_state["server_cursor"] += 1
        role = server_roles.get(server_id, "Unknown")
        hist = st.session_state.resp_history.setdefault(server_id, deque(maxlen=window))

        if feed is not None and feed.has(server_id):
            reading, wrapped = _replay_reading(feed, server_id, st.session_state["replay_cursors"])
            if wrapped:
                hist.clear()
        else:
            prob = ANOMALY_INJECT_PROB * _server_weight(server_id, server_ids)
            reading = _simulate_raw_reading(server_id, prob, role, role_baselines)
        reading["role"] = role
        counts = st.session_state["reading_counts"]
        counts[server_id] = counts.get(server_id, 0) + 1
        reading["reading_no"] = counts[server_id]

        feature_values.append(_feature_values(reading, scale_params, hist, role_baselines))
        hist.append(reading["response_time"])
        readings.append(reading)

    X = pd.DataFrame([[values[c] for c in feature_cols] for values in feature_values], columns=feature_cols)
    scores = model.decision_function(X)
    zstats = bundle.get("zscore_stats")

    for reading, values, score in zip(readings, feature_values, scores):
        source = detection_source(bool(score < 0), bool(zstats) and zscore_flag(values, zstats))
        reading["detection"] = source
        reading["is_anomaly"] = int(source is not None)
        reading["anomaly_score"] = round(float(score), 4)
        reading["round"] = st.session_state["round"]
        if reading["is_anomaly"]:
            diagnosis = diagnose(values, role=reading["role"])
            reading["alert_type"] = "Other" if diagnosis["cause_type"] == "UNCLASSIFIED" else diagnosis["cause_type"]
            reading["signals_raw"] = diagnosis["signals_raw"]
            reading["severity_level"] = SEVERITY_MAP.get(reading["alert_type"], "None")
        else:
            reading["alert_type"] = None
            reading["signals_raw"] = None
            reading["severity_level"] = "None"
        reading.pop("_injected_type", None)
        reading.pop("source_timestamp", None)

    new_df = pd.DataFrame(readings)
    st.session_state["total_readings"] += len(new_df)
    st.session_state["total_anomalies"] += int(new_df["is_anomaly"].sum())

    combined = pd.concat([st.session_state.readings, new_df], ignore_index=True)
    st.session_state.readings = combined.groupby("server_id", group_keys=False).tail(HISTORY_PER_SERVER)

    new_alerts = new_df[new_df["is_anomaly"] == 1]
    if not new_alerts.empty:
        st.session_state.alerts = pd.concat([st.session_state.alerts, new_alerts], ignore_index=True).tail(BUFFER_MAX_ROWS)

    return new_df


def localize_alerts(df, lang=None):
    lang = lang or get_lang()
    out = df.copy()
    for column in ("probable_cause", "top_signals", "suggested_check"):
        out[column] = None
    flagged = out["is_anomaly"] == 1
    if not flagged.any():
        return out

    subset = out.loc[flagged]
    causes = subset["alert_type"].where(subset["alert_type"] != "Other", "UNCLASSIFIED")
    roles = subset["role"].fillna("") if "role" in subset else pd.Series("", index=subset.index)
    raw_signals = subset["signals_raw"] if "signals_raw" in subset else pd.Series(None, index=subset.index)

    titles = {c: cause_title(c, lang) for c in causes.unique()}
    signal_text = {raw: format_signals(decode_signals(raw), lang) for raw in raw_signals.dropna().unique()}
    empty_signals = format_signals([], lang)
    checks = {(c, r): suggested_check(c, r, lang) for c, r in set(zip(causes, roles))}

    out.loc[flagged, "probable_cause"] = causes.map(titles)
    out.loc[flagged, "top_signals"] = raw_signals.map(signal_text).fillna(empty_signals)
    out.loc[flagged, "suggested_check"] = [checks[(c, r)] for c, r in zip(causes, roles)]
    return out


DEFAULT_REFRESH_SECONDS = 4
DEFAULT_ROWS_PER_TICK = 5


def live_settings():
    refresh = st.sidebar.slider(
        t("sidebar.refresh"), 2, 15, st.session_state.get("refresh_seconds", DEFAULT_REFRESH_SECONDS),
        format=f"%d {t('unit.seconds')}")
    rows = st.sidebar.slider(
        t("sidebar.rows"), 1, 20, st.session_state.get("rows_per_tick", DEFAULT_ROWS_PER_TICK),
        format=f"%d {t('unit.rows')}")
    paused = st.sidebar.checkbox(t("sidebar.pause"), value=st.session_state.get("live_paused", False),
                                 help=t("sidebar.pause_help"))
    st.session_state["refresh_seconds"] = refresh
    st.session_state["rows_per_tick"] = rows
    st.session_state["live_paused"] = paused
    st.sidebar.caption(t("sidebar.caption", rows=rows, seconds=refresh))
    if st.sidebar.button(t("sidebar.clear")):
        st.session_state.readings = pd.DataFrame()
        st.session_state.alerts = pd.DataFrame()
        st.session_state.resp_history = {}
        st.session_state["total_readings"] = 0
        st.session_state["total_anomalies"] = 0
        for stale in ("alert_pool", "alert_pool_signature", "alert_pool_time", "saved_alert_choice"):
            st.session_state.pop(stale, None)
    return refresh, rows, paused


def live_update(refresh, rows, paused, key):
    if paused:
        st.sidebar.info(t("sidebar.paused_note"))
        return None
    st_autorefresh(interval=refresh * 1000, key=key)
    return tick(n_new=rows)


def show_wrapped_table(df):
    st.table(df.reset_index(drop=True).style.hide(axis="index"))


def current_incidents():
    return group_incidents(st.session_state.alerts, st.session_state["reading_counts"])


def record_feedback(incident_id, verdict, detection):
    st.session_state["feedback"][incident_id] = {"verdict": verdict, "detection": detection}
