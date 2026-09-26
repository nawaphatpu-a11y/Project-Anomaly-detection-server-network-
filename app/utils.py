"""
app/utils.py
ฟังก์ชันใช้ร่วมกันระหว่าง Home.py และ pages/1_Alerts.py

แนวคิด: โหลดโมเดล Isolation Forest ที่เทรน+เซฟไว้แล้วจาก train_model.py มาสคอร์ข้อมูล
"live" (จำลอง) แบบ real-time โดยคำนวณ feature ให้ตรงกับตอนเทรนเป๊ะ (normalize ด้วย
min/max ชุดเดียวกับตอนเทรน จาก scale_params.json, คำนวณ trend feature จาก buffer
ย้อนหลังของแต่ละ server ที่เก็บไว้ใน session_state)

ใช้เฉพาะ Isolation Forest เพราะเป็นโมเดลหลักของ prototype ตามแผน (LOF/Z-score ใน
train_model.py มีไว้ทำ benchmark เทียบผลแบบออฟไลน์เท่านั้น ไม่ได้ตั้งใจเอาไป deploy)
"""
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # app/utils.py -> root ของโปรเจกต์
MODEL_PATH = "models/isolation_forest.joblib"
SCALE_PARAMS_PATH = "data/processed/scale_params.json"
BUFFER_MAX_ROWS = 3000          # กันไม่ให้ session state โตไม่หยุดระหว่างเดโมยาวๆ
ANOMALY_INJECT_PROB = 0.04      # โอกาสเฉลี่ยที่ live reading รอบนี้จะเป็น anomaly (จำลองเพื่อโชว์เดโม)

# ให้แต่ละเครื่องมีนิสัยไม่เท่ากัน (บางเครื่อง "ป่วยง่าย" กว่าเครื่องอื่น) ไม่งั้นกราฟ
# "Anomaly แยกตามเครื่อง" จะไม่มีความหมายอะไรเลย เพราะทุกเครื่องสุ่มด้วยโอกาสเท่ากันหมด
# ตัวเลขที่เห็นต่างกันเป็นแค่ noise ล้วนๆ ไม่ได้สะท้อนว่าเครื่องไหนแย่จริง
def _server_weight(server_id: str, server_ids: list) -> float:
    idx = server_ids.index(server_id) if server_id in server_ids else 0
    # กระจายตัวคูณระหว่าง 0.4x ถึง 2.2x แบบ deterministic ตามลำดับเครื่อง
    return 0.4 + (idx % 6) * 0.36


def _ensure_pipeline_has_run():
    # สำคัญสำหรับตอน deploy จริง: Streamlit Community Cloud แค่ clone repo แล้วรัน
    # `streamlit run app/Home.py` ทันที "ไม่" รัน generate_data.py/etl.py/train_model.py
    # ให้เองก่อน ถ้าไม่มีไฟล์โมเดลอยู่ใน repo เว็บจะพังตั้งแต่เปิดครั้งแรก
    # จึงให้ dashboard เช็คเองตอนโหลดครั้งแรก แล้วรัน pipeline offline ให้อัตโนมัติถ้ายังไม่มี
    # (รันแค่ครั้งเดียวตอนยังไม่มีไฟล์ ครั้งต่อไปเร็วปกติเพราะมีไฟล์แล้ว)
    model_path = PROJECT_ROOT / MODEL_PATH
    scale_path = PROJECT_ROOT / SCALE_PARAMS_PATH
    if model_path.exists() and scale_path.exists():
        return

    with st.spinner("ยังไม่มีโมเดล — กำลังรัน pipeline ครั้งแรก (generate_data → etl → train_model)..."):
        for script in ["generate_data.py", "etl.py", "train_model.py"]:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / script)],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            )
            if result.returncode != 0:
                st.error(f"รัน {script} ไม่สำเร็จ:\n{result.stderr[-2000:]}")
                st.stop()


@st.cache_resource
def load_model_bundle():
    """โหลด {model, feature_cols, trend_window, server_ids} ที่ train_model.py เซฟไว้"""
    _ensure_pipeline_has_run()
    try:
        return joblib.load(PROJECT_ROOT / MODEL_PATH), None
    except FileNotFoundError:
        return None, f"ไม่พบไฟล์โมเดลที่ {MODEL_PATH} แม้รัน pipeline อัตโนมัติแล้ว ลองรันเองด้วยมือดูอีกครั้ง"


@st.cache_resource
def load_scale_params():
    with open(PROJECT_ROOT / SCALE_PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _minmax(value: float, col: str, scale_params: dict) -> float:
    # ต้องเหมือนกับ train_model.py::apply_min_max เป๊ะ (รวม clip) ไม่งั้นจะเกิด
    # training-serving skew: ตอนเทรน/ประเมินผล ค่า norm ถูก clip ไว้ที่ 0-1 เสมอ แต่ log
    # ตอน live เดิมไม่ได้ clip ทำให้ reading ที่แรงกว่าที่โมเดลเคยเห็นตอนเทรน (เช่น network
    # พุ่งแรงมากในโหมดจำลอง) กลายเป็นค่า norm ติดลบ/เกิน 1 ซึ่งโมเดลไม่เคยเห็นรูปแบบนี้มา
    # ก่อน อาจทำให้ตัดสินใจผิดเพี้ยนไปจากที่ควรจะเป็น -- เจอจากการตรวจสอบ evaluate_independent.py
    # ที่ scale ของข้อมูลอิสระต่างจากตอนเทรนเยอะ แล้วพบว่า precision ร่วงหนักเพราะเหตุผลเดียวกันนี้
    lo, hi = scale_params[col]["min"], scale_params[col]["max"]
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _simulate_raw_reading(server_id: str, anomaly_prob: float) -> dict:
    """สุ่มค่า metric หนึ่งแถว จำลองการอ่านค่าจริงจาก server ณ เวลาปัจจุบัน
    (สุ่มช่วงใกล้เคียงกับข้อมูลที่ generate_data.py ใช้เทรนโมเดล)"""
    cpu = np.random.normal(30, 6)
    mem = np.random.normal(40, 8)
    net_in = abs(np.random.normal(80, 25))
    net_out = abs(np.random.normal(70, 20))
    resp = abs(np.random.normal(90, 20))

    injected_type = None
    if random.random() < anomaly_prob:
        injected_type = random.choice(["CPU_SPIKE", "NETWORK_SURGE", "RESPONSE_DEGRADATION"])

    if injected_type == "CPU_SPIKE":
        cpu = np.random.uniform(92, 100)
        mem += np.random.uniform(10, 20)
    elif injected_type == "NETWORK_SURGE":
        net_in *= np.random.uniform(5, 10)
        net_out *= np.random.uniform(5, 10)
    elif injected_type == "RESPONSE_DEGRADATION":
        resp *= np.random.uniform(2.5, 4.5)  # ค่าเดี่ยวพุ่งแรงพอให้เห็นผลใน 1 tick ของเดโม

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


def _build_feature_row(reading: dict, scale_params: dict, feature_cols: list,
                        resp_history: deque) -> pd.DataFrame:
    """คำนวณ feature ให้ตรงกับตอนเทรนเป๊ะ (etl.py) แล้วจัดเรียงคอลัมน์ตาม feature_cols"""
    network_total = reading["network_in"] + reading["network_out"]
    cpu_memory_ratio = reading["cpu_usage"] / reading["memory_usage"] if reading["memory_usage"] else 0.0

    hour = reading["timestamp"].hour
    day_of_week = reading["timestamp"].weekday()
    is_peak_hour = int(9 <= hour <= 17)

    # trend feature: เทียบกับค่าเฉลี่ย response_time ย้อนหลังของ "เครื่องนี้" (ก่อนรอบนี้)
    # -- ตรงกับ etl.py::add_trend_feature (shift(1).rolling(window).mean())
    if len(resp_history) >= 3:
        roll_mean = float(np.mean(resp_history))
    else:
        roll_mean = reading["response_time"]  # ยังไม่มีประวัติพอ -> ให้ dev = 0 ไปก่อน
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
        "response_time_dev_from_roll": response_time_dev_from_roll,  # ดิบ ไม่ normalize (ตรงกับ ISO_FEATURE_COLS)
    }
    return pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)


def init_buffer():
    if "readings" not in st.session_state:
        st.session_state.readings = pd.DataFrame()
    if "alerts" not in st.session_state:
        st.session_state.alerts = pd.DataFrame()
    if "resp_history" not in st.session_state:
        st.session_state.resp_history = {}  # server_id -> deque ของ response_time ย้อนหลัง


def tick(n_new: int = 5):
    """เรียกทุกครั้งที่หน้าเว็บ autorefresh: จำลอง reading ใหม่ n_new แถว สคอร์ แล้วอัปเดต buffer"""
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

    new_rows = []
    for _ in range(n_new):
        server_id = random.choice(server_ids)
        prob = ANOMALY_INJECT_PROB * _server_weight(server_id, server_ids)
        reading = _simulate_raw_reading(server_id, prob)

        hist = st.session_state.resp_history.setdefault(server_id, deque(maxlen=window))
        X = _build_feature_row(reading, scale_params, feature_cols, hist)

        pred = model.predict(X)[0]              # -1 = anomaly, 1 = normal
        score = model.decision_function(X)[0]    # ยิ่งต่ำ ยิ่งผิดปกติ

        reading["is_anomaly"] = int(pred == -1)
        reading["anomaly_score"] = round(float(score), 4)
        reading["alert_type"] = reading["_injected_type"] or ("Other" if reading["is_anomaly"] else None)
        reading.pop("_injected_type", None)

        hist.append(reading["response_time"])  # อัปเดตประวัติ "หลัง" สคอร์รอบนี้แล้ว (กันเห็นค่าตัวเอง)
        new_rows.append(reading)

    new_df = pd.DataFrame(new_rows)
    st.session_state.readings = pd.concat([st.session_state.readings, new_df], ignore_index=True).tail(BUFFER_MAX_ROWS)

    new_alerts = new_df[new_df["is_anomaly"] == 1]
    if not new_alerts.empty:
        st.session_state.alerts = pd.concat([st.session_state.alerts, new_alerts], ignore_index=True).tail(BUFFER_MAX_ROWS)

    return new_df
