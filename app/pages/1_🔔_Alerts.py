"""
app/pages/1_🔔_Alerts.py
หน้าแจ้งเตือนแยกต่างหาก — Streamlit จะแสดงเป็นเมนูแยกใน sidebar โดยอัตโนมัติ
(ตามที่วางแผนไว้ตอนแรก: "หน้าต่างแจ้งเตือนแยกต่างหาก" คือหน้านี้)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from utils import init_buffer, tick

st.set_page_config(page_title="Alerts", page_icon="🔔", layout="wide")
init_buffer()

st_autorefresh(interval=4000, key="alerts_autorefresh")
tick(n_new=3)

st.title("🔔 Alert Log")

alerts = st.session_state.alerts

if alerts.empty:
    st.success("ยังไม่มีการแจ้งเตือน ระบบปกติดี ✅")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    type_filter = st.multiselect("กรองตามประเภท", options=sorted(alerts["alert_type"].dropna().unique()))
with col2:
    server_filter = st.multiselect("กรองตามเครื่อง", options=sorted(alerts["server_id"].unique()))

filtered = alerts.copy()
if type_filter:
    filtered = filtered[filtered["alert_type"].isin(type_filter)]
if server_filter:
    filtered = filtered[filtered["server_id"].isin(server_filter)]

filtered = filtered.sort_values("timestamp", ascending=False)

st.caption(f"อัปเดตล่าสุด: {alerts['timestamp'].max()}")
st.metric("จำนวนการแจ้งเตือนทั้งหมด (session)", len(alerts))

st.dataframe(
    filtered[["timestamp", "server_id", "alert_type", "cpu_usage", "memory_usage",
              "network_in", "network_out", "response_time", "anomaly_score"]],
    use_container_width=True,
    hide_index=True,
)
