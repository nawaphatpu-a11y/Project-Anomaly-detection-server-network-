"""Alerts page."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
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
    st.success("No alerts yet. Everything looks normal ✅")
    st.stop()

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    type_filter = st.multiselect("Filter by type", options=sorted(alerts["alert_type"].dropna().unique()))
with col2:
    server_filter = st.multiselect("Filter by server", options=sorted(alerts["server_id"].unique()))
with col3:
    severity_filter = st.multiselect("Filter by severity", options=["High", "Medium", "Low"])
with col4:
    sort_mode = st.selectbox("Sort by", ["Highest severity first", "Most recent first"])

filtered = alerts.copy()
if type_filter:
    filtered = filtered[filtered["alert_type"].isin(type_filter)]
if server_filter:
    filtered = filtered[filtered["server_id"].isin(server_filter)]
if severity_filter:
    filtered = filtered[filtered["severity_level"].isin(severity_filter)]

if sort_mode == "Highest severity first":
    filtered = filtered.sort_values("anomaly_score", ascending=True)
else:
    filtered = filtered.sort_values("timestamp", ascending=False)

st.caption(f"Last updated: {alerts['timestamp'].max()}")

col_m, col_dl = st.columns([3, 1])
with col_m:
    st.metric("Total alerts (session)", len(alerts))
with col_dl:
    csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name=f"alerts_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.dataframe(
    filtered[["timestamp", "server_id", "alert_type", "severity_level", "cpu_usage", "memory_usage",
              "network_in", "network_out", "response_time", "anomaly_score"]],
    use_container_width=True,
    hide_index=True,
)
