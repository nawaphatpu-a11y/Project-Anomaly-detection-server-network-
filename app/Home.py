"""
Main dashboard page.
Run from the project root: streamlit run app/Home.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from utils import init_buffer, tick

st.set_page_config(page_title="Server & Network Health", page_icon="📊", layout="wide")
init_buffer()

st.sidebar.header("Settings")
refresh_seconds = st.sidebar.slider("⏱️ Refresh every", 2, 15, 4, format="%d s")
rows_per_tick = st.sidebar.slider("📊 New readings per round", 1, 20, 5, format="%d rows")
st.sidebar.caption(f"Generates {rows_per_tick} new rows every {refresh_seconds} seconds")
if st.sidebar.button("Clear all data"):
    st.session_state.readings = pd.DataFrame()
    st.session_state.alerts = pd.DataFrame()
    st.session_state.resp_history = {}

st_autorefresh(interval=refresh_seconds * 1000, key="dashboard_autorefresh")
new_readings = tick(n_new=rows_per_tick)

df = st.session_state.readings

st.title("📊 Server & Network Health Dashboard")
st.caption("Live data is simulated and scored by the Isolation Forest trained in train_model.py")

if df.empty:
    st.info("Waiting for the first round of live data...")
    st.stop()

total_readings = len(df)
total_anomalies = int(df["is_anomaly"].sum())
health_score = 100 * (1 - total_anomalies / total_readings) if total_readings else 100
servers_online = df["server_id"].nunique()

c1, c2, c3 = st.columns(3)
c1.metric("System Health Score", f"{health_score:.1f}%")
c2.metric("Total Anomalies (session)", f"{total_anomalies}")
c3.metric("Servers Online", f"{servers_online}")

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Average CPU / Memory Trend")
    trend = df.groupby(df.index // max(1, rows_per_tick)).agg(
        CPU=("cpu_usage", "mean"),
        Memory=("memory_usage", "mean"),
    ).reset_index()
    fig = px.line(trend, x="index", y=["CPU", "Memory"],
                  labels={"value": "Percent", "index": "Data Round (Oldest → Newest)", "variable": "Metric"})
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Anomaly Count Trend")
    anomaly_trend = df.groupby(df.index // max(1, rows_per_tick))["is_anomaly"].sum().reset_index()
    fig = px.line(anomaly_trend, x="index", y="is_anomaly",
                  labels={"index": "Data Round (Oldest → Newest)", "is_anomaly": "Anomaly Count"})
    st.plotly_chart(fig, use_container_width=True)

col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Anomalies by Server")
    by_server = df[df["is_anomaly"] == 1]["server_id"].value_counts().reset_index()
    by_server.columns = ["server_id", "count"]
    if not by_server.empty:
        fig = px.bar(by_server, x="server_id", y="count", text="count")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomalies in this session yet")

with col_d:
    st.subheader("Anomalies by Type")
    anomalies = df[df["is_anomaly"] == 1]
    known = anomalies[anomalies["alert_type"] != "Other"]
    other_count = int((anomalies["alert_type"] == "Other").sum())
    by_type = known["alert_type"].value_counts().reset_index()
    by_type.columns = ["alert_type", "count"]
    if not by_type.empty:
        fig = px.bar(by_type, x="alert_type", y="count", text="count")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomalies with an identifiable type in this session yet")
    st.caption(f"Excludes \"Other\" (flagged without a clear dominant signal): {other_count}")

col_e, col_f = st.columns(2)
with col_e:
    st.subheader("CPU vs Response Time (anomalies highlighted)")
    fig = px.scatter(df, x="cpu_usage", y="response_time",
                      color=df["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
                      color_discrete_map={"Normal": "#4C78A8", "Anomaly": "#E45756"})
    st.plotly_chart(fig, use_container_width=True)

with col_f:
    st.subheader("Network Traffic Distribution")
    fig = px.box(df, y="network_in", points="outliers")
    st.plotly_chart(fig, use_container_width=True)

if "role" in df.columns:
    col_g, col_h = st.columns(2)
    with col_g:
        st.subheader("CPU Distribution by Role")
        st.plotly_chart(px.box(df, x="role", y="cpu_usage", color="role"), use_container_width=True)
    with col_h:
        st.subheader("Memory Distribution by Role")
        st.plotly_chart(px.box(df, x="role", y="memory_usage", color="role"), use_container_width=True)

    role_summary = (
        df.groupby("role")
        .agg(readings=("is_anomaly", "size"), alerts=("is_anomaly", "sum"),
             median_cpu=("cpu_usage", "median"), median_memory=("memory_usage", "median"),
             median_response=("response_time", "median"))
        .round(1)
    )
    role_summary["alert_rate_%"] = (role_summary["alerts"] / role_summary["readings"] * 100).round(1)
    st.subheader("Summary by Role")
    st.dataframe(role_summary, use_container_width=True)

st.subheader("Latest 10 Anomalies")
recent_alerts = df[df["is_anomaly"] == 1].sort_values("timestamp", ascending=False).head(10)
if not recent_alerts.empty:
    st.dataframe(
        recent_alerts[["timestamp", "server_id", "role", "alert_type", "probable_cause", "top_signals",
                        "suggested_check", "anomaly_score"]],
        use_container_width=True, hide_index=True,
    )
else:
    st.success("No anomalies in this session yet ✅")
