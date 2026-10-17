"""Main dashboard page.
Run from the project root: streamlit run app/Home.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from i18n import column_labels, language_selector, t, type_label
from utils import init_buffer, live_settings, localize_alerts, tick

st.set_page_config(page_title="Server & Network Health", page_icon="📊", layout="wide")
init_buffer()
language_selector()

st.sidebar.header(t("sidebar.settings"))
refresh_seconds, rows_per_tick = live_settings()

st_autorefresh(interval=refresh_seconds * 1000, key="dashboard_autorefresh")
new_readings = tick(n_new=rows_per_tick)

df = st.session_state.readings

st.title(t("home.title"))
st.caption(t("home.caption"))

if df.empty:
    st.info(t("home.waiting"))
    st.stop()

total_readings = len(df)
total_anomalies = int(df["is_anomaly"].sum())
health_score = 100 * (1 - total_anomalies / total_readings) if total_readings else 100
servers_online = df["server_id"].nunique()

c1, c2, c3 = st.columns(3)
c1.metric(t("metric.health"), f"{health_score:.1f}%")
c2.metric(t("metric.anomalies"), f"{total_anomalies}")
c3.metric(t("metric.servers"), f"{servers_online}")

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader(t("chart.trend"))
    trend = df.groupby(df.index // max(1, rows_per_tick)).agg(
        CPU=("cpu_usage", "mean"),
        Memory=("memory_usage", "mean"),
    ).reset_index()
    fig = px.line(trend, x="index", y=["CPU", "Memory"],
                  labels={"value": t("label.percent"), "index": t("label.round"), "variable": t("label.metric")})
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader(t("chart.anomaly_trend"))
    anomaly_trend = df.groupby(df.index // max(1, rows_per_tick))["is_anomaly"].sum().reset_index()
    fig = px.line(anomaly_trend, x="index", y="is_anomaly",
                  labels={"index": t("label.round"), "is_anomaly": t("label.anomaly_count")})
    st.plotly_chart(fig, use_container_width=True)

col_c, col_d = st.columns(2)
with col_c:
    st.subheader(t("chart.by_server"))
    by_server = df[df["is_anomaly"] == 1]["server_id"].value_counts().reset_index()
    by_server.columns = ["server_id", "count"]
    if not by_server.empty:
        fig = px.bar(by_server, x="server_id", y="count", text="count",
                     labels={"server_id": t("col.server_id"), "count": t("label.anomaly_count")})
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(t("empty.by_server"))

with col_d:
    st.subheader(t("chart.by_type"))
    anomalies = df[df["is_anomaly"] == 1]
    known = anomalies[anomalies["alert_type"] != "Other"]
    other_count = int((anomalies["alert_type"] == "Other").sum())
    by_type = known["alert_type"].value_counts().reset_index()
    by_type.columns = ["alert_type", "count"]
    if not by_type.empty:
        by_type["alert_type"] = by_type["alert_type"].map(type_label)
        fig = px.bar(by_type, x="alert_type", y="count", text="count",
                     labels={"alert_type": t("col.alert_type"), "count": t("label.anomaly_count")})
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(t("empty.by_type"))
    st.caption(t("caption.other", n=other_count))

col_e, col_f = st.columns(2)
with col_e:
    st.subheader(t("chart.cpu_response"))
    normal_label, anomaly_label = t("label.normal"), t("label.anomaly")
    fig = px.scatter(df, x="cpu_usage", y="response_time",
                     color=df["is_anomaly"].map({0: normal_label, 1: anomaly_label}),
                     color_discrete_map={normal_label: "#4C78A8", anomaly_label: "#E45756"},
                     labels={"cpu_usage": t("col.cpu_usage"), "response_time": t("col.response_time"), "color": ""})
    st.plotly_chart(fig, use_container_width=True)

with col_f:
    st.subheader(t("chart.network"))
    fig = px.box(df, y="network_in", points="outliers", labels={"network_in": t("col.network_in")})
    st.plotly_chart(fig, use_container_width=True)

if "role" in df.columns:
    col_g, col_h = st.columns(2)
    with col_g:
        st.subheader(t("chart.cpu_role"))
        st.plotly_chart(px.box(df, x="role", y="cpu_usage", color="role",
                               labels={"role": t("col.role"), "cpu_usage": t("col.cpu_usage")}),
                        use_container_width=True)
    with col_h:
        st.subheader(t("chart.memory_role"))
        st.plotly_chart(px.box(df, x="role", y="memory_usage", color="role",
                               labels={"role": t("col.role"), "memory_usage": t("col.memory_usage")}),
                        use_container_width=True)

    role_summary = (
        df.groupby("role")
        .agg(readings=("is_anomaly", "size"), alerts=("is_anomaly", "sum"),
             median_cpu=("cpu_usage", "median"), median_memory=("memory_usage", "median"),
             median_response=("response_time", "median"))
        .round(1)
    )
    role_summary["alert_rate"] = (role_summary["alerts"] / role_summary["readings"] * 100).round(1)
    role_summary.index.name = "role"
    st.subheader(t("section.role_summary"))
    st.dataframe(role_summary.rename(columns=column_labels(role_summary.columns)).rename_axis(t("col.role")),
                 use_container_width=True)

st.subheader(t("section.latest"))
recent = localize_alerts(df[df["is_anomaly"] == 1].sort_values("timestamp", ascending=False).head(10))
if not recent.empty:
    recent["alert_type"] = recent["alert_type"].map(type_label)
    shown = recent[["timestamp", "server_id", "role", "alert_type", "probable_cause", "top_signals",
                    "suggested_check", "anomaly_score"]]
    st.dataframe(shown.rename(columns=column_labels(shown.columns)), use_container_width=True, hide_index=True)
else:
    st.success(t("ok.no_anomalies"))
