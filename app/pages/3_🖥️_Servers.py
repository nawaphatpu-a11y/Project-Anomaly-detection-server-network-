"""Per-server page: fleet overview and single-server drill-down."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from i18n import column_labels, language_selector, severity_label, t, type_label
from server_view import STATUS_ICON, normal_band, overview, status_counts
from utils import HISTORY_PER_SERVER, init_buffer, live_settings, live_update, load_model_bundle, localize_alerts

METRICS = ["cpu_usage", "memory_usage", "network_in", "network_out", "response_time"]
BAND_METRICS = {"cpu_usage", "memory_usage", "network_in", "network_out", "response_time"}

st.set_page_config(page_title="Servers", page_icon="🖥️", layout="wide")
init_buffer()
language_selector()

st.sidebar.header(t("sidebar.settings"))
refresh_seconds, rows_per_tick, paused = live_settings()

live_update(refresh_seconds, rows_per_tick, paused, "servers_autorefresh")

st.title(t("servers.title"))
st.caption(t("servers.caption"))

df = st.session_state.readings
if df.empty:
    st.info(t("servers.waiting"))
    st.stop()

bundle, _ = load_model_bundle()
baselines = bundle.get("role_baselines") if bundle else None

table = overview(df)
counts = status_counts(table)

m1, m2, m3 = st.columns(3)
m1.metric(t("servers.count_alert"), counts["alert"])
m2.metric(t("servers.count_watch"), counts["watch"])
m3.metric(t("servers.count_ok"), counts["ok"])

st.subheader(t("servers.overview"))
shown = table.copy()
shown["status"] = shown["status"].map(lambda s: f"{STATUS_ICON[s]} {t(f'status.{s}')}")
shown["last_seen"] = shown["last_seen"].astype(str).str[:19]
shown = shown[["server_id", "role", "status", "alerts", "recent_alerts", "alert_rate",
               "cpu_usage", "memory_usage", "response_time", "last_seen"]]
st.dataframe(shown.rename(columns=column_labels(shown.columns)), use_container_width=True, hide_index=True)
st.caption(t("servers.status_help"))

status_of = dict(zip(table["server_id"], table["status"]))
role_of = dict(zip(table["server_id"], table["role"]))
server_ids = sorted(table["server_id"])
if st.session_state.get("server_choice") not in server_ids:
    st.session_state["server_choice"] = table["server_id"].iloc[0]
server = st.selectbox(
    t("servers.select"),
    server_ids,
    format_func=lambda s: f"{s} | {role_of[s]} | {STATUS_ICON[status_of[s]]} {t(f'status.{status_of[s]}')}",
    key="server_choice",
)

st.divider()
st.subheader(t("servers.detail", server=server))

srv = df[df["server_id"] == server].sort_values("timestamp")
role = role_of[server]
latest = srv.iloc[-1]
row = table[table["server_id"] == server].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{t('col.cpu_usage')} ({t('servers.latest')})", f"{latest['cpu_usage']:.1f}%")
c2.metric(f"{t('col.memory_usage')} ({t('servers.latest')})", f"{latest['memory_usage']:.1f}%")
c3.metric(f"{t('col.response_time')} ({t('servers.latest')})", f"{latest['response_time']:.0f}")
c4.metric(t("col.alert_rate"), f"{row['alert_rate']:.1f}%")

metric = st.selectbox(t("servers.metric"), METRICS, format_func=lambda m: t(f"col.{m}"), key="server_metric")
st.caption(t("servers.history", n=len(srv), max=HISTORY_PER_SERVER))
if len(srv) < 20:
    st.info(t("servers.few", n=len(srv)))

fig = px.line(srv, x="timestamp", y=metric, labels={"timestamp": t("col.timestamp"), metric: t(f"col.{metric}")})
band = normal_band(baselines, role, metric) if metric in BAND_METRICS else None
if band:
    fig.add_hrect(y0=band[0], y1=band[1], fillcolor="green", opacity=0.12, line_width=0)
flagged = srv[srv["is_anomaly"] == 1]
if not flagged.empty:
    fig.add_scatter(x=flagged["timestamp"], y=flagged[metric], mode="markers",
                    marker=dict(color="#E45756", size=9), name=t("label.anomaly"))
st.plotly_chart(fig, use_container_width=True)
if band:
    st.caption(t("servers.band"))

st.subheader(t("servers.alert_history"))
if flagged.empty:
    st.success(t("servers.no_alerts"))
else:
    history = localize_alerts(flagged.sort_values("timestamp", ascending=False).head(20))
    history["alert_type"] = history["alert_type"].map(type_label)
    history["severity_level"] = history["severity_level"].map(severity_label)
    columns = ["timestamp", "alert_type", "severity_level", "probable_cause", "top_signals", "suggested_check"]
    st.dataframe(history[columns].rename(columns=column_labels(columns)),
                 use_container_width=True, hide_index=True)

st.subheader(t("servers.peers", role=role))
peers = df[df["role"] == role].copy()
peers["selected"] = (peers["server_id"] == server).map({True: server, False: "—"})
fig = px.box(peers, x="server_id", y=metric, color="selected",
             labels={"server_id": t("col.server_id"), metric: t(f"col.{metric}"), "selected": ""},
             color_discrete_map={server: "#E45756", "—": "#4C78A8"})
st.plotly_chart(fig, use_container_width=True)
