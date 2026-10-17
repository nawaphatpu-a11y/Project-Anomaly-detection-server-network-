"""Alerts page."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from i18n import column_labels, language_selector, severity_label, t, type_label
from utils import init_buffer, live_settings, live_update, localize_alerts

st.set_page_config(page_title="Alerts", page_icon="🔔", layout="wide")
init_buffer()
language_selector()

st.sidebar.header(t("sidebar.settings"))
refresh_seconds, rows_per_tick, paused = live_settings()

live_update(refresh_seconds, rows_per_tick, paused, "alerts_autorefresh")

st.title(t("alerts.title"))

alerts = st.session_state.alerts

if alerts.empty:
    st.success(t("alerts.none"))
    st.stop()

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    type_filter = st.multiselect(t("filter.type"), options=sorted(alerts["alert_type"].dropna().unique()),
                                 format_func=type_label, key="filter_type")
with col2:
    server_filter = st.multiselect(t("filter.server"), options=sorted(alerts["server_id"].unique()), key="filter_server")
with col3:
    severity_filter = st.multiselect(t("filter.severity"), options=["High", "Medium", "Low"],
                                     format_func=severity_label, key="filter_severity")
with col4:
    sort_mode = st.selectbox(t("sort.label"), ["severity", "recent"], format_func=lambda k: t(f"sort.{k}"), key="sort_mode")

filtered = alerts.copy()
if type_filter:
    filtered = filtered[filtered["alert_type"].isin(type_filter)]
if server_filter:
    filtered = filtered[filtered["server_id"].isin(server_filter)]
if severity_filter:
    filtered = filtered[filtered["severity_level"].isin(severity_filter)]

if sort_mode == "severity":
    filtered = filtered.sort_values("anomaly_score", ascending=True)
else:
    filtered = filtered.sort_values("timestamp", ascending=False)

filtered = localize_alerts(filtered)

st.caption(t("alerts.updated", ts=alerts["timestamp"].max()))

table_columns = ["timestamp", "server_id", "alert_type", "severity_level", "probable_cause", "top_signals",
                 "cpu_usage", "memory_usage", "network_in", "network_out", "response_time", "anomaly_score"]
display = filtered[table_columns].copy()
display["alert_type"] = display["alert_type"].map(type_label)
display["severity_level"] = display["severity_level"].map(severity_label)
display = display.rename(columns=column_labels(table_columns))

col_m, col_dl = st.columns([3, 1])
with col_m:
    st.metric(t("alerts.total"), len(alerts))
with col_dl:
    csv_bytes = display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        t("alerts.download"),
        data=csv_bytes,
        file_name=f"alerts_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.dataframe(display, use_container_width=True, hide_index=True)

st.subheader(t("details.title"))
candidates = filtered.head(50).reset_index(drop=True)
if candidates.empty:
    st.info(t("details.no_match"))
    st.stop()
candidates["alert_key"] = candidates["timestamp"].astype(str) + " | " + candidates["server_id"]
labels = dict(zip(candidates["alert_key"], candidates["alert_key"] + " | " + candidates["alert_type"].map(type_label)))
if st.session_state.get("alert_choice") not in labels:
    st.session_state["alert_choice"] = candidates["alert_key"].iloc[0]
choice = st.selectbox(t("details.select"), candidates["alert_key"].tolist(),
                      format_func=lambda k: labels[k], key="alert_choice")
selected = candidates[candidates["alert_key"] == choice].iloc[0]
d1, d2 = st.columns(2)
d1.metric(t("details.cause"), selected["probable_cause"])
d2.metric(t("details.role"), selected["role"])
st.write(f"**{t('details.signals')}** {selected['top_signals']}")
st.info(t("details.check", text=selected["suggested_check"]))
st.caption(t("details.caveat"))
