"""Alerts page."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from i18n import column_labels, language_selector, severity_label, t, type_label
from server_view import keep_valid, pick_index
from utils import init_buffer, live_settings, live_update, load_model_bundle, localize_alerts

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
bundle, _ = load_model_bundle()
type_options = ["CPU_SPIKE", "NETWORK_SURGE", "RESPONSE_DEGRADATION", "Other"]
server_options = sorted(bundle["server_ids"]) if bundle else sorted(alerts["server_id"].unique())
severity_options = ["High", "Medium", "Low"]
sort_options = ["severity", "recent"]
with col1:
    type_filter = st.multiselect(t("filter.type"), options=type_options, format_func=type_label,
                                 default=keep_valid(st.session_state.get("saved_type_filter"), type_options))
with col2:
    server_filter = st.multiselect(t("filter.server"), options=server_options,
                                   default=keep_valid(st.session_state.get("saved_server_filter"), server_options))
with col3:
    severity_filter = st.multiselect(t("filter.severity"), options=severity_options, format_func=severity_label,
                                     default=keep_valid(st.session_state.get("saved_severity_filter"), severity_options))
with col4:
    sort_mode = st.selectbox(t("sort.label"), sort_options,
                             index=pick_index(sort_options, st.session_state.get("saved_sort_mode")),
                             format_func=lambda k: t(f"sort.{k}"))
st.session_state["saved_type_filter"] = type_filter
st.session_state["saved_server_filter"] = server_filter
st.session_state["saved_severity_filter"] = severity_filter
st.session_state["saved_sort_mode"] = sort_mode

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
current = filtered.head(50).reset_index(drop=True)
if current.empty:
    st.info(t("details.no_match"))
    st.stop()
current["alert_key"] = current["timestamp"].astype(str) + " | " + current["server_id"]
signature = (tuple(type_filter), tuple(server_filter), tuple(severity_filter), sort_mode)
refresh_list = st.button(t("details.refresh"))
if refresh_list or "alert_pool" not in st.session_state or st.session_state.get("alert_pool_signature") != signature:
    st.session_state["alert_pool"] = current
    st.session_state["alert_pool_signature"] = signature
    st.session_state["alert_pool_time"] = pd.Timestamp.now().strftime("%H:%M:%S")
candidates = localize_alerts(st.session_state["alert_pool"])
labels = dict(zip(candidates["alert_key"], candidates["alert_key"] + " | " + candidates["alert_type"].map(type_label)))
alert_keys = candidates["alert_key"].tolist()
choice = st.selectbox(t("details.select"), alert_keys,
                      index=pick_index(alert_keys, st.session_state.get("saved_alert_choice")),
                      format_func=lambda k: labels[k])
st.session_state["saved_alert_choice"] = choice
selected = candidates[candidates["alert_key"] == choice].iloc[0]
d1, d2 = st.columns(2)
d1.metric(t("details.cause"), selected["probable_cause"])
d2.metric(t("details.role"), selected["role"])
st.write(f"**{t('details.signals')}** {selected['top_signals']}")
st.info(t("details.check", text=selected["suggested_check"]))
st.caption(t("details.snapshot", ts=st.session_state["alert_pool_time"]))
st.caption(t("details.caveat"))
