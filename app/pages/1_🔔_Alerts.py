"""Alerts page."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from i18n import column_labels, language_selector, severity_label, t, type_label
from server_view import keep_valid, pick_index
from utils import (SEVERITY_RANK, current_incidents, init_buffer, live_settings, live_update, load_model_bundle,
                   localize_alerts, record_feedback)

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

incidents = current_incidents()

bundle, _ = load_model_bundle()
type_options = ["CPU_SPIKE", "NETWORK_SURGE", "RESPONSE_DEGRADATION", "Other"]
server_options = sorted(bundle["server_ids"]) if bundle else sorted(alerts["server_id"].unique())
severity_options = ["High", "Medium", "Low"]
detection_options = ["both", "zscore", "forest"]
sort_options = ["severity", "recent"]

col1, col2, col3, col4, col5 = st.columns(5)
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
    detection_filter = st.multiselect(t("filter.detection"), options=detection_options,
                                      format_func=lambda d: t(f"detection.{d}"), help=t("filter.detection_help"),
                                      default=keep_valid(st.session_state.get("saved_detection_filter"), detection_options))
with col5:
    sort_mode = st.selectbox(t("sort.label"), sort_options,
                             index=pick_index(sort_options, st.session_state.get("saved_sort_mode")),
                             format_func=lambda k: t(f"sort.{k}"))
st.session_state["saved_type_filter"] = type_filter
st.session_state["saved_server_filter"] = server_filter
st.session_state["saved_severity_filter"] = severity_filter
st.session_state["saved_detection_filter"] = detection_filter
st.session_state["saved_sort_mode"] = sort_mode

filtered = incidents
if type_filter:
    filtered = filtered[filtered["alert_type"].isin(type_filter)]
if server_filter:
    filtered = filtered[filtered["server_id"].isin(server_filter)]
if severity_filter:
    filtered = filtered[filtered["severity_level"].isin(severity_filter)]
if detection_filter:
    filtered = filtered[filtered["detection"].isin(detection_filter)]

if sort_mode == "severity":
    filtered = (filtered.assign(_rank=filtered["severity_level"].map(SEVERITY_RANK))
                .sort_values(["_rank", "start"], ascending=[False, False]).drop(columns="_rank"))
else:
    filtered = filtered.sort_values("start", ascending=False)

filtered = localize_alerts(filtered.assign(is_anomaly=1))

st.caption(t("alerts.updated", ts=alerts["timestamp"].max()))

table_columns = ["start", "server_id", "role", "alert_type", "severity_level", "detection", "readings", "status",
                 "probable_cause", "top_signals"]
display = filtered[table_columns].copy()
display["start"] = display["start"].astype(str).str[:19]
display["alert_type"] = display["alert_type"].map(type_label)
display["severity_level"] = display["severity_level"].map(severity_label)
display["detection"] = display["detection"].map(lambda d: t(f"detection.{d}"))
display["status"] = display["status"].map(lambda s: t(f"status.{s}"))
display = display.rename(columns=column_labels(table_columns))

col_m1, col_m2, col_dl = st.columns([1, 1, 1])
with col_m1:
    st.metric(t("alerts.incidents"), len(incidents))
with col_m2:
    st.metric(t("alerts.readings"), st.session_state["total_anomalies"])
with col_dl:
    csv_bytes = display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        t("alerts.download"),
        data=csv_bytes,
        file_name=f"incidents_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        width="stretch",
    )

st.dataframe(display, width="stretch", hide_index=True)

st.subheader(t("details.title"))
current = filtered.head(50).reset_index(drop=True)
if current.empty:
    st.info(t("details.no_match"))
    st.stop()
signature = (tuple(type_filter), tuple(server_filter), tuple(severity_filter), tuple(detection_filter), sort_mode)
refresh_list = st.button(t("details.refresh"))
if refresh_list or "alert_pool" not in st.session_state or st.session_state.get("alert_pool_signature") != signature:
    st.session_state["alert_pool"] = current[["incident_id", "start", "server_id", "alert_type"]]
    st.session_state["alert_pool_signature"] = signature
    st.session_state["alert_pool_time"] = pd.Timestamp.now().strftime("%H:%M:%S")
pool = st.session_state["alert_pool"]
labels = {row.incident_id: f"{str(row.start)[:19]} | {row.server_id} | {type_label(row.alert_type)}"
          for row in pool.itertuples()}
incident_ids = list(labels)
choice = st.selectbox(t("details.select"), incident_ids,
                      index=pick_index(incident_ids, st.session_state.get("saved_alert_choice")),
                      format_func=lambda k: labels[k])
st.session_state["saved_alert_choice"] = choice

fresh = localize_alerts(incidents.assign(is_anomaly=1))
selected_rows = fresh[fresh["incident_id"] == choice]
if selected_rows.empty:
    st.info(t("details.no_match"))
else:
    selected = selected_rows.iloc[0]
    d1, d2, d3 = st.columns(3)
    d1.metric(t("details.cause"), selected["probable_cause"])
    d2.metric(t("details.role"), selected["role"])
    d3.metric(t("details.detected_by"), t(f"detection.{selected['detection']}"))
    st.write(f"**{t('details.signals')}** {selected['top_signals']}")
    st.write(t("details.span", readings=int(selected["readings"]), start=str(selected["start"])[:19],
               end=str(selected["end"])[:19]))
    st.info(t("details.check", text=selected["suggested_check"]))

    st.markdown(f"**{t('feedback.title')}**")
    f1, f2 = st.columns(2)
    f1.button(t("feedback.confirm"), on_click=record_feedback, args=(choice, "confirmed", selected["detection"]))
    f2.button(t("feedback.false"), on_click=record_feedback, args=(choice, "false", selected["detection"]))
    given = st.session_state["feedback"].get(choice)
    if given:
        st.caption(t("feedback.saved", verdict=t(f"feedback.v_{given['verdict']}")))

st.caption(t("details.snapshot", ts=st.session_state["alert_pool_time"]))
st.caption(t("details.caveat"))

st.subheader(t("feedback.summary"))
feedback = st.session_state["feedback"]
if not feedback:
    st.caption(t("feedback.none"))
else:
    answers = pd.DataFrame(feedback.values())
    summary = answers.pivot_table(index="detection", columns="verdict", aggfunc="size", fill_value=0)
    for column in ("confirmed", "false"):
        if column not in summary:
            summary[column] = 0
    summary["share"] = (summary["confirmed"] / (summary["confirmed"] + summary["false"]) * 100).round(0).astype(int).astype(str) + "%"
    summary.index = [t(f"detection.{d}") for d in summary.index]
    summary = summary[["confirmed", "false", "share"]]
    summary.columns = [t("feedback.col_real"), t("feedback.col_false"), t("feedback.col_share")]
    st.table(summary.rename_axis(t("col.detection")))

with st.expander(t("alerts.readings_title")):
    reading_rows = localize_alerts(alerts.sort_values("timestamp", ascending=False).head(200))
    reading_rows["alert_type"] = reading_rows["alert_type"].map(type_label)
    reading_rows["severity_level"] = reading_rows["severity_level"].map(severity_label)
    reading_rows["detection"] = reading_rows["detection"].map(lambda d: t(f"detection.{d}"))
    reading_columns = ["timestamp", "server_id", "alert_type", "severity_level", "detection", "probable_cause",
                       "top_signals", "cpu_usage", "memory_usage", "network_in", "network_out", "response_time",
                       "anomaly_score"]
    st.dataframe(reading_rows[reading_columns].rename(columns=column_labels(reading_columns)),
                 width="stretch", hide_index=True)
