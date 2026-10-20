"""
เทสต์หน่วยของ generate_data.py, etl.py และ train_model.py
รัน: pytest -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import generate_data
import etl
import train_model


def test_make_server_profiles_shape():
    rng = np.random.default_rng(1)
    profiles = generate_data.make_server_profiles(5, rng)
    assert len(profiles) == 5
    assert {"server_id", "hostname", "role", "base_cpu"}.issubset(profiles.columns)


def test_generate_baseline_within_clipped_bounds():
    rng = np.random.default_rng(1)
    profiles = generate_data.make_server_profiles(3, rng)
    hours = np.array([10] * 200)
    baseline = generate_data.generate_baseline(200, hours, profiles.iloc[0], rng)
    assert (baseline["cpu_usage"] >= 1).all() and (baseline["cpu_usage"] <= 99).all()
    assert (baseline["network_in"] >= 1).all()


def test_inject_true_anomalies_sets_positive_labels():
    rng = np.random.default_rng(1)
    profiles = generate_data.make_server_profiles(3, rng)
    hours = np.array([10] * 200)
    frames = []
    for _, p in profiles.iterrows():
        baseline = generate_data.generate_baseline(200, hours, p, rng)
        frame = pd.DataFrame({"server_id": p["server_id"], **baseline})
        frame["is_anomaly"] = 0
        frame["anomaly_type"] = "NONE"
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = generate_data.inject_true_anomalies(df, rng, target_ratio=0.05)
    assert df["is_anomaly"].sum() > 0
    assert set(df.loc[df["is_anomaly"] == 1, "anomaly_type"].unique()).issubset(
        {"CPU_SPIKE", "NETWORK_SURGE", "RESPONSE_DEGRADATION"}
    )


def test_inject_data_errors_never_touches_true_anomaly_rows():
    rng = np.random.default_rng(7)
    profiles = generate_data.make_server_profiles(3, rng)
    hours = np.array([10] * 200)
    frames = []
    for _, p in profiles.iterrows():
        baseline = generate_data.generate_baseline(200, hours, p, rng)
        frame = pd.DataFrame({"server_id": p["server_id"], **baseline})
        frame["is_anomaly"] = 0
        frame["anomaly_type"] = "NONE"
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = generate_data.inject_true_anomalies(df, rng, target_ratio=0.1)
    df = generate_data.inject_data_errors(df, rng, error_ratio=0.1)

    corrupted = (
        (df["cpu_usage"] > 100) | (df["response_time"] < 0)
        | df[["cpu_usage", "memory_usage", "network_in", "network_out", "response_time"]].isna().any(axis=1)
    )
    overlap = df[(df["is_anomaly"] == 1) & corrupted]
    assert len(overlap) == 0, "data error ไปทับ True Anomaly row -- บั๊กเดิมกลับมาอีกแล้ว"


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(3)
    profiles = generate_data.make_server_profiles(3, rng)
    hours = np.array([10] * 300)
    frames = []
    for _, p in profiles.iterrows():
        baseline = generate_data.generate_baseline(300, hours, p, rng)
        frame = pd.DataFrame({
            "server_id": p["server_id"],
            "timestamp": pd.date_range("2026-01-01", periods=300, freq="5min"),
            **baseline,
        })
        frame["is_anomaly"] = 0
        frame["anomaly_type"] = "NONE"
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = generate_data.inject_true_anomalies(df, rng, target_ratio=0.05)
    df = generate_data.inject_data_errors(df, rng, error_ratio=0.03)
    return df


def test_fix_data_errors_removes_impossible_values(raw_df):
    cleaned = etl.fix_data_errors(raw_df.copy())
    assert cleaned["cpu_usage"].max() <= 100 or pd.isna(cleaned["cpu_usage"]).any()
    assert (cleaned["response_time"].dropna() >= 0).all()


def test_handle_missing_leaves_no_nan(raw_df):
    cleaned = etl.fix_data_errors(raw_df.copy())
    filled = etl.handle_missing(cleaned)
    metric_cols = ["cpu_usage", "memory_usage", "network_in", "network_out", "response_time"]
    assert filled[metric_cols].isna().sum().sum() == 0


def test_fit_apply_min_max_train_only_stays_in_zero_one_range(raw_df):
    df = etl.fix_data_errors(raw_df.copy())
    df = etl.handle_missing(df)
    scale_info = train_model.fit_min_max(df, ["cpu_usage", "memory_usage"])
    df = train_model.apply_min_max(df, ["cpu_usage", "memory_usage"], scale_info)
    assert df["cpu_usage_norm"].between(0, 1).all()
    assert df["memory_usage_norm"].between(0, 1).all()
    assert "cpu_usage" in scale_info and "min" in scale_info["cpu_usage"]


def test_apply_min_max_clips_out_of_range_values():
    train_part = pd.DataFrame({"cpu_usage": [10.0, 20.0, 30.0]})
    scale_info = train_model.fit_min_max(train_part, ["cpu_usage"])
    unseen_part = pd.DataFrame({"cpu_usage": [-5.0, 100.0]})
    result = train_model.apply_min_max(unseen_part, ["cpu_usage"], scale_info)
    assert result["cpu_usage_norm"].between(0, 1).all()


def test_trend_feature_is_causal_not_using_future_values():
    df = pd.DataFrame({
        "server_id": ["SRV-001"] * 20,
        "response_time": [50.0] * 15 + [500.0] * 5,
    })
    result_before_spike = etl.add_trend_feature(df.copy(), window=5)
    df2 = df.copy()
    df2.loc[18, "response_time"] = 9999.0
    result_after_edit = etl.add_trend_feature(df2, window=5)
    assert result_before_spike.loc[14, "response_time_dev_from_roll"] == \
        result_after_edit.loc[14, "response_time_dev_from_roll"]


def test_time_based_split_no_row_overlap_and_chronological():
    df = pd.DataFrame({
        "server_id": ["SRV-001"] * 10 + ["SRV-002"] * 10,
        "timestamp": list(pd.date_range("2026-01-01", periods=10, freq="h")) * 2,
    })
    train, test = train_model.time_based_split(df, test_ratio=0.2)
    assert len(train) + len(test) == len(df)
    assert set(train.index).isdisjoint(set(test.index))
    for server in df["server_id"].unique():
        train_max = train.loc[train["server_id"] == server, "timestamp"].max()
        test_min = test.loc[test["server_id"] == server, "timestamp"].min()
        assert train_max < test_min, "test set ต้องมาทีหลัง train set เสมอ (time-based split)"


def test_fix_timestamp_errors_removes_malformed_rows():
    df = pd.DataFrame({
        "server_id": ["SRV-001"] * 4,
        "timestamp": ["2026-01-01 00:00:00", "not-a-date", "2026-01-01 00:05:00", ""],
        "cpu_usage": [10, 20, 30, 40],
    })
    cleaned = etl.fix_timestamp_errors(df)
    assert len(cleaned) == 2
    assert cleaned["timestamp"].notna().all()


def test_add_severity_maps_known_types_and_defaults_none():
    df = pd.DataFrame({"anomaly_type": ["CPU_SPIKE", "NETWORK_SURGE", "RESPONSE_DEGRADATION", "NONE", "WEIRD_UNKNOWN"]})
    result = etl.add_severity(df)
    assert result["severity_level"].tolist() == ["High", "High", "Medium", "None", "None"]


def test_iqr_baseline_flags_far_outliers_not_normal_points():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    test = pd.DataFrame({"x": [0.1, -0.2, 50.0, -50.0]})
    pred = train_model.iqr_baseline(train, test)
    assert list(pred) == [0, 0, 1, 1]


def test_server_profiles_cover_all_roles_and_differ_in_baseline():
    rng = np.random.default_rng(3)
    profiles = generate_data.make_server_profiles(15, rng)
    assert set(profiles["role"]) == set(generate_data.SERVER_ROLES)
    med = profiles.groupby("role")[["base_cpu", "base_mem", "base_net_out", "base_resp"]].mean()
    assert med.loc["Database Server", "base_mem"] > med.loc["Web Server", "base_mem"] + 15
    assert med.loc["App Server", "base_cpu"] > med.loc["Web Server", "base_cpu"] + 10
    assert med.loc["Web Server", "base_net_out"] > med.loc["Database Server", "base_net_out"] + 40


def test_cpu_spike_never_pushes_memory_above_100():
    rng = np.random.default_rng(5)
    profiles = generate_data.make_server_profiles(6, rng)
    hours = np.array([10] * 300)
    frames = []
    for _, p in profiles.iterrows():
        baseline = generate_data.generate_baseline(300, hours, p, rng)
        frame = pd.DataFrame({"server_id": p["server_id"], **baseline})
        frame["is_anomaly"] = 0
        frame["anomaly_type"] = "NONE"
        frames.append(frame)
    df = generate_data.inject_true_anomalies(pd.concat(frames, ignore_index=True), rng, target_ratio=0.3)
    assert df["memory_usage"].max() <= 100


def test_role_relative_feature_is_zero_at_role_median_and_flags_other_roles():
    df = pd.DataFrame({
        "role": ["Web Server"] * 5 + ["App Server"] * 5,
        "cpu_usage": [30, 31, 32, 33, 34, 55, 56, 57, 58, 59],
        "memory_usage": [30] * 10, "network_in": [100] * 10,
        "network_out": [100] * 10, "response_time": [80] * 10,
    })
    baselines = etl.compute_role_baselines(df)
    out = etl.add_role_relative_features(df, baselines)
    assert out.loc[2, "cpu_usage_role_z"] == 0 and out.loc[7, "cpu_usage_role_z"] == 0
    web_with_app_level = pd.DataFrame({"role": ["Web Server"], "cpu_usage": [57], "memory_usage": [30],
                                       "network_in": [100], "network_out": [100], "response_time": [80]})
    z = etl.add_role_relative_features(web_with_app_level, baselines)["cpu_usage_role_z"].iloc[0]
    assert z > 10


def test_role_relative_feature_falls_back_to_global_for_unseen_role():
    df = pd.DataFrame({"role": ["Web Server"] * 4, "cpu_usage": [30, 31, 32, 33],
                       "memory_usage": [30] * 4, "network_in": [100] * 4,
                       "network_out": [100] * 4, "response_time": [80] * 4})
    baselines = etl.compute_role_baselines(df)
    unseen = df.iloc[:1].copy()
    unseen["role"] = "Cache Server"
    out = etl.add_role_relative_features(unseen, baselines)
    assert np.isfinite(out["cpu_usage_role_z"].iloc[0])


import diagnose


def _features(**overrides):
    base = {"cpu_usage_role_z": 0.0, "memory_usage_role_z": 0.0, "network_in_role_z": 0.0,
            "network_out_role_z": 0.0, "response_time_role_z": 0.0, "response_time_dev_from_roll": 0.0}
    base.update(overrides)
    return base


def test_diagnose_cpu_spike():
    out = diagnose.diagnose(_features(cpu_usage_role_z=9.0, memory_usage_role_z=3.0))
    assert out["cause_type"] == "CPU_SPIKE"
    assert out["top_signals"].startswith("CPU")


def test_diagnose_network_surge_needs_both_directions():
    both = diagnose.diagnose(_features(network_in_role_z=8.0, network_out_role_z=7.0))
    only_in = diagnose.diagnose(_features(network_in_role_z=8.0))
    assert both["cause_type"] == "NETWORK_SURGE"
    assert only_in["cause_type"] == "UNCLASSIFIED"


def test_diagnose_response_degradation_requires_rising_trend():
    rising = diagnose.diagnose(_features(response_time_role_z=6.0, response_time_dev_from_roll=20.0))
    flat = diagnose.diagnose(_features(response_time_role_z=6.0, response_time_dev_from_roll=-5.0))
    assert rising["cause_type"] == "RESPONSE_DEGRADATION"
    assert flat["cause_type"] == "UNCLASSIFIED"


def test_diagnose_unclassified_when_nothing_stands_out():
    out = diagnose.diagnose(_features(cpu_usage_role_z=1.0))
    assert out["cause_type"] == "UNCLASSIFIED"
    assert out["top_signals"] == "none above 2 MAD"


def test_diagnose_suggestion_depends_on_role():
    features = _features(response_time_role_z=6.0, response_time_dev_from_roll=20.0)
    db = diagnose.diagnose(features, role="Database Server")["suggested_check"]
    web = diagnose.diagnose(features, role="Web Server")["suggested_check"]
    unknown = diagnose.diagnose(features, role="Cache Server")["suggested_check"]
    assert "slow queries" in db and db != web
    assert unknown == diagnose.CAUSES["RESPONSE_DEGRADATION"]["en"]["checks"]["default"]


def test_diagnose_tolerates_missing_and_nan_features():
    out = diagnose.diagnose({"cpu_usage_role_z": float("nan")})
    assert out["cause_type"] == "UNCLASSIFIED"


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import i18n


def test_every_ui_text_exists_in_both_languages_with_same_placeholders():
    import string

    def placeholders(text):
        return {name for _, name, _, _ in string.Formatter().parse(text) if name}

    assert set(i18n.TEXT["en"]) == set(i18n.TEXT["th"])
    for key in i18n.TEXT["en"]:
        assert placeholders(i18n.TEXT["en"][key]) == placeholders(i18n.TEXT["th"][key]), key


def test_translation_switches_language_and_formats_values():
    assert i18n.t("sidebar.clear", lang="EN") == "Clear all data"
    assert i18n.t("sidebar.clear", lang="TH") == "ล้างข้อมูลทั้งหมด"
    assert "5" in i18n.t("sidebar.caption", lang="TH", rows=5, seconds=4)
    assert i18n.t("no.such.key", lang="TH") == "no.such.key"


def test_label_helpers_fall_back_to_the_raw_value():
    assert i18n.type_label("CPU_SPIKE", "TH") != "CPU_SPIKE"
    assert i18n.type_label("SOMETHING_NEW", "TH") == "SOMETHING_NEW"
    assert i18n.column_labels(["cpu_usage", "unknown"], "EN") == {"cpu_usage": "CPU"}


def test_signals_survive_encode_decode_and_render_in_both_languages():
    signals = [("network_in_role_z", 72.2), ("cpu_usage_role_z", -2.1)]
    assert diagnose.decode_signals(diagnose.encode_signals(signals)) == signals
    assert diagnose.format_signals(signals, "en") == "Network in +72.2, CPU -2.1"
    assert diagnose.format_signals(signals, "th") == "Network เข้า +72.2, CPU -2.1"
    assert diagnose.decode_signals(None) == []
    assert diagnose.format_signals([], "th") == diagnose.NO_SIGNAL["th"]


def test_diagnosis_text_follows_requested_language():
    features = _features(cpu_usage_role_z=9.0)
    en = diagnose.diagnose(features, role="Web Server", lang="en")
    th = diagnose.diagnose(features, role="Web Server", lang="th")
    assert en["cause_type"] == th["cause_type"] == "CPU_SPIKE"
    assert en["probable_cause"] != th["probable_cause"]
    assert en["suggested_check"] != th["suggested_check"]


import server_view


def _readings():
    rows = []
    for server, role, flags in [("A", "Web Server", [0, 0, 0, 1]), ("B", "App Server", [0, 1, 0, 0]),
                                ("C", "App Server", [0, 0, 0, 0])]:
        for i, flag in enumerate(flags):
            rows.append({"server_id": server, "role": role, "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
                         "is_anomaly": flag, "cpu_usage": 30.0, "memory_usage": 40.0, "response_time": 90.0})
    return pd.DataFrame(rows)


def test_server_overview_status_and_ordering():
    table = server_view.overview(_readings())
    status = dict(zip(table["server_id"], table["status"]))
    assert status == {"A": "alert", "B": "watch", "C": "ok"}
    assert table["server_id"].tolist() == ["A", "B", "C"]
    assert table.set_index("server_id").loc["B", "alert_rate"] == 25.0
    assert server_view.status_counts(table) == {"alert": 1, "watch": 1, "ok": 1}


def test_server_overview_uses_latest_reading_by_time_not_row_order():
    df = _readings().sort_values("timestamp", ascending=False)
    status = dict(zip(*server_view.overview(df)[["server_id", "status"]].values.T))
    assert status["A"] == "alert"


def test_normal_band_uses_role_baseline_and_handles_unknown_role():
    baselines = {"Web Server": {"cpu_usage": {"median": 30.0, "mad": 4.0}}}
    low, high = server_view.normal_band(baselines, "Web Server", "cpu_usage")
    assert low < 30.0 < high
    assert round(high - 30.0, 2) == round(30.0 - low, 2)
    assert server_view.normal_band(baselines, "Cache Server", "cpu_usage") is None
    assert server_view.normal_band(None, "Web Server", "cpu_usage") is None


def test_pick_index_restores_saved_choice_and_falls_back():
    options = ["SRV-001", "SRV-002", "SRV-003"]
    assert server_view.pick_index(options, "SRV-003") == 2
    assert server_view.pick_index(options, "SRV-999") == 0
    assert server_view.pick_index(options, None, fallback=1) == 1


def test_keep_valid_drops_choices_that_no_longer_exist():
    assert server_view.keep_valid(["a", "z"], ["a", "b"]) == ["a"]
    assert server_view.keep_valid(None, ["a"]) == []


def _history():
    return pd.DataFrame({
        "alert_type": ["CPU_SPIKE", "Other", "Other", "NETWORK_SURGE"],
        "signals_raw": [
            "cpu_usage_role_z:12.40;memory_usage_role_z:4.00",
            "memory_usage_role_z:-3.80;network_in_role_z:-2.40",
            "cpu_usage_role_z:-4.20",
            "network_out_role_z:70.40;network_in_role_z:43.50",
        ],
    })


def test_history_filter_by_type_and_hide_other():
    df = _history()
    assert len(server_view.filter_history(df)) == 4
    assert server_view.filter_history(df, types=["NETWORK_SURGE"]).index.tolist() == [3]
    assert server_view.filter_history(df, hide_other=True).index.tolist() == [0, 3]


def test_history_filter_by_metric_in_strongest_signals():
    df = _history()
    assert server_view.filter_history(df, metrics=["memory_usage"]).index.tolist() == [0, 1]
    assert server_view.filter_history(df, metrics=["cpu_usage", "network_out"]).index.tolist() == [0, 2, 3]


def test_history_filter_above_normal_only_drops_low_values():
    df = _history()
    assert server_view.filter_history(df, metrics=["memory_usage"], above_only=True).index.tolist() == [0]
    assert server_view.filter_history(df, above_only=True).index.tolist() == [0, 3]


def test_history_filter_combines_conditions_and_tolerates_missing_signals():
    df = _history()
    assert server_view.filter_history(df, types=["Other"], metrics=["cpu_usage"]).index.tolist() == [2]
    df.loc[1, "signals_raw"] = None
    assert server_view.filter_history(df, metrics=["memory_usage"]).index.tolist() == [0]


def test_role_relative_features_are_not_model_inputs_but_remain_in_output_for_diagnosis():
    for cols in (train_model.ISO_FEATURE_COLS, train_model.LOF_FEATURE_COLS,
                 train_model.ZSCORE_FEATURE_COLS, train_model.IQR_FEATURE_COLS):
        assert not set(train_model.ROLE_Z_COLS) & set(cols)
    assert set(train_model.ROLE_Z_COLS) <= set(train_model.ALL_FEATURE_COLS)


import replay


def _replay_frame():
    rows = []
    for server in ("A", "B"):
        for i in range(3):
            rows.append({"server_id": server, "role": "Web Server", "cpu_usage": float(i), "memory_usage": 40.0,
                         "network_in": 100.0, "network_out": 100.0, "response_time": 80.0,
                         "is_anomaly": int(i == 2), "anomaly_type": "CPU_SPIKE" if i == 2 else "NONE"})
    return pd.DataFrame(rows)


def test_replay_feed_steps_in_order_per_server_and_wraps_around():
    feed = replay.ReplayFeed(_replay_frame())
    cursors = {}
    seen = [feed.next_reading("A", cursors) for _ in range(4)]
    assert [r["cpu_usage"] for r, _ in seen] == [0.0, 1.0, 2.0, 0.0]
    assert [wrapped for _, wrapped in seen] == [False, False, False, True]
    assert feed.next_reading("B", cursors)[0]["cpu_usage"] == 0.0
    assert not feed.has("C")


def test_replay_sample_keeps_the_earliest_rows_of_each_server_and_the_needed_columns():
    rows = []
    for server in ("A", "B"):
        for i in range(10):
            rows.append({"server_id": server, "role": "Web Server", "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
                         "cpu_usage": float(i), "memory_usage": 40.0, "network_in": 1.0, "network_out": 1.0,
                         "response_time": 80.0, "is_anomaly": 0, "anomaly_type": "NONE", "extra": 1})
    sample = train_model.build_replay_sample(pd.DataFrame(rows).sample(frac=1, random_state=0), rows_per_server=4)
    assert list(sample.columns) == train_model.REPLAY_COLUMNS
    assert len(sample) == 8
    assert sorted(sample[sample.server_id == "A"]["cpu_usage"]) == [0.0, 1.0, 2.0, 3.0]


import detection


def test_zscore_flag_matches_the_benchmark_rule():
    train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0] * 20, "b": [10.0, 10.0, 11.0, 9.0, 10.0] * 20})
    stats = detection.zscore_stats(train, ["a", "b"], threshold=3.0)
    assert detection.zscore_flag({"a": 2.0, "b": 10.0}, stats) is False
    assert detection.zscore_flag({"a": 100.0, "b": 10.0}, stats) is True
    assert detection.zscore_flag({"a": 2.0, "b": -50.0}, stats) is True
    test = pd.DataFrame({"a": [2.0, 100.0], "b": [10.0, 10.0]})
    benchmark = train_model.zscore_baseline(train, test, threshold=3.0)
    assert [detection.zscore_flag(r, stats) for r in test.to_dict("records")] == [bool(x) for x in benchmark]


def test_constant_feature_does_not_divide_by_zero():
    train = pd.DataFrame({"a": [5.0] * 10})
    stats = detection.zscore_stats(train, ["a"])
    assert stats["std"]["a"] == 1
    assert detection.zscore_flag({"a": 5.0}, stats) is False


def test_detection_source_labels_each_combination():
    assert detection.detection_source(True, True) == "both"
    assert detection.detection_source(False, True) == "zscore"
    assert detection.detection_source(True, False) == "forest"
    assert detection.detection_source(False, False) is None


def _alerts():
    base = pd.Timestamp("2026-01-01 10:00:00")
    rows = []
    def add(server, no, kind, severity, source, score):
        rows.append({"server_id": server, "reading_no": no, "timestamp": base + pd.Timedelta(minutes=no), "role": "Web Server",
                     "alert_type": kind, "severity_level": severity, "detection": source, "anomaly_score": score,
                     "signals_raw": f"cpu_usage_role_z:{abs(score) * 100:.1f}"})
    for no in (10, 11, 12):
        add("A", no, "CPU_SPIKE", "High", "forest" if no < 12 else "both", -0.05 * (no - 9))
    add("A", 14, "Other", "Low", "forest", -0.01)
    add("A", 40, "NETWORK_SURGE", "High", "zscore", 0.02)
    add("B", 11, "Other", "Low", "forest", -0.02)
    return pd.DataFrame(rows)


def test_alerts_close_together_are_merged_into_one_incident():
    incidents = detection.group_incidents(_alerts(), {"A": 41, "B": 60})
    a = incidents[incidents["server_id"] == "A"].sort_values("start")
    assert len(incidents) == 3
    assert a.iloc[0]["readings"] == 4 and a.iloc[0]["alert_type"] == "CPU_SPIKE"
    assert a.iloc[0]["detection"] == "both"
    assert a.iloc[0]["incident_id"] == "A-10"
    assert a.iloc[1]["alert_type"] == "NETWORK_SURGE" and a.iloc[1]["detection"] == "zscore"


def test_incident_status_depends_on_how_recent_the_last_flagged_reading_is():
    incidents = detection.group_incidents(_alerts(), {"A": 41, "B": 60}).set_index("incident_id")
    assert incidents.loc["A-40", "status"] == "ongoing"
    assert incidents.loc["A-10", "status"] == "ended"
    assert incidents.loc["B-11", "status"] == "ended"


def test_incident_uses_the_most_severe_reading_and_the_strongest_signal():
    incidents = detection.group_incidents(_alerts(), {"A": 41, "B": 60}).set_index("incident_id")
    assert incidents.loc["A-10", "severity_level"] == "High"
    assert incidents.loc["A-10", "signals_raw"] == "cpu_usage_role_z:15.0"


def test_no_alerts_gives_an_empty_incident_table_with_the_expected_columns():
    empty = detection.group_incidents(pd.DataFrame())
    assert empty.empty and list(empty.columns) == detection.INCIDENT_COLUMNS
