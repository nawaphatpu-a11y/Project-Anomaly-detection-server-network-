"""
tests/test_pipeline.py
เทสต์ระดับ unit ของฟังก์ชันหลักใน generate_data.py / etl.py / train_model.py
รัน: pytest -v (จาก root ของโปรเจกต์ — conftest.py เติม path ให้ import สคริปต์ระดับ root ได้)

เทสต์บางตัวในนี้เขียนขึ้นเพื่อ "กันบั๊กที่เคยเจอจริง" ไม่ให้กลับมาอีก (regression test)
ไม่ใช่แค่เช็คว่ารันไม่พังผิวเผิน
"""
import numpy as np
import pandas as pd
import pytest

import generate_data
import etl
import train_model


# ---------- generate_data.py ----------

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
    # REGRESSION TEST: บั๊กเดิมคือ inject_data_errors สุ่มจาก df.index ทั้งหมด รวมถึงแถวที่
    # เพิ่งถูกทำเป็น True Anomaly ไปแล้ว ทำให้ label เพี้ยนตอน ETL เติม missing ทับค่าจริง
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


# ---------- etl.py ----------

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
    # normalize ย้ายไปอยู่ train_model.py แล้ว (fit จาก train เท่านั้น กัน data leakage --
    # ดู train_model.py::fit_min_max / apply_min_max)
    df = etl.fix_data_errors(raw_df.copy())
    df = etl.handle_missing(df)
    scale_info = train_model.fit_min_max(df, ["cpu_usage", "memory_usage"])
    df = train_model.apply_min_max(df, ["cpu_usage", "memory_usage"], scale_info)
    assert df["cpu_usage_norm"].between(0, 1).all()
    assert df["memory_usage_norm"].between(0, 1).all()
    assert "cpu_usage" in scale_info and "min" in scale_info["cpu_usage"]


def test_apply_min_max_clips_out_of_range_values():
    # REGRESSION TEST: ค่าที่หลุดช่วง [min, max] ที่ fit ไว้ (เช่น test set มี anomaly
    # แรงกว่าที่เคยเห็นตอน fit จาก train) ต้องถูก clip ไว้ที่ 0-1 ไม่ใช่ติดลบ/เกิน 1
    train_part = pd.DataFrame({"cpu_usage": [10.0, 20.0, 30.0]})
    scale_info = train_model.fit_min_max(train_part, ["cpu_usage"])
    unseen_part = pd.DataFrame({"cpu_usage": [-5.0, 100.0]})  # นอกช่วง [10, 30] ที่ fit ไว้
    result = train_model.apply_min_max(unseen_part, ["cpu_usage"], scale_info)
    assert result["cpu_usage_norm"].between(0, 1).all()


def test_trend_feature_is_causal_not_using_future_values():
    # REGRESSION TEST: ต้องมั่นใจว่า response_time_dev_from_roll ไม่แอบใช้ค่าจาก "อนาคต"
    # (ไม่งั้นจะกลายเป็น data leakage ตอนเทรน/ประเมินผล)
    df = pd.DataFrame({
        "server_id": ["SRV-001"] * 20,
        "response_time": [50.0] * 15 + [500.0] * 5,  # ค่าพุ่งช่วงท้าย
    })
    result_before_spike = etl.add_trend_feature(df.copy(), window=5)
    # แก้ไขค่าช่วง "อนาคต" (index 18) แล้วเช็คว่า dev_from_roll ของแถวก่อนหน้า (index 14) ไม่เปลี่ยน
    df2 = df.copy()
    df2.loc[18, "response_time"] = 9999.0
    result_after_edit = etl.add_trend_feature(df2, window=5)
    assert result_before_spike.loc[14, "response_time_dev_from_roll"] == \
        result_after_edit.loc[14, "response_time_dev_from_roll"]


# ---------- train_model.py ----------

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
