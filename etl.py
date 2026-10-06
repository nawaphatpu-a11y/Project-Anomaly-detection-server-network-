"""
etl.py
Clean the raw metrics data (Extract -> Transform -> Load) per Step 3: Data Preparation

# หลักการสำคัญ (ตามที่วางแผนไว้):
# - Data Error (ค่าที่เป็นไปไม่ได้จริง) -> ต้อง Remove/Replace
# - True Anomaly (is_anomaly=1 ที่ generate_data.py ใส่ไว้) -> ห้ามลบ เก็บไว้เทรนโมเดล
# - Missing value -> เติมด้วย Forward-Fill (เหมาะกับ time-series)
# - Normalize ด้วย Min-Max
# - เพิ่ม time-based feature (hour, day_of_week, is_peak_hour) และ trend feature

Usage:
    python etl.py --infile data/raw/metrics.csv --outfile data/processed/metrics_clean.csv
"""

import argparse
import os
import numpy as np
import pandas as pd

# Dim_AlertType ตามที่ออกแบบไว้ใน Star Schema -- ผูกระดับความรุนแรงเข้ากับชนิด anomaly
# CPU_SPIKE/NETWORK_SURGE พุ่งกะทันหันจนกระทบบริการได้ทันที = High, RESPONSE_DEGRADATION
# ค่อยๆ แย่ลง (ยังพอมีเวลาตอบสนอง) = Medium, "Other" (โมเดลเจอเองไม่ตรงชนิดที่รู้จัก) = Low
# เพราะยังไม่รู้สาเหตุแน่ชัด ต้องดูรายละเอียดเพิ่มก่อนตัดสินความรุนแรงจริง
SEVERITY_MAP = {
    "CPU_SPIKE": "High",
    "NETWORK_SURGE": "High",
    "RESPONSE_DEGRADATION": "Medium",
    "Other": "Low",
    "NONE": "None",
}


def add_severity(df: pd.DataFrame, type_col: str = "anomaly_type") -> pd.DataFrame:
    df["severity_level"] = df[type_col].map(SEVERITY_MAP).fillna("None")
    return df


def fix_data_errors(df: pd.DataFrame) -> pd.DataFrame:
    # แก้ 'ค่าที่เป็นไปไม่ได้จริง' (Data Error) เท่านั้น
    # - cpu_usage / memory_usage ต้องอยู่ในช่วง 0-100
    # - response_time, network_in/out ต้องไม่ติดลบ
    # ค่าที่ผิดเงื่อนไขเหล่านี้ = ความผิดพลาดของระบบเก็บข้อมูล (ไม่ใช่ True Anomaly)
    # จึงแทนที่ด้วย NaN ก่อน แล้วปล่อยให้ handle_missing() จัดการต่อ
    before = len(df)

    invalid_cpu = (df["cpu_usage"] < 0) | (df["cpu_usage"] > 100)
    invalid_mem = (df["memory_usage"] < 0) | (df["memory_usage"] > 100)
    invalid_resp = df["response_time"] < 0
    invalid_net_in = df["network_in"] < 0
    invalid_net_out = df["network_out"] < 0

    n_invalid = (invalid_cpu | invalid_mem | invalid_resp | invalid_net_in | invalid_net_out).sum()

    df.loc[invalid_cpu, "cpu_usage"] = np.nan
    df.loc[invalid_mem, "memory_usage"] = np.nan
    df.loc[invalid_resp, "response_time"] = np.nan
    df.loc[invalid_net_in, "network_in"] = np.nan
    df.loc[invalid_net_out, "network_out"] = np.nan

    print(f"[fix_data_errors] found {n_invalid:,} impossible values (set to NaN) out of {before:,} rows")
    return df


def fix_timestamp_errors(df: pd.DataFrame) -> pd.DataFrame:
    # Data Error อีกชนิดที่แผนงานระบุไว้ (timestamp ผิดรูปแบบ) แต่ยังไม่เคยเช็คมาก่อน
    # errors="coerce" ทำให้ค่าที่ parse ไม่ได้กลายเป็น NaT แทนที่จะโยน error ทั้งไฟล์
    # ต่างจาก missing value เชิงตัวเลข (เติมด้วย forward-fill ได้) timestamp ที่หายไป
    # เติมแทนไม่ได้อย่างมีความหมาย (ไม่รู้ว่าควรเป็นเวลาไหน) จึง Remove ทิ้งไปเลยตามแผน
    before = len(df)
    parsed = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    n_bad = int(parsed.isna().sum())

    df = df.loc[parsed.notna()].copy()
    df["timestamp"] = parsed.loc[parsed.notna()]

    print(f"[fix_timestamp_errors] found {n_bad:,} rows with malformed timestamp -> removed (out of {before:,} rows)")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    # เติม missing value ด้วย Forward-Fill แยกตามแต่ละ server
    # (ใช้ Forward-Fill เพราะเป็นข้อมูล time-series -- ค่าก่อนหน้ามักใกล้เคียงค่าปัจจุบันมากกว่าค่าเฉลี่ยรวม)
    metric_cols = ["cpu_usage", "memory_usage", "network_in", "network_out", "response_time"]
    n_missing_before = df[metric_cols].isna().sum().sum()

    df = df.sort_values(["server_id", "timestamp"])
    df[metric_cols] = df.groupby("server_id")[metric_cols].transform(lambda s: s.ffill().bfill())

    n_missing_after = df[metric_cols].isna().sum().sum()
    print(f"[handle_missing] missing before: {n_missing_before:,} -> after: {n_missing_after:,}")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # เพิ่ม feature เชิงเวลา ตามที่วางแผนไว้ใน Step 3.2 / 4.1
    # timestamp ผ่าน fix_timestamp_errors() มาแล้ว (เป็น datetime64 ที่ parse ได้ครบทุกแถว)
    # เรียก pd.to_datetime ซ้ำตรงนี้อีกชั้นเป็นแค่ safety net เผื่อมีคนเรียกฟังก์ชันนี้ตรงๆ
    # โดยข้าม fix_timestamp_errors ไป (เช่น เขียนเทสต์แยก) -- ไม่มีผลเสียถ้า column เป็น
    # datetime64 อยู่แล้ว pandas จะคืนค่าเดิมกลับมาเฉยๆ
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=จันทร์
    df["is_peak_hour"] = df["hour"].between(9, 17).astype(int)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # feature เสริมตามที่วางแผนไว้ใน Step 4.1
    df["network_total"] = df["network_in"] + df["network_out"]
    df["cpu_memory_ratio"] = df["cpu_usage"] / df["memory_usage"].replace(0, np.nan)
    df["cpu_memory_ratio"] = df["cpu_memory_ratio"].fillna(df["cpu_memory_ratio"].median())
    return df


def add_trend_feature(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    # เพิ่มมาเพื่อจับ RESPONSE_DEGRADATION (ค่อยๆ แย่ลงแบบ ramp) โดยเฉพาะ -- feature
    # แบบ snapshot ที่มีอยู่เดิม (ค่า ณ ขณะนั้น) มองไม่เห็น "แนวโน้ม" เลย โมเดลจึงจับ
    # ช่วงต้นของ ramp ไม่ค่อยได้ (ทดสอบพบ recall ของ Isolation Forest แค่ ~0.34 เฉพาะ
    # anomaly ชนิดนี้ เทียบกับ 1.00 ของ CPU_SPIKE/NETWORK_SURGE ที่ค่าพุ่งกะทันหัน)
    #
    # วิธีคิด: เทียบ response_time ปัจจุบัน กับค่าเฉลี่ยย้อนหลัง `window` จุดก่อนหน้า
    # shift(1) ก่อน rolling กันไม่ให้ค่าปัจจุบันไปปนกับ baseline ของตัวเอง -- เป็น causal
    # feature ล้วนๆ (มองย้อนหลังอย่างเดียว ไม่ใช้อนาคต) ใช้ได้ทั้งตอนเทรนและตอน live
    # scoring จริง เพราะระบบจริงก็มีแค่ประวัติย้อนหลังตอนนั้นเหมือนกัน
    # window=12 ที่ interval 5 นาที = ย้อนหลัง 1 ชั่วโมง
    #
    # NOTE: เคยลองใส่ trend feature (rolling mean + deviation + slope) ให้ทุก metric
    # (cpu/memory/network/response) พร้อมกันหมด (12 feature เพิ่ม) แต่ผลแย่ลงทุกโมเดล
    # (F1 ของ Isolation Forest ตกจาก 0.80 เหลือ 0.64) เพราะ feature ที่ไม่เกี่ยวกับ
    # anomaly ประเภทอื่นกลายเป็น noise เจือจางสัญญาณจริงไป -- ใส่เฉพาะตัวที่ตรงจุด
    # (response_time) ตัวเดียวถึงจะช่วยจริง โดยไม่ทำร้าย CPU_SPIKE/NETWORK_SURGE ที่ดีอยู่แล้ว
    roll_mean = df.groupby("server_id")["response_time"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).mean()
    )
    df["response_time_dev_from_roll"] = (df["response_time"] - roll_mean).fillna(0)
    return df


    # NOTE: Min-Max Normalization ย้ายไปทำใน train_model.py แล้ว (ดู
    # train_model.py::fit_min_max / apply_min_max)


def main():
    parser = argparse.ArgumentParser(description="ETL: clean raw metrics data")
    parser.add_argument("--infile", type=str, default="data/raw/metrics.csv")
    parser.add_argument("--outfile", type=str, default="data/processed/metrics_clean.csv")
    args = parser.parse_args()

    print(f"Loading: {args.infile}")
    df = pd.read_csv(args.infile)
    print(f"Loaded: {len(df):,} rows")

    # ---------- Transform ----------
    df = fix_data_errors(df)          # 1) แก้ Data Error เชิงตัวเลข (คนละส่วนกับ True Anomaly)
    df = fix_timestamp_errors(df)     # 1.1) แก้ Data Error เชิง timestamp (ต้องทำก่อน sort/group ตามเวลา)
    df = handle_missing(df)           # 2) เติม missing value
    df = add_time_features(df)        # 3) เพิ่ม time-based feature
    df = add_engineered_features(df)  # 4) เพิ่ม engineered feature
    df = add_trend_feature(df)        # 4.1) เพิ่ม trend feature (จับ RESPONSE_DEGRADATION)
    df = add_severity(df)             # 4.2) เพิ่ม severity_level (Dim_AlertType)

    # NOTE: Min-Max Normalization ย้ายไปทำใน train_model.py แล้ว (FIX: data leakage --
    # เดิมคำนวณ min/max ตรงนี้จากข้อมูล "ทั้งหมด" รวมส่วนที่จะกลายเป็น test set ใน
    # train_model.py ด้วย ทำให้ค่า min/max ของ test รั่วเข้าไปในตอนประเมินผล ทั้งที่
    # Z-score baseline ตั้งใจคำนวณ mean/std จาก train เท่านั้นอยู่แล้ว -- ไม่สอดคล้องกัน)
    #
    # train_model.py จะ fit min/max จาก train split เท่านั้นสำหรับตอนประเมินผล และ fit
    # จากข้อมูลทั้งหมดแยกอีกรอบสำหรับโมเดลสุดท้ายที่ deploy จริง (ซึ่งควรใช้ full-data
    # range เพื่อครอบคลุม scoring ข้อมูลใหม่ในอนาคต) -- ดู fit_min_max/apply_min_max
    # ใน train_model.py สำหรับรายละเอียด
    # etl.py จึงส่งออกเฉพาะค่า "ดิบ" ที่ยังไม่ normalize เท่านั้น

    # ---------- Load ----------
    out_dir = os.path.dirname(args.outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)  # FIX: กัน FileNotFoundError ถ้ายังไม่มีโฟลเดอร์ปลายทาง

    df.to_csv(args.outfile, index=False)

    print(f"Saved cleaned file -> {args.outfile}")
    print(f"True Anomaly rows preserved: {(df['is_anomaly'] == 1).sum():,} (none removed)")


if __name__ == "__main__":
    main()
