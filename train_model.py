"""
train_model.py
Train the Anomaly Detection model(s) per Step 4: Modeling, and evaluate on a
held-out TEST split the models never saw during fit (Step 5 groundwork).

# โมเดลหลักที่ใช้ใน Prototype: Isolation Forest
# โมเดลเปรียบเทียบ (Benchmark เท่านั้น ไม่ใช่ Prototype แยก): LOF, Z-score

# ทำไมต้องแบ่ง train/test:
# ถ้า fit และวัดผลบนข้อมูลชุดเดียวกันทั้งหมด ตัวเลข precision/recall ที่ได้จะไม่บอกว่า
# โมเดลจะจับความผิดปกติ "ใหม่" ที่ไม่เคยเห็นได้ดีแค่ไหน จึงต้องกันข้อมูลส่วนหนึ่งไว้ทดสอบ
#
# วิธีแบ่ง: ตามเวลา (time-based) แยกทีละ server -- เอาข้อมูล "เก่า" ของแต่ละเครื่องไป train
# แล้วเอา "ใหม่สุด" ไป test เท่านั้น ห้ามแบ่งแบบสุ่มแถว เพราะ True Anomaly ถูกแทรกเป็นช่วง
# ต่อเนื่อง (6-20 แถวติดกัน) การสุ่มแถวจะทำให้ episode เดียวกันหลุดไปอยู่ทั้ง train/test
# กลายเป็น data leakage ทำให้ผลวัดดีเกินจริง

Usage:
    python train_model.py --infile data/processed/metrics_clean.csv --outfile data/processed/predictions.csv
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# metric ดิบที่ต้อง normalize -- ย้ายมาจาก etl.py เพื่อควบคุมว่า fit จากข้อมูลชุดไหน
# (train เท่านั้นสำหรับประเมินผล / ข้อมูลทั้งหมดสำหรับโมเดลที่ deploy จริง -- ดู main())
NORMALIZE_COLS = [
    "cpu_usage", "memory_usage", "network_in", "network_out",
    "response_time", "network_total", "cpu_memory_ratio",
    "response_time_dev_from_roll",
]

# feature หลักที่ทุกโมเดลใช้ร่วมกัน -- เวอร์ชัน normalize แล้วสำหรับ metric ปกติ ส่วน
# hour/day_of_week ตั้งใจปล่อยเป็นค่าดิบ (ทดสอบแล้วว่าให้ LOF แม่นกว่า normalize)
BASE_FEATURE_COLS = [
    "cpu_usage_norm", "memory_usage_norm", "network_in_norm", "network_out_norm",
    "response_time_norm", "network_total_norm", "cpu_memory_ratio_norm",
    "hour", "day_of_week", "is_peak_hour",
]

# feature เสริมสำหรับจับ RESPONSE_DEGRADATION (ดูรายละเอียดใน etl.py::add_trend_feature)
# ทดสอบพบว่าแต่ละโมเดลชอบ scale ไม่เหมือนกัน:
#   - Isolation Forest / Z-score: ใช้เวอร์ชันดิบ -> F1 ดีขึ้น (0.80 -> 0.88 สำหรับ IsoForest)
#   - LOF (ตัดสินด้วยระยะทาง): ต้องใช้เวอร์ชัน normalize ไม่งั้น F1 พังจาก 0.87 เหลือ 0.40
#     (scale ใหญ่ของค่าดิบไปครอบงำระยะทางแบบเดียวกับปัญหา hour/day_of_week ก่อนหน้านี้)
TREND_COL_RAW = "response_time_dev_from_roll"
TREND_COL_NORM = "response_time_dev_from_roll_norm"

ISO_FEATURE_COLS = BASE_FEATURE_COLS + [TREND_COL_RAW]
LOF_FEATURE_COLS = BASE_FEATURE_COLS + [TREND_COL_NORM]
ZSCORE_FEATURE_COLS = BASE_FEATURE_COLS + [TREND_COL_RAW]  # z-score standardize เองในตัวอยู่แล้ว scale ไม่มีผล

# รวม column ทั้งหมดที่ใช้จริง (สำหรับเซฟไปกับ output ให้ตรวจสอบย้อนหลังได้)
ALL_FEATURE_COLS = BASE_FEATURE_COLS + [TREND_COL_RAW, TREND_COL_NORM]


def fit_min_max(source_df: pd.DataFrame, cols: list) -> dict:
    # หา min/max จาก source_df ที่ส่งเข้ามา -- ผู้เรียกเป็นคนตัดสินใจว่าจะส่ง train
    # เท่านั้น (สำหรับประเมินผลอย่างเป็นธรรม) หรือข้อมูลทั้งหมด (สำหรับโมเดล deploy จริง)
    scale_info = {}
    for col in cols:
        col_min, col_max = source_df[col].min(), source_df[col].max()
        scale_info[col] = {"min": float(col_min), "max": float(col_max)}
    return scale_info


def apply_min_max(df: pd.DataFrame, cols: list, scale_info: dict) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        col_min, col_max = scale_info[col]["min"], scale_info[col]["max"]
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"{col}_norm"] = 0.0
        # ค่าที่ส่งเข้ามาอาจหลุดช่วง [min, max] ที่ fit ไว้ได้ (เช่น test set มี anomaly
        # แรงกว่าที่เคยเห็นตอน fit หรือ live reading ใหม่ที่แรงกว่าประวัติทั้งหมด) clip
        # ไว้ที่ 0-1 กันไม่ให้ค่า norm ติดลบ/เกิน 1 จนโมเดลสับสน
        df[f"{col}_norm"] = df[f"{col}_norm"].clip(0.0, 1.0)
    return df


def time_based_split(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """แบ่ง train/test ตามเวลา แยกทีละ server_id เพื่อกัน data leakage ข้าม episode"""
    df = df.sort_values(["server_id", "timestamp"])
    train_parts, test_parts = [], []
    for _, group in df.groupby("server_id", sort=False):
        cut = int(len(group) * (1 - test_ratio))
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)


def train_isolation_forest(X_train: pd.DataFrame, X_test: pd.DataFrame,
                            contamination: float, seed: int) -> np.ndarray:
    # fit บน train เท่านั้น แล้ว predict บน test ที่โมเดลไม่เคยเห็น (inductive โดยธรรมชาติ)
    model = IsolationForest(contamination=contamination, n_estimators=200,
                             random_state=seed, n_jobs=-1)
    model.fit(X_train)
    pred = model.predict(X_test)  # -1 = anomaly, 1 = normal
    return (pred == -1).astype(int)


def train_lof(X_train: pd.DataFrame, X_test: pd.DataFrame, contamination: float) -> np.ndarray:
    # ค่าเริ่มต้นของ LOF (novelty=False) รองรับแค่ fit_predict บนชุดที่ fit เท่านั้น
    # ต้องเปิด novelty=True ถึงจะ fit(X_train) แล้ว predict(X_test) แยกกันได้จริง
    # แปลงเป็น numpy array ก่อนส่งเข้า fit/predict -- กัน sklearn UserWarning เรื่อง
    # feature names ไม่ตรงกัน (cosmetic bug ของ sklearn ไม่กระทบผลลัพธ์ แต่กันไว้ให้ log สะอาด)
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination,
                                novelty=True, n_jobs=-1)
    model.fit(X_train.to_numpy())
    pred = model.predict(X_test.to_numpy())
    return (pred == -1).astype(int)


def zscore_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 3.0) -> np.ndarray:
    # คำนวณ mean/std จาก train เท่านั้น แล้วเอาไปวัด test -- ห้ามคำนวณ mean/std จาก test
    mean, std = X_train.mean(), X_train.std(ddof=0).replace(0, 1)  # กัน std=0 หาร 0
    z = (X_test - mean) / std
    return (z.abs() > threshold).any(axis=1).astype(int)


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection models with a held-out test split")
    parser.add_argument("--infile", type=str, default="data/processed/metrics_clean.csv")
    parser.add_argument("--outfile", type=str, default="data/processed/predictions.csv")
    parser.add_argument("--contamination", type=float, default=0.03)
    parser.add_argument("--test-ratio", type=float, default=0.2,
                         help="สัดส่วนข้อมูล 'ใหม่สุด' ต่อ server ที่กันไว้เป็น test (ไม่ใช้ตอน fit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="models/isolation_forest.joblib",
                         help="ที่เก็บโมเดล Isolation Forest ตัวสุดท้าย (เทรนจากข้อมูลทั้งหมด) สำหรับ live scoring")
    args = parser.parse_args()

    print(f"Loading: {args.infile}")
    df = pd.read_csv(args.infile)

    # คำเตือน: day_of_week ที่ปล่อยเป็นค่าดิบ (0-6) จะกลายเป็นตัวการทำให้ LOF พังได้ ถ้า
    # จำลองข้อมูลสั้นเกินไป (--days ใน generate_data.py น้อยกว่า ~2 สัปดาห์) เพราะช่วง
    # test อาจตกวันในสัปดาห์ที่ train ไม่เคยเห็นเลย (ทดสอบจริงที่ --days 5: LOF precision
    # ร่วงเหลือ 0.07) ใช้ --days อย่างน้อย 14-30 วันเสมอ
    if df["timestamp"].pipe(pd.to_datetime).dt.normalize().nunique() < 10:
        print("[warn] ข้อมูลครอบคลุมน้อยกว่า 10 วัน — day_of_week อาจทำให้ LOF ประเมินผิดพลาดได้ "
              "แนะนำ generate_data.py --days อย่างน้อย 14 วันขึ้นไป")

    train_df, test_df = time_based_split(df, args.test_ratio)
    print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows (test = {args.test_ratio:.0%} ล่าสุดของแต่ละ server)")

    # FIX (data leakage): normalize สำหรับ "ประเมินผล" ต้อง fit จาก train เท่านั้น --
    # ไม่ใช่จากข้อมูลทั้งหมดเหมือนที่ etl.py เคยทำ (ซึ่งทำให้ min/max ของ test รั่วเข้าไป
    # ตอนเทรน) หลักการเดียวกับที่ zscore_baseline() ทำกับ mean/std อยู่แล้ว
    eval_scale_info = fit_min_max(train_df, NORMALIZE_COLS)
    train_df = apply_min_max(train_df, NORMALIZE_COLS, eval_scale_info)
    test_df = apply_min_max(test_df, NORMALIZE_COLS, eval_scale_info)

    print("Training Isolation Forest (main prototype model)...")
    Xtr, Xte = train_df[ISO_FEATURE_COLS].fillna(0), test_df[ISO_FEATURE_COLS].fillna(0)
    test_df["pred_isoforest"] = train_isolation_forest(Xtr, Xte, args.contamination, args.seed)

    print("Training LOF (benchmark only)...")
    Xtr, Xte = train_df[LOF_FEATURE_COLS].fillna(0), test_df[LOF_FEATURE_COLS].fillna(0)
    test_df["pred_lof"] = train_lof(Xtr, Xte, args.contamination)

    print("Computing Z-score baseline (benchmark only)...")
    Xtr, Xte = train_df[ZSCORE_FEATURE_COLS].fillna(0), test_df[ZSCORE_FEATURE_COLS].fillna(0)
    test_df["pred_zscore"] = zscore_baseline(Xtr, Xte)

    keep_cols = ["server_id", "timestamp", "is_anomaly", "anomaly_type",
                 "pred_isoforest", "pred_lof", "pred_zscore"] + ALL_FEATURE_COLS

    out_dir = os.path.dirname(args.outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # NOTE: predictions.csv เก็บเฉพาะแถวของ "test set" เท่านั้น (ข้อมูลที่โมเดลไม่เคยเห็น
    # ตอน fit) evaluate.py จึงวัดผลบน out-of-sample data โดยอัตโนมัติ
    test_df[keep_cols].to_csv(args.outfile, index=False)

    print(f"Saved predictions (test set only) -> {args.outfile}")
    for col in ["pred_isoforest", "pred_lof", "pred_zscore"]:
        print(f"  {col}: flagged {int(test_df[col].sum()):,} rows as anomaly (out of {len(test_df):,} test rows)")

    # ---------- เทรนโมเดลตัวสุดท้ายสำหรับใช้งานจริง (live scoring) ----------
    # ตัวที่ประเมินผลข้างบน (fit บน train 80%) มีไว้เพื่อ "วัดผล" เท่านั้น
    # โมเดลที่จะเอาไป deploy ควรเทรนจากข้อมูลทั้งหมดที่มี (train+test รวมกัน) เพื่อให้ได้
    # ประโยชน์จากข้อมูลครบที่สุด -- นี่คือโมเดลที่ dashboard (Streamlit) จะโหลดไปใช้สคอร์
    # ข้อมูล live จริง จึงเซฟเฉพาะ Isolation Forest (โมเดลหลักของ prototype) ไม่รวม
    # LOF/Z-score เพราะสองตัวนั้นมีไว้ทำ benchmark เทียบผลเท่านั้นตามที่ระบุไว้ในแผน
    #
    # หมายเหตุสำคัญ: ต่างจาก normalize ตอนประเมินผลด้านบน (ที่ fit จาก train เท่านั้นเพื่อ
    # กัน leakage) โมเดลที่ deploy จริงนี้ "ควร" fit min/max จากข้อมูลทั้งหมด เพราะ
    # ไม่มีแนวคิด train/test ตอน deploy แล้ว -- ต้องการ range ที่ครอบคลุมข้อมูลในอดีต
    # ทั้งหมดเพื่อ scoring ข้อมูล live ใหม่ในอนาคตได้แม่นยำที่สุด
    print("Training final Isolation Forest on ALL data (for live deployment)...")
    deploy_scale_info = fit_min_max(df, NORMALIZE_COLS)
    df = apply_min_max(df, NORMALIZE_COLS, deploy_scale_info)

    X_all = df[ISO_FEATURE_COLS].fillna(0)
    final_model = IsolationForest(contamination=args.contamination, n_estimators=200,
                                   random_state=args.seed, n_jobs=-1)
    final_model.fit(X_all)

    model_dir = os.path.dirname(args.model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    joblib.dump({
        "model": final_model,
        "feature_cols": ISO_FEATURE_COLS,
        "trend_window": 12,          # ต้องตรงกับ window ใน etl.py::add_trend_feature
        "server_ids": sorted(df["server_id"].unique().tolist()),
    }, args.model_out)
    print(f"Saved deployment model -> {args.model_out}")

    # เซฟ scale_info ที่ fit จากข้อมูลทั้งหมด (ตัวเดียวกับที่ใช้เทรน final_model ด้านบน)
    # ให้ app/utils.py เอาไป normalize ข้อมูล live ตอน scoring จริง -- ไปไว้โฟลเดอร์
    # เดียวกับ predictions.csv (data/processed/) ตามที่ app/utils.py คาดหวังไว้
    scale_path = os.path.join(out_dir or ".", "scale_params.json")
    with open(scale_path, "w", encoding="utf-8") as f:
        json.dump(deploy_scale_info, f, ensure_ascii=False, indent=2)
    print(f"Saved scale params (full-data, for live scoring) -> {scale_path}")


if __name__ == "__main__":
    main()
