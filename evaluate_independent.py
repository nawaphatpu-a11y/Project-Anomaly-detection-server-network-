"""
evaluate_independent.py
ประเมินผลโมเดลที่เทรน+เซฟไว้แล้ว (models/isolation_forest.joblib) กับชุดข้อมูลทดสอบ
"อิสระ" ที่ generate จาก AI คนละตัว (เช่น ChatGPT) ตามที่โจทย์ระบุไว้ใน Step 5:
"generate ชุดข้อมูลทดสอบด้วย AI อีกตัวเพื่อประเมินผล"

ต่างจาก evaluate.py (ที่วัดผลบน test split จากข้อมูลชุดเดียวกับตอนเทรน) -- ไฟล์นี้ใช้
ข้อมูลที่มาจากคนละแหล่ง คนละตรรกะการสุ่มเลย จึงเป็นการยืนยันว่าโมเดล generalize ได้จริง
ไม่ใช่แค่จำ pattern เฉพาะของข้อมูลชุดเดิม

ใช้ฟังก์ชันทำความสะอาดชุดเดียวกับ etl.py (import ตรงๆ) เพื่อให้แน่ใจว่า feature ที่ป้อน
เข้าโมเดลคำนวณด้วยตรรกะเดียวกับตอนเทรนเป๊ะ และใช้ scale_params.json (fit จากข้อมูล
ทั้งหมดตอนเทรน) แปลง normalize แทนการ fit ใหม่จากชุดนี้ -- ห้าม fit ใหม่ เพราะโมเดล
ต้องเห็นข้อมูลผ่าน "เลนส์" เดียวกับตอนเทรนเท่านั้น ไม่งั้นผลเปรียบเทียบจะไม่ยุติธรรม

Usage:
    python evaluate_independent.py --infile independent_test_set.csv
"""

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from etl import fix_data_errors, handle_missing, add_time_features, add_engineered_features, add_trend_feature

MODEL_PATH = "models/isolation_forest.joblib"
SCALE_PARAMS_PATH = "data/processed/scale_params.json"


def apply_min_max(df: pd.DataFrame, cols: list, scale_info: dict) -> pd.DataFrame:
    # เหมือนกับ train_model.py::apply_min_max เป๊ะ -- ใช้ scale_info ที่ fit ไว้แล้ว
    # ("เลนส์" เดียวกับตอนเทรน) ไม่ fit ใหม่จากชุดข้อมูลนี้
    df = df.copy()
    for col in cols:
        col_min, col_max = scale_info[col]["min"], scale_info[col]["max"]
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"{col}_norm"] = 0.0
        df[f"{col}_norm"] = df[f"{col}_norm"].clip(0.0, 1.0)
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved model on an independently-generated test set")
    parser.add_argument("--infile", type=str, default="independent_test_set.csv")
    parser.add_argument("--outfile", type=str, default="data/processed/evaluation_report_independent.csv")
    args = parser.parse_args()

    print(f"Loading independent test set: {args.infile}")
    df = pd.read_csv(args.infile)
    print(f"Loaded: {len(df):,} rows")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    window = bundle["trend_window"]

    with open(SCALE_PARAMS_PATH, "r", encoding="utf-8") as f:
        scale_info = json.load(f)

    # ---------- ทำความสะอาด + สร้าง feature ด้วยตรรกะเดียวกับตอนเทรน (import จาก etl.py) ----------
    df = fix_data_errors(df)
    df = handle_missing(df)
    df = add_time_features(df)
    df = add_engineered_features(df)
    df = add_trend_feature(df, window=window)  # ใช้ window เดียวกับตอนเทรน (จาก bundle)

    normalize_cols = list(scale_info.keys())
    df = apply_min_max(df, normalize_cols, scale_info)

    X = df[feature_cols].fillna(0)

    # ---------- Predict ----------
    pred = model.predict(X)                 # -1 = anomaly, 1 = normal
    df["pred_isoforest"] = (pred == -1).astype(int)

    # ---------- Evaluate เทียบกับ is_anomaly ที่ ChatGPT ใส่มาให้ ----------
    y_true = df["is_anomaly"]
    y_pred = df["pred_isoforest"]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    report = pd.DataFrame([{
        "dataset": "Independent test set (ChatGPT-generated)",
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
    }]).set_index("dataset")

    print(report.to_string())

    out_dir = os.path.dirname(args.outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    report.to_csv(args.outfile)
    print(f"\nSaved -> {args.outfile}")


if __name__ == "__main__":
    main()
