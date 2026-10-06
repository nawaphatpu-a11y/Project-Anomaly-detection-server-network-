"""
app/pages/2_📈_Model_Evaluation.py
แสดงผลการประเมินโมเดลแบบ offline (จาก evaluate.py และ evaluate_independent.py) ในตัว
dashboard เอง เพื่อให้ผู้ดู (เช่นอาจารย์) เห็นความแม่นยำของโมเดลได้โดยไม่ต้องเปิดไฟล์
รายงานแยก -- หน้านี้แค่ "อ่าน" ไฟล์ CSV ที่สองสคริปต์นั้นสร้างไว้แล้ว ไม่ได้รันโมเดลใหม่
และไม่เกี่ยวกับข้อมูล live ในหน้า Home/Alerts เลย

หมายเหตุสำคัญ: ไฟล์ CSV ที่หน้านี้อ่าน ถูกสร้างโดย app/utils.py::_ensure_pipeline_has_run()
ไปแล้วตอนเปิดแอปครั้งแรก (รัน evaluate.py ให้ล่วงหน้าด้วย) ถ้าไม่มี ensure ตรงนี้
หน้านี้จะว่างตลอดไปบน deploy ใหม่ เพราะไม่มีอะไรไปรัน evaluate.py ให้บนเซิร์ฟเวอร์เลย
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import ensure_evaluation_has_run

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_PATH = PROJECT_ROOT / "data/processed/evaluation_report.csv"
# ใช้ "_after" ถ้ามี (ผลหลัง retrain ด้วยข้อมูลรวม) ไม่งั้น fallback ไปไฟล์ก่อน retrain
EVAL_INDEP_PATHS = [
    PROJECT_ROOT / "data/processed/evaluation_report_independent_after.csv",
    PROJECT_ROOT / "data/processed/evaluation_report_independent.csv",
]

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")
st.title("📈 Model Evaluation")
st.caption("ผลประเมินแบบ offline จาก evaluate.py / evaluate_independent.py — เป็นข้อมูลนิ่ง ไม่ใช่ live")

ensure_evaluation_has_run()  # รัน evaluate.py (+independent) ให้เองถ้ายังไม่มีไฟล์ -- กันหน้านี้ว่างตลอดไปตอน deploy ใหม่


def _show_metric_bars(df: pd.DataFrame, title: str):
    melted = df.reset_index().melt(id_vars=df.index.name or "index",
                                     value_vars=["precision", "recall", "f1"],
                                     var_name="metric", value_name="score")
    fig = px.bar(melted, x=melted.columns[0], y="score", color="metric", barmode="group",
                 range_y=[0, 1], title=title)
    st.plotly_chart(fig, use_container_width=True)


# ---------- ส่วนที่ 1: ประเมินผลบน test split (ข้อมูลชุดเดียวกับตอนเทรน) ----------
st.header("1) Test Split — ข้อมูลชุดเดียวกับตอนเทรน")
if EVAL_PATH.exists():
    eval_df = pd.read_csv(EVAL_PATH, index_col=0)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(eval_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
                     use_container_width=True)
    with c2:
        _show_metric_bars(eval_df, "Precision / Recall / F1 — Test Split")
    best_model = eval_df["f1"].idxmax()
    st.success(f"โมเดลที่ F1 ดีที่สุดบน test split: **{best_model}** (F1 = {eval_df.loc[best_model, 'f1']:.3f})")
else:
    st.warning(f"ยังไม่พบ `{EVAL_PATH.relative_to(PROJECT_ROOT)}` — รัน `python evaluate.py` ก่อน")

st.divider()

# ---------- ส่วนที่ 2: ประเมินผลบนชุดข้อมูลอิสระ (AI คนละตัว) ----------
st.header("2) Independent Test Set — ข้อมูลจาก AI คนละตัว (Step 5 ตามโจทย์)")
st.caption("ทดสอบว่าโมเดล generalize ได้จริงแค่ไหน กับข้อมูลที่ไม่ได้มาจากตรรกะการสุ่มเดียวกับตอนเทรน")

indep_path = next((p for p in EVAL_INDEP_PATHS if p.exists()), None)
if indep_path:
    indep_df = pd.read_csv(indep_path, index_col=0)
    c3, c4 = st.columns([2, 3])
    with c3:
        st.dataframe(indep_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
                     use_container_width=True)
    with c4:
        _show_metric_bars(indep_df, "Precision / Recall / F1 — Independent Set")

    if indep_path.name.endswith("_after.csv"):
        st.info("ผลนี้มาจาก**หลัง retrain ด้วยข้อมูลรวม** (metrics.csv + independent_test_set.csv เดิม) "
                "ทดสอบกับชุดข้อมูลอิสระชุดใหม่ที่ไม่เคยเห็น (independent_test_set_v2.csv)")
    else:
        st.info("ผลนี้มาจาก**ก่อน retrain** — เทรนจาก metrics.csv อย่างเดียว")
else:
    st.warning("ยังไม่พบไฟล์ evaluation_report_independent*.csv")

st.divider()
st.caption("หมายเหตุ: หน้านี้อ่านไฟล์ CSV ที่มีอยู่แล้วเท่านั้น ไม่ได้รันโมเดลหรือประเมินผลใหม่ "
           "ถ้าอยากอัปเดตตัวเลข ต้องรันสคริปต์ evaluate.py / evaluate_independent.py ใหม่แล้วรีเฟรชหน้านี้")
