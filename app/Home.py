"""
app/Home.py
หน้า Dashboard หลัก — รัน: streamlit run app/Home.py (รันจาก root ของโปรเจกต์
เพื่อให้ path models/ และ data/processed/ ถูกต้อง)

ก่อนรันหน้านี้ครั้งแรก ต้องมีไฟล์โมเดลก่อน:
    python generate_data.py
    python etl.py
    python train_model.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from utils import init_buffer, tick

st.set_page_config(page_title="Server & Network Health", page_icon="📊", layout="wide")
init_buffer()

st.sidebar.header("การตั้งค่า")
refresh_seconds = st.sidebar.slider("⏱️ อัปเดตข้อมูลทุก", 2, 15, 4, format="%d วินาที")
rows_per_tick = st.sidebar.slider("📊 จำนวน reading ใหม่ต่อรอบ", 1, 20, 5, format="%d แถว")
# ใส่หน่วย (วินาที/แถว) ต่อท้ายตัวเลขบนแท่งเลื่อนเลย + สรุปเป็นประโยคอ่านง่ายด้านล่าง
# เพราะทดสอบใช้งานจริงแล้วพบว่าตัวเลขเปล่าๆ บนแท่งเลื่อน 2 อันแยกแยะยากด้วยตาเปล่า
st.sidebar.caption(f"ระบบจะสุ่มข้อมูลใหม่ {rows_per_tick} แถว ทุกๆ {refresh_seconds} วินาที")
if st.sidebar.button("ล้างข้อมูลทั้งหมด"):
    st.session_state.readings = pd.DataFrame()
    st.session_state.alerts = pd.DataFrame()
    st.session_state.resp_history = {}

st_autorefresh(interval=refresh_seconds * 1000, key="dashboard_autorefresh")
new_readings = tick(n_new=rows_per_tick)

df = st.session_state.readings

st.title("📊 Server & Network Health Dashboard")
st.caption("ข้อมูล live เป็นข้อมูลจำลอง สคอร์ด้วย Isolation Forest ที่เทรนจริงจาก train_model.py")

if df.empty:
    st.info("กำลังรอข้อมูล live รอบแรก...")
    st.stop()

total_readings = len(df)
total_anomalies = int(df["is_anomaly"].sum())
health_score = 100 * (1 - total_anomalies / total_readings) if total_readings else 100
servers_online = df["server_id"].nunique()

c1, c2, c3 = st.columns(3)
c1.metric("System Health Score", f"{health_score:.1f}%")
c2.metric("Total Anomalies (session)", f"{total_anomalies}")
c3.metric("เครื่องที่ Online", f"{servers_online}")

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("แนวโน้ม CPU / Memory เฉลี่ย")
    trend = df.groupby(df.index // max(1, rows_per_tick)).agg(
        cpu_usage=("cpu_usage", "mean"),
        memory_usage=("memory_usage", "mean"),
    ).reset_index()
    fig = px.line(trend, y=["cpu_usage", "memory_usage"], labels={"value": "%", "index": "รอบข้อมูล (เก่า→ใหม่)"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("ดูว่าค่าเฉลี่ยทั้งฟลีทกำลังไต่ขึ้นต่อเนื่องไหม (สัญญาณเตือนล่วงหน้าก่อนเกิด anomaly จริง) เส้นแบนคือระบบนิ่งดี")

with col_b:
    st.subheader("แนวโน้มจำนวน Anomaly")
    anomaly_trend = df.groupby(df.index // max(1, rows_per_tick))["is_anomaly"].sum().reset_index()
    fig = px.line(anomaly_trend, x="index", y="is_anomaly", labels={"index": "รอบข้อมูล (เก่า→ใหม่)", "is_anomaly": "จำนวน anomaly"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("นับจำนวน anomaly ต่อรอบ ถ้าเส้นนี้ไต่ขึ้นเรื่อยๆ (ไม่ใช่แค่กระตุกครั้งเดียว) แปลว่าอาจมีปัญหาลุกลามในระบบ ควรเข้าไปดูหน้า Alerts")

col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Anomaly แยกตามเครื่อง")
    by_server = df[df["is_anomaly"] == 1]["server_id"].value_counts().reset_index()
    by_server.columns = ["server_id", "count"]
    if not by_server.empty:
        st.plotly_chart(px.bar(by_server, x="server_id", y="count"), use_container_width=True)
        st.caption("เครื่องที่แท่งสูงสุด = เครื่องที่ควรตรวจสอบก่อน ถ้าเครื่องเดิมขึ้นสูงซ้ำๆ ทุกครั้งที่เช็ค อาจเป็นปัญหาเฉพาะเครื่องนั้น ไม่ใช่ปัญหาทั้งระบบ")
    else:
        st.info("ยังไม่มี anomaly เกิดขึ้นในเซสชันนี้")

with col_d:
    st.subheader("Anomaly แยกตามประเภท")
    by_type = df[df["is_anomaly"] == 1]["alert_type"].value_counts().reset_index()
    by_type.columns = ["alert_type", "count"]
    if not by_type.empty:
        st.plotly_chart(px.bar(by_type, x="alert_type", y="count"), use_container_width=True)
        st.caption("บอกว่าปัญหาที่เจอบ่อยสุดเป็นประเภทไหน (CPU พุ่ง / Network พุ่ง / Response ช้าลง) ช่วยเลือกว่าจะไปสืบต่อทางไหนก่อน")
    else:
        st.info("ยังไม่มี anomaly เกิดขึ้นในเซสชันนี้")

col_e, col_f = st.columns(2)
with col_e:
    st.subheader("CPU vs Response Time (ไฮไลต์ anomaly)")
    fig = px.scatter(df, x="cpu_usage", y="response_time",
                      color=df["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
                      color_discrete_map={"Normal": "#4C78A8", "Anomaly": "#E45756"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("จุดสีแดงที่อยู่ห่างจากกลุ่มจุดสีน้ำเงินหนาแน่น = จุดที่โมเดลตัดสินว่าผิดปกติ ยิ่งอยู่ไกลจากกลุ่ม ยิ่งชัดเจนว่าผิดปกติจริง")

with col_f:
    st.subheader("Correlation Matrix")
    num_cols = ["cpu_usage", "memory_usage", "network_in", "network_out", "response_time"]
    corr = df[num_cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                     use_container_width=True)
    st.caption("แผงนี้เป็นแนว EDA (สำหรับนักวิเคราะห์ข้อมูล) มากกว่าใช้เฝ้าระบบจริง — สีแดงเข้ม (ใกล้ 1.00) แปลว่าสอง metric นั้นขึ้นลงพร้อมกันเสมอ เช่น network_in/out ที่ 0.71 ปกติของมันคือไปด้วยกัน")

st.subheader("การกระจายตัวของ Network Traffic")
# FIX: จำกัดช่วงแกน x ไม่ให้ anomaly ที่ network พุ่งแรงมากดึงสเกลจนข้อมูลปกติ (ส่วนใหญ่)
# ถูกบีบเป็นเส้นบางๆ อ่านไม่ออก -- ครอบที่ percentile 95 ของข้อมูลทั้งหมด แล้วบอกจำนวนที่ตกขอบ
p95 = df["network_in"].quantile(0.95)
x_max = max(p95 * 1.3, df["network_in"].quantile(0.5) * 2)
n_beyond = int((df["network_in"] > x_max).sum())
fig = px.histogram(df, x="network_in", nbins=30, range_x=[0, x_max])
st.plotly_chart(fig, use_container_width=True)
caption = "ก้อนสูงคือช่วงค่าปกติที่เกิดบ่อยที่สุด"
if n_beyond:
    caption += f" (ตัดแกนไว้ที่ {x_max:.0f} เพื่อให้เห็นก้อนปกติชัดๆ — มี {n_beyond} แถวที่ network พุ่งเกินขอบนี้ไป ซึ่งคือ anomaly ที่เห็นในหน้า Alerts นั่นเอง)"
st.caption(caption)

st.subheader("Top 10 เหตุการณ์ผิดปกติล่าสุด")
recent_alerts = df[df["is_anomaly"] == 1].sort_values("timestamp", ascending=False).head(10)
if not recent_alerts.empty:
    st.dataframe(
        recent_alerts[["timestamp", "server_id", "alert_type", "cpu_usage", "memory_usage",
                        "response_time", "anomaly_score"]],
        use_container_width=True, hide_index=True,
    )
else:
    st.success("ยังไม่มีเหตุการณ์ผิดปกติในเซสชันนี้ ✅")
