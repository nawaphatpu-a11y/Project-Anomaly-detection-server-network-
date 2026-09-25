# ระบบตรวจจับความผิดปกติของเซิร์ฟเวอร์/เครือข่าย (Anomaly Detection Prototype)

## 1) ติดตั้ง

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## 2) รัน pipeline offline (ต้องรันก่อนเปิด dashboard ครั้งแรก)

```bash
python generate_data.py          # -> data/raw/metrics.csv
python etl.py                    # -> data/processed/metrics_clean.csv
python train_model.py            # -> predictions.csv, models/isolation_forest.joblib, scale_params.json
python evaluate.py               # -> data/processed/evaluation_report.csv
```

`train_model.py` แบ่ง train/test ตามเวลา (fit min-max/mean-std จาก train เท่านั้นตอน
ประเมินผล กัน data leakage) แล้วเทรนโมเดล Isolation Forest ตัวสุดท้ายจากข้อมูล**ทั้งหมด**
แยกอีกรอบสำหรับ deploy จริง เซฟไว้ที่ `models/isolation_forest.joblib` พร้อม
`data/processed/scale_params.json` — สองไฟล์นี้คือสิ่งที่ dashboard ต้องใช้

## 3) ประเมินผลกับชุดข้อมูลอิสระ (Step 5 เสริม — สร้างจาก AI คนละตัว)

```bash
python Generate_data_AI.py                                    # -> independent_test_set_v2.csv
python evaluate_independent.py --infile independent_test_set_v2.csv
```

ยืนยันว่าโมเดล generalize ข้าม "แหล่งข้อมูล" ได้จริงแค่ไหน (ไม่ใช่แค่จำ pattern เฉพาะ
ของข้อมูลชุดเดิม) — **ข้อสังเกตสำคัญที่เจอจริง**: precision ร่วงจาก ~0.86 เหลือ ~0.10 บน
ชุดข้อมูลอิสระ ทั้งที่ recall ยังสูงอยู่ (~0.95) เพราะข้อมูล "ปกติ" ของอีก AI มี scale
(baseline ของ memory/network/response) ต่างจากที่โมเดลเคยเห็นตอนเทรนมาก (covariate
shift) — คุ้มเอาไปเขียนในหัวข้อ Discussion/Limitation ของรายงาน

## 4) เปิด live dashboard

```bash
streamlit run app/Home.py
```

รันจาก root ของโปรเจกต์เท่านั้น เปิดแล้วจะมี 2 หน้าใน sidebar: **Home** (dashboard) และ
**🔔 Alerts** (หน้าแจ้งเตือนแยกต่างหาก) ข้อมูล live เป็นข้อมูลจำลอง สคอร์ด้วยโมเดล
Isolation Forest ตัวเดียวกับที่ประเมินผลไว้ข้างบน

หน้า dashboard จะรัน pipeline offline (ข้อ 2) ให้อัตโนมัติเองถ้ายังไม่มีโมเดล — สำคัญ
สำหรับตอน deploy บน Streamlit Community Cloud ซึ่งไม่รันสคริปต์ offline ให้เองก่อน

## 5) รันเทสต์

```bash
pytest tests/ -v
```

มี GitHub Actions (`.github/workflows/tests.yml`) รันเทสต์ชุดนี้อัตโนมัติทุกครั้งที่ push
