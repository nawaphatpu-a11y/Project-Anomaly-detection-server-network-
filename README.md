# ระบบตรวจจับความผิดปกติของเซิร์ฟเวอร์/เครือข่าย (Anomaly Detection Prototype)

## 1) ติดตั้ง

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## 2) รัน pipeline offline

```bash
python generate_data.py          # -> data/raw/metrics.csv
python etl.py                    # -> data/processed/metrics_clean.csv
python train_model.py            # -> predictions.csv, models/isolation_forest.joblib, scale_params.json
python evaluate.py               # -> data/processed/evaluation_report.csv + evaluation_scatter.png
```

`train_model.py` แบ่ง train/test ตามเวลา (fit min-max/mean-std จาก train เท่านั้นตอน
ประเมินผล กัน data leakage) แล้วเทรน Isolation Forest ตัวสุดท้ายจากข้อมูล**ทั้งหมด**
แยกอีกรอบสำหรับ deploy จริง benchmark เทียบ 4 วิธี: Isolation Forest, LOF, Z-score, IQR

**ไฟล์ที่ commit ไว้ใน repo แล้ว (ไม่ต้องรันก็เปิด dashboard ได้ทันที):**
`models/isolation_forest.joblib`, `data/processed/scale_params.json`,
`data/processed/evaluation_report*.csv`, `data/processed/evaluation_scatter*.png`,
`independent_test_set_v2.csv` — dashboard จึงข้ามขั้น bootstrap ตอน container ตื่น
(ลดเวลารอ) ส่วนข้อมูลดิบ/ไฟล์ขั้นกลางขนาดใหญ่ถูก `.gitignore` ไว้ สร้างใหม่ได้ด้วยคำสั่งข้างบน
**ถ้าแก้โมเดล/ข้อมูลแล้ว ต้อง commit ไฟล์พวกนี้ใหม่ด้วย** ไม่งั้นเว็บยังใช้ของเก่าอยู่

## 3) ประเมินผลกับชุดข้อมูลอิสระ (Step 5 เสริม — สร้างจาก AI คนละตัว)

```bash
python Generate_data_AI.py                                    # -> independent_test_set_v2.csv
python evaluate_independent.py --infile independent_test_set_v2.csv
python retrain_with_independent.py --independent-infile independent_test_set_v2.csv   # ทดลอง retrain
```

ข้อสังเกตสำคัญที่เจอจริง: precision ร่วงจาก ~0.86 เหลือ ~0.10 บนชุดข้อมูลอิสระ ทั้งที่
recall ยังสูง (~0.95) เพราะ baseline ของข้อมูล "ปกติ" ต่างจากที่โมเดลเคยเห็น (covariate
shift) และการ retrain ด้วยข้อมูลผสมแบบตรงไปตรงมาแทบไม่ช่วย (ข้อมูลใหม่เป็นแค่ ~3.9% ของ
training set) — `retrain_with_independent.py` เซฟโมเดลใหม่แยกไว้ ไม่ทับโมเดล deploy

## 4) เปิด live dashboard

```bash
streamlit run app/Home.py
```

รันจาก root ของโปรเจกต์เท่านั้น มี 3 หน้าใน sidebar: **Home** (dashboard), **🔔 Alerts**
(หน้าแจ้งเตือน เรียงตามความรุนแรง กรองได้ ดาวน์โหลด CSV ได้) และ **📈 Model Evaluation**
(ผลประเมินโมเดล) ข้อมูล live เป็นข้อมูลจำลอง สคอร์ด้วย Isolation Forest ตัวเดียวกับที่ประเมินผลไว้

ถ้าไม่มีไฟล์โมเดล แอปจะรัน pipeline offline ให้เองตอนเปิดครั้งแรก (กันพังตอน deploy ใหม่)

## 5) Deploy + กันแอปหลับ (Streamlit Community Cloud)

Deploy จาก GitHub repo นี้ โดยชี้ Main file path ไปที่ `app/Home.py`
Streamlit พักแอปฟรีเองหลังไม่มี traffic จริง 12 ชม. — ตั้ง keepalive กันไว้ได้:

1. GitHub repo → Settings → Secrets and variables → Actions → แท็บ **Variables**
2. New repository variable: Name `STREAMLIT_APP_URL`, Value = ลิงก์แอปจริง
3. workflow `.github/workflows/keepalive.yml` จะเปิดเว็บด้วยเบราว์เซอร์จริง (Playwright)
   ทุก 4 ชม. และกดปุ่มปลุกให้ถ้าแอปหลับ (HTTP ping ธรรมดาใช้ไม่ได้ผล) — กดรันเองได้ที่
   แท็บ Actions → Streamlit Keepalive → Run workflow

## 6) รันเทสต์

```bash
pytest tests/ -v
```

มี GitHub Actions (`.github/workflows/tests.yml`) รันเทสต์ชุดนี้อัตโนมัติทุกครั้งที่ push
