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
python train_model.py            # -> predictions.csv, models/isolation_forest.joblib, scale_params.json, replay_sample.csv
python evaluate.py               # -> evaluation_report.csv, recall_by_anomaly_type.csv, cause_diagnosis_report.csv, evaluation_scatter.png
```

`generate_data.py` ใช้ `--seed` และ `--start` (ค่าเริ่มต้น 2026-01-05) ตรึงไว้ ผลจึงซ้ำได้ทุกครั้งที่รัน

`train_model.py` แบ่ง train/test ตามเวลาแยกทีละ server (fit min-max และ baseline จาก train เท่านั้นตอนประเมินผล)
แล้วเทรน Isolation Forest ตัวสุดท้ายจากข้อมูล**ทั้งหมด**แยกอีกรอบสำหรับ deploy จริง benchmark เทียบ 4 วิธี:
Isolation Forest, LOF, Z-score, IQR ทุกวิธีใช้ชุดฟีเจอร์เดียวกัน (ยกเว้น LOF ที่ใช้ trend แบบ normalize)

**ไฟล์ที่ commit ไว้ใน repo แล้ว (ไม่ต้องรันก็เปิด dashboard ได้ทันที):**
`models/isolation_forest.joblib`, `data/processed/scale_params.json`, `data/processed/replay_sample.csv`,
`data/processed/evaluation_report*.csv`, `recall_by_anomaly_type.csv`, `cause_diagnosis_report.csv`,
`robustness_sweep.csv`, `model_comparison.csv`, `evaluation_scatter*.png`, `independent_test_set_v2.csv`
ส่วนข้อมูลดิบ/ไฟล์ขั้นกลางขนาดใหญ่ถูก `.gitignore` ไว้ สร้างใหม่ได้ด้วยคำสั่งข้างบน
**ถ้าแก้โมเดล/ข้อมูลแล้ว ต้อง commit ไฟล์พวกนี้ใหม่ด้วย** ไม่งั้นเว็บยังใช้ของเก่าอยู่

### Role baseline และ role_z

แต่ละ role มีค่าปกติต่างกัน (Web = network หนัก/CPU-memory เบา, Database = memory สูง/response ช้ากว่า,
App = CPU สูงสุด) `etl.py` คำนวณ `*_role_z` = ห่างจากค่ามัธยฐานของ role ตัวเองกี่ MAD โดย fit จาก train เท่านั้น
และเก็บ baseline ไว้ในไฟล์โมเดล (`role_baselines`)

**`role_z` ใช้วินิจฉัยสาเหตุเท่านั้น ไม่ใช่ input ของโมเดล** เพราะวัดผลแล้วว่าทำให้โมเดลใช้กับข้อมูลจากแหล่งอื่นไม่ได้
(ผลเฉลี่ย 5 seed ใน `data/processed/robustness_sweep.csv`):

| Isolation Forest | F1 บน test split | F1 บนชุดอิสระ | precision บนชุดอิสระ |
|---|---|---|---|
| ไม่มี role_z (ใช้อยู่) | 0.860 | 0.719 (0.418–0.865) | 0.670 |
| มี role_z | 0.905 | 0.096 (0.086–0.108) | 0.051 |

role_z ให้ F1 ในข้อมูลชุดเดียวกันดีขึ้นเล็กน้อย แต่ baseline ของ role ใน generator ต่างจากข้อมูลแหล่งอื่นนิดเดียว
ค่านี้ก็พุ่งทุกแถวและโมเดล flag เกือบทั้งหมด

### Probable-cause diagnosis

`diagnose.py` turns each flagged reading into a hypothesis: a probable cause (runaway process / traffic surge / gradual slowdown),
the strongest signals (distance from the server role's normal, in MAD) and a role-specific first thing to check. It is rule-based on
the `*_role_z` values, so every conclusion can be traced back to numbers. `evaluate.py` writes `cause_diagnosis_report.csv`:
on readings flagged by the Isolation Forest the diagnosed cause matches the injected type 100% for CPU_SPIKE and NETWORK_SURGE
and about 93% for RESPONSE_DEGRADATION, and about 84% of false alarms are left unclassified. The rules were written around the
simulated patterns, so these figures are optimistic for real systems.

### เหตุการณ์รวม, ระดับความมั่นใจ และปุ่มป้อนกลับ (หน้า Alerts)

- **เหตุการณ์ (incident):** ข้อมูลที่ถูกตรวจพบติดๆ กัน (ห่างกันไม่เกิน 3 รอบ) บนเซิร์ฟเวอร์เดียวกันถูกรวมเป็น 1 เหตุการณ์
  (บน test split: 977 ข้อมูลที่ถูก flag เป็นเพียง 149 เหตุการณ์ เฉลี่ย 6.6 รายการต่อเหตุการณ์ เทียบเหตุการณ์จริง 70 ครั้ง)
- **ระดับความมั่นใจ:** dashboard ใช้ทั้ง Isolation Forest และ Z-score ถ้าเห็นตรงกันถือเป็น "มั่นใจสูง" ถ้าเห็นฝ่ายเดียวระบุว่าเป็น
  "ค่าใดค่าหนึ่งผิดปกติชัดเจน" (Z-score) หรือ "รูปแบบ/ช่วงเวลาผิดปกติ" (Isolation Forest) บนชุดอิสระ (เฉลี่ย 5 seed) ระดับมั่นใจสูงแม่น 99%
  จับได้ 76.5% ของเหตุการณ์จริง ส่วน Isolation Forest ฝ่ายเดียวแม่นเพียง 27% (4–50%) ตัวเลขนี้มาจากสคริปต์ชั่วคราว ยังไม่อยู่ใน repo
- **ปุ่มป้อนกลับ:** ผู้ใช้ตอบ "เป็นปัญหาจริง/เตือนผิด" ต่อเหตุการณ์ แล้วหน้านี้สรุปสัดส่วนที่เป็นจริงแยกตามระดับความมั่นใจ
  เก็บอยู่ใน session เท่านั้น (ต้นแบบ ยังไม่บันทึกถาวร)
- ข้อมูล live ใช้ชั่วโมงเดิมของข้อมูลที่ replay เป็นฟีเจอร์เวลา ไม่ใช่นาฬิกาปัจจุบัน ผลของเดโมจึงไม่เปลี่ยนตามเวลาที่เปิดใช้

### การจูน hyperparameter

`scripts/tune_isolation_forest.py` ทำ grid search 16 ชุดค่า (contamination, n_estimators, max_samples) บน validation split ตามเวลา
ผล: `contamination` มีผลมากที่สุด (F1 บน validation: 0.01 → 0.49–0.50, 0.02 → 0.79–0.80, **0.03 → 0.83–0.88**, 0.05 → 0.59–0.61)
ค่าที่ validation เลือก (`max_samples=512`) ดีขึ้นเฉลี่ย 5 seed เล็กน้อย (F1 test 0.860 → 0.877, ชุดอิสระ 0.719 → 0.750)
แต่ชนะ default เพียง 3/5 และ 2/5 seed จึง**ไม่ได้นำมาใช้** โมเดลที่ deploy ใช้ค่า default (contamination 0.03, 200 ต้น)
การใช้ป้ายคำตอบจูนเกณฑ์ทำได้เพราะข้อมูลเป็นข้อมูลจำลอง ในระบบจริงต้องอาศัยตัวอย่างที่มีคนยืนยัน (ปุ่มป้อนกลับ)

### Language switch (EN / TH)

The dashboard sidebar has an EN / TH toggle. All UI text lives in `app/i18n.py` and the diagnosis texts in `diagnose.py`.
Alerts store language-neutral data (type code and raw signals) and the text is built at display time. A test checks that
both dictionaries have the same keys and placeholders.

## 3) ประเมินผลกับชุดข้อมูลอิสระ (Step 5 เสริม — สร้างจาก AI คนละตัว)

```bash
python Generate_data_AI.py                                    # -> independent_test_set_v2.csv
python evaluate_independent.py --infile independent_test_set_v2.csv
python retrain_with_independent.py --independent-infile independent_test_set_v2.csv   # ทดลอง retrain
python scripts/robustness_sweep.py                             # ทดสอบความทนทานข้ามหลาย seed
python scripts/model_comparison.py                             # ความไวต่อเกณฑ์, การรวมสองวิธี, stress test เชิงบริบท
python scripts/tune_isolation_forest.py                        # grid search 16 ชุดค่าบน validation split ตามเวลา
```

ผลที่ควรรายงานตามจริง:

1. **ชุดอิสระเล็ก** (2 วัน, 7,000 แถว, 218 anomaly) และสร้างโดย AI อีกตัว จึงเป็นหลักฐานเบื้องต้น ไม่ใช่ข้อสรุปขาด
2. **ผลบนชุดอิสระแกว่งมากตามชุดเทรนที่สร้างขึ้นใหม่:** ทดสอบ 5 seed ได้ F1 ของ Isolation Forest ระหว่าง 0.418–0.865
   (เฉลี่ย 0.719) precision แกว่งตั้งแต่ 0.27 ถึง 0.91 ขณะที่ recall นิ่ง (0.83–0.91) เหตุผลที่เป็นไปได้: เกณฑ์ตัดสินของ
   Isolation Forest ถูกตั้งจากสัดส่วน `contamination` ของข้อมูลเทรน พอข้อมูลปกติของแหล่งอื่นอยู่ใกล้ขอบเกณฑ์ ตัวเลขจึงขยับง่าย
3. **Z-score ทำได้ดีและนิ่งกว่า Isolation Forest บนชุดอิสระ:** F1 เฉลี่ย 0.857 (0.686–0.915) เทียบ 0.719 และบน test split
   สองวิธีใกล้เคียงกัน (0.846 กับ 0.860) ดังนั้นข้อมูลชุดนี้ยังไม่ได้พิสูจน์ว่า Isolation Forest เหนือกว่า baseline อย่างง่าย
   ซึ่งสอดคล้องกับธรรมชาติของข้อมูลจำลองที่ anomaly เป็นค่าพุ่งชัดเจน ควรเขียนเป็นข้อจำกัดและข้อเสนอแนะในรายงาน
4. **การ retrain วัดเทียบบนชุดแถวเดียวกัน** (`evaluation_report_retrain_holdout.csv`): โมเดลเดิมและโมเดลที่ retrain ถูกวัดบนครึ่งหลัง
   ของชุดอิสระเหมือนกัน (3,504 แถว) ได้ F1 0.294 → 0.447 ในการรันนี้ (รันเดียว ยังไม่ได้ทดสอบข้ามหลาย seed)
   `retrain_with_independent.py` เซฟโมเดลใหม่แยกไว้ ไม่ทับโมเดล deploy
5. **ความไวต่อเกณฑ์ตัดสิน** (`model_comparison.csv`, 5 seed, บนชุดอิสระ): F1 ของ Isolation Forest เปลี่ยนมากตาม `contamination`
   (0.01 → 0.28, 0.02 → 0.81, 0.03 → 0.72, 0.05 → 0.31) ส่วน Z-score นิ่งกว่าเมื่อเกณฑ์ตั้งแต่ 3.0 ขึ้นไป (F1 0.857–0.884)
   แต่แย่เมื่อเกณฑ์ 2.5 (0.56) ตัวเลขนี้แสดงความไว ไม่ใช่ค่าที่ควรเลือกใช้ เพราะการเลือกเกณฑ์จากป้ายคำตอบของชุดทดสอบเป็นการจูนกับข้อสอบ
6. **การรวมสองวิธี:** flag เมื่อทั้งสองเห็นตรงกัน (AND) ได้ F1 0.863 (0.846–0.891) precision 0.991 recall 0.765 นิ่งที่สุด
   ส่วน flag เมื่อวิธีใดวิธีหนึ่งเห็น (OR) ได้ recall 0.918 แต่ precision 0.652
7. **Stress test เชิงบริบท** (ออกแบบเอง: โหลดระดับกลางวันในช่วง 18:00–08:00 โดยแต่ละค่าไม่เกินช่วงปกติเมื่อดูแยกกัน):
   Z-score แทบไม่จับได้เลย (recall 0.006–0.009) Isolation Forest จับระดับ strong ได้ 0.351 (0.270–0.470) และระดับ moderate ได้ 0.027
   แสดงว่า Isolation Forest มีความสามารถเชิงบริบทที่ Z-score ไม่มี แต่จำกัด และ stress test นี้ออกแบบให้โมเดลที่ใช้ `hour` ได้เปรียบ จึงไม่ใช่หลักฐานบนระบบจริง

## 4) เปิด live dashboard

```bash
streamlit run app/Home.py
```

รันจาก root ของโปรเจกต์เท่านั้น มี 4 หน้าใน sidebar: **Home** (ภาพรวม), **Alerts** (กรอง ดาวน์โหลด CSV ดูสาเหตุที่เป็นไปได้),
**Model Evaluation** (ผลประเมินโมเดล) และ **Servers** (ภาพรวมทุกเครื่องและรายละเอียดรายเครื่อง)

ข้อมูล live เป็นการ **replay** ข้อมูลจำลองที่บันทึกไว้ (`data/processed/replay_sample.csv` ซึ่งตัดมาจากส่วน test split)
เข้าโมเดลทีละเครื่องตามลำดับเวลา ไม่ใช่การสุ่มค่าใหม่ จึงมีการกระจายของข้อมูลเหมือนที่โมเดลเรียนรู้ ในการทดสอบ false alarm
อยู่ราว 2% และจับ anomaly ที่ฝังไว้ได้ราว 96% (ตัวจำลองสุ่มแบบเดิมได้ false alarm 5–10%) โมเดลที่ deploy ถูกเทรนจากข้อมูลทั้งหมด
ซึ่งรวมช่วงที่ replay ด้วย ผลบนหน้านี้จึงใช้สาธิตการทำงานของระบบ ไม่ใช่หลักฐานการประเมินผล

ถ้าไม่มีไฟล์โมเดล แอปจะรัน pipeline offline ให้เองตอนเปิดครั้งแรก และถ้าไม่มี `replay_sample.csv` จะสลับไปใช้ตัวจำลองแบบสุ่ม

## 5) Deploy + กันแอปหลับ (Streamlit Community Cloud)

Deploy จาก GitHub repo นี้ โดยชี้ Main file path ไปที่ `app/Home.py`
Streamlit พักแอปฟรีเองหลังไม่มี traffic จริง 12 ชม. — ตั้ง keepalive กันไว้ได้:

1. GitHub repo → Settings → Secrets and variables → Actions → แท็บ **Variables**
2. New repository variable: Name `STREAMLIT_APP_URL`, Value = ลิงก์แอปจริง
3. workflow `.github/workflows/keepalive.yml` จะเปิดเว็บด้วยเบราว์เซอร์จริง (Playwright)
   ทุก 4 ชม. และกดปุ่มปลุกให้ถ้าแอปหลับ (HTTP ping ธรรมดาใช้ไม่ได้ผล) — กดรันเองได้ที่
   แท็บ Actions → Streamlit Keepalive → Run workflow

`requirements.txt` กำหนด `streamlit>=1.51` เพราะโค้ดใช้ `width="stretch"` แทน `use_container_width` ที่ถูกประกาศเลิกใช้แล้ว

## 6) รันเทสต์

```bash
pytest tests/ -v
```

มี GitHub Actions (`.github/workflows/tests.yml`) รันเทสต์ชุดนี้อัตโนมัติทุกครั้งที่ push

## 7) เหตุผลของการออกแบบ (Design decisions)

- **แบ่ง train/test ตามเวลา แยกทีละ server:** anomaly ถูกแทรกเป็นช่วงต่อเนื่อง 6–20 แถว การสุ่มแบ่งแถวจะทำให้แถวของเหตุการณ์เดียวกัน
  ไปอยู่ทั้ง train และ test (data leakage) ผลวัดจึงดีเกินจริง
- **fit ตัวปรับสเกลและ baseline จาก train เท่านั้น** (min-max, mean/std, median/MAD) ไม่ใช้ข้อมูล test ช่วยตั้งเกณฑ์ตัวเอง
- **Data Error กับ True Anomaly แยกกัน:** ค่าที่เป็นไปไม่ได้ (CPU > 100, ค่าติดลบ) เป็นความผิดพลาดของระบบเก็บข้อมูล ETL แทนด้วย NaN แล้วเติมด้วย
  forward-fill ต่อ server ส่วน True Anomaly (`is_anomaly = 1`) ห้ามลบ เพราะเป็นสิ่งที่ใช้ประเมินผล
- **ไม่ normalize `hour` และ `day_of_week`:** เคยทดสอบ normalize แล้ว F1 ของ LOF ลดจาก 0.605 เหลือ 0.304 เพราะ `hour` ที่มี scale ใหญ่ช่วยจัดกลุ่มเพื่อนบ้านตามช่วงเวลา
- **ฟีเจอร์ trend (`response_time_dev_from_roll`, หน้าต่าง 12 จุด):** anomaly แบบ RESPONSE_DEGRADATION ค่อยๆ ไต่ขึ้น ฟีเจอร์ที่ดูค่าทีละจุดมองไม่เห็นช่วงต้นของเหตุการณ์
  recall ของประเภทนี้จึงต่ำกว่า CPU_SPIKE/NETWORK_SURGE ในทุกโมเดล
- **`contamination = 0.03`** ตั้งตามสัดส่วน anomaly ที่ฝังไว้ ในระบบจริงไม่ทราบค่านี้ จึงเป็นสมมติฐานที่ต้องปรับตามสภาพแวดล้อม
- **median/MAD สำหรับ role baseline** ทนต่อ anomaly ที่ปนอยู่ในข้อมูล (~3%) ได้ดีกว่า mean/std
- **ไม่ใช้ `role_z` เป็น input ของโมเดล** ดูหัวข้อ role_z ด้านบน
- **ให้คะแนนแบบ batch:** dashboard ให้คะแนนทุก reading ของแต่ละรอบในการเรียกโมเดลครั้งเดียว (เดิมเรียกทีละแถว ช้ากว่าราว 19 เท่าเมื่อวัด 20 แถว)
- **รายการ dropdown ต้องคงที่:** Streamlit ถือว่า dropdown ที่ตัวเลือกหรือข้อความของตัวเลือกเปลี่ยนเป็น widget ตัวใหม่ แล้วรีเซ็ตค่าที่เลือก
  จึงไม่ใส่สถานะสดในข้อความของตัวเลือก และเก็บค่าที่เลือกไว้ใน session แยกต่างหาก
- **ผลลัพธ์ทั้งหมดอิงข้อมูลจำลอง:** label มาจาก generator ที่เราเขียนเอง ผลจึงบอกว่าโมเดลจับรูปแบบที่ตั้งใจฝังได้ไหม ไม่ได้พิสูจน์ประสิทธิภาพบนระบบจริง
