"""Rule-based root-cause hints for flagged anomalies, driven by deviation from each role's normal baseline."""
import math

METRIC_KEYS = [
    "cpu_usage_role_z", "memory_usage_role_z", "network_in_role_z",
    "network_out_role_z", "response_time_role_z",
]

METRIC_LABELS = {
    "en": {
        "cpu_usage_role_z": "CPU", "memory_usage_role_z": "Memory", "network_in_role_z": "Network in",
        "network_out_role_z": "Network out", "response_time_role_z": "Response time",
    },
    "th": {
        "cpu_usage_role_z": "CPU", "memory_usage_role_z": "Memory", "network_in_role_z": "Network เข้า",
        "network_out_role_z": "Network ออก", "response_time_role_z": "Response time",
    },
}

NO_SIGNAL = {"en": "none above 2 MAD", "th": "ไม่มีค่าใดเกิน 2 MAD"}

CAUSES = {
    "CPU_SPIKE": {
        "en": {
            "title": "Runaway process or CPU saturation",
            "checks": {
                "default": "Check the top processes by CPU and any recent deployment or scheduled job.",
                "Web Server": "Check web worker processes and request handlers stuck in a loop.",
                "Database Server": "Check long-running queries and background jobs such as backups or index builds.",
                "App Server": "Check application threads, stuck jobs and the most recent release.",
            },
        },
        "th": {
            "title": "โปรเซสทำงานหนักผิดปกติหรือ CPU เต็ม",
            "checks": {
                "default": "ตรวจโปรเซสที่ใช้ CPU สูงสุด รวมถึงการ deploy หรืองานที่ตั้งเวลาไว้ล่าสุด",
                "Web Server": "ตรวจ worker ของเว็บและ request handler ที่ติดลูป",
                "Database Server": "ตรวจ query ที่รันนาน และงานเบื้องหลัง เช่น backup หรือการสร้าง index",
                "App Server": "ตรวจ thread ของแอป งานที่ค้าง และ release ล่าสุด",
            },
        },
    },
    "NETWORK_SURGE": {
        "en": {
            "title": "Traffic surge",
            "checks": {
                "default": "Check the top traffic sources and compare with expected load such as batch transfers or backups.",
                "Web Server": "Check request volume and source addresses for unusual spikes or scanning.",
                "Database Server": "Check replication, bulk exports and backup traffic.",
                "App Server": "Check calls to other services and retry storms.",
            },
        },
        "th": {
            "title": "ทราฟฟิกพุ่งสูง",
            "checks": {
                "default": "ตรวจแหล่งทราฟฟิกสูงสุด แล้วเทียบกับโหลดที่คาดไว้ เช่น การส่งข้อมูลเป็นชุดหรือ backup",
                "Web Server": "ตรวจจำนวน request และ IP ต้นทางว่ามีการพุ่งหรือการสแกนผิดปกติหรือไม่",
                "Database Server": "ตรวจ replication การ export ข้อมูลก้อนใหญ่ และทราฟฟิก backup",
                "App Server": "ตรวจการเรียกบริการอื่นและการ retry ซ้ำๆ",
            },
        },
    },
    "RESPONSE_DEGRADATION": {
        "en": {
            "title": "Gradual slowdown",
            "checks": {
                "default": "Check dependencies, disk I/O and queue depth, and watch whether the trend keeps rising.",
                "Web Server": "Check upstream latency and connection pool saturation.",
                "Database Server": "Check slow queries, lock waits and disk I/O.",
                "App Server": "Check memory growth for a possible leak, garbage collection and downstream latency.",
            },
        },
        "th": {
            "title": "ระบบช้าลงอย่างต่อเนื่อง",
            "checks": {
                "default": "ตรวจ dependency, disk I/O และความยาวของคิว แล้วดูว่าแนวโน้มยังเพิ่มขึ้นต่อหรือไม่",
                "Web Server": "ตรวจ latency ของบริการต้นทางและ connection pool ที่เต็ม",
                "Database Server": "ตรวจ slow query, การรอ lock และ disk I/O",
                "App Server": "ตรวจการใช้ memory ที่เพิ่มขึ้น (อาจรั่ว), garbage collection และ latency ของบริการปลายทาง",
            },
        },
    },
    "UNCLASSIFIED": {
        "en": {
            "title": "No dominant signal",
            "checks": {"default": "Compare the metrics with the role baseline and watch the next few readings."},
        },
        "th": {
            "title": "ไม่มีสัญญาณเด่นชัด",
            "checks": {"default": "เทียบ metric กับค่าปกติของ role และดู reading ถัดไปอีกสักพัก"},
        },
    },
}

TREND_KEY = "response_time_dev_from_roll"
MIN_STRENGTH = 3.5


def _z(features, key):
    value = features.get(key, 0.0)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return float(value)


def cause_title(cause_type, lang="en"):
    return CAUSES[cause_type][lang.lower()]["title"]


def suggested_check(cause_type, role=None, lang="en"):
    checks = CAUSES[cause_type][lang.lower()]["checks"]
    return checks.get(role, checks["default"])


def top_signals(features, limit=3, min_abs=2.0):
    scored = sorted(((abs(_z(features, k)), k) for k in METRIC_KEYS), reverse=True)
    return [(k, _z(features, k)) for mag, k in scored[:limit] if mag >= min_abs]


def format_signals(signals, lang="en"):
    if not signals:
        return NO_SIGNAL[lang.lower()]
    labels = METRIC_LABELS[lang.lower()]
    return ", ".join(f"{labels[k]} {z:+.1f}" for k, z in signals)


def encode_signals(signals):
    return ";".join(f"{k}:{z:.2f}" for k, z in signals)


def decode_signals(raw):
    if not raw or not isinstance(raw, str):
        return []
    return [(part.split(":")[0], float(part.split(":")[1])) for part in raw.split(";")]


def diagnose(features, role=None, min_strength=MIN_STRENGTH, lang="en"):
    strengths = {
        "CPU_SPIKE": _z(features, "cpu_usage_role_z"),
        "NETWORK_SURGE": min(_z(features, "network_in_role_z"), _z(features, "network_out_role_z")),
        "RESPONSE_DEGRADATION": _z(features, "response_time_role_z") if _z(features, TREND_KEY) > 0 else 0.0,
    }
    cause_type, strength = max(strengths.items(), key=lambda kv: kv[1])
    if strength < min_strength:
        cause_type, strength = "UNCLASSIFIED", 0.0
    signals = top_signals(features)
    return {
        "cause_type": cause_type,
        "probable_cause": cause_title(cause_type, lang),
        "top_signals": format_signals(signals, lang),
        "suggested_check": suggested_check(cause_type, role, lang),
        "signals_raw": encode_signals(signals),
        "strength": round(strength, 2),
    }


def diagnose_frame(df, role_col=None, lang="en"):
    rows = []
    for _, row in df.iterrows():
        role = row[role_col] if role_col and role_col in df.columns else None
        rows.append(diagnose(row.to_dict(), role=role, lang=lang))
    keys = ("cause_type", "probable_cause", "top_signals", "suggested_check")
    return df.assign(**{k: [r[k] for r in rows] for k in keys})
