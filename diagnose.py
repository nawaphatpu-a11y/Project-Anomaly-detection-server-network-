"""Rule-based root-cause hints for flagged anomalies, driven by deviation from each role's normal baseline."""
import math

METRIC_LABELS = {
    "cpu_usage_role_z": "CPU",
    "memory_usage_role_z": "Memory",
    "network_in_role_z": "Network in",
    "network_out_role_z": "Network out",
    "response_time_role_z": "Response time",
}

CAUSES = {
    "CPU_SPIKE": {
        "title": "Runaway process or CPU saturation",
        "checks": {
            "default": "Check the top processes by CPU and any recent deployment or scheduled job.",
            "Web Server": "Check web worker processes and request handlers stuck in a loop.",
            "Database Server": "Check long-running queries and background jobs such as backups or index builds.",
            "App Server": "Check application threads, stuck jobs and the most recent release.",
        },
    },
    "NETWORK_SURGE": {
        "title": "Traffic surge",
        "checks": {
            "default": "Check the top traffic sources and compare with expected load such as batch transfers or backups.",
            "Web Server": "Check request volume and source addresses for unusual spikes or scanning.",
            "Database Server": "Check replication, bulk exports and backup traffic.",
            "App Server": "Check calls to other services and retry storms.",
        },
    },
    "RESPONSE_DEGRADATION": {
        "title": "Gradual slowdown",
        "checks": {
            "default": "Check dependencies, disk I/O and queue depth, and watch whether the trend keeps rising.",
            "Web Server": "Check upstream latency and connection pool saturation.",
            "Database Server": "Check slow queries, lock waits and disk I/O.",
            "App Server": "Check memory growth for a possible leak, garbage collection and downstream latency.",
        },
    },
    "UNCLASSIFIED": {
        "title": "No dominant signal",
        "checks": {
            "default": "Compare the metrics with the role baseline and watch the next few readings.",
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


def _top_signals(features, limit=3, min_abs=2.0):
    scored = sorted(((abs(_z(features, k)), k) for k in METRIC_LABELS), reverse=True)
    parts = [f"{METRIC_LABELS[k]} {_z(features, k):+.1f}" for mag, k in scored[:limit] if mag >= min_abs]
    return ", ".join(parts) if parts else "none above 2 MAD"


def diagnose(features, role=None, min_strength=MIN_STRENGTH):
    strengths = {
        "CPU_SPIKE": _z(features, "cpu_usage_role_z"),
        "NETWORK_SURGE": min(_z(features, "network_in_role_z"), _z(features, "network_out_role_z")),
        "RESPONSE_DEGRADATION": _z(features, "response_time_role_z") if _z(features, TREND_KEY) > 0 else 0.0,
    }
    cause_type, strength = max(strengths.items(), key=lambda kv: kv[1])
    if strength < min_strength:
        cause_type, strength = "UNCLASSIFIED", 0.0
    cause = CAUSES[cause_type]
    return {
        "cause_type": cause_type,
        "probable_cause": cause["title"],
        "top_signals": _top_signals(features),
        "suggested_check": cause["checks"].get(role, cause["checks"]["default"]),
        "strength": round(strength, 2),
    }


def diagnose_frame(df, role_col=None):
    rows = []
    for _, row in df.iterrows():
        rows.append(diagnose(row.to_dict(), role=row[role_col] if role_col and role_col in df.columns else None))
    return df.assign(**{k: [r[k] for r in rows] for k in ("cause_type", "probable_cause", "top_signals", "suggested_check")})
