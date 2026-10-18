import pandas as pd

RECENT_WINDOW = 10
STATUS_ORDER = {"alert": 0, "watch": 1, "ok": 2}
STATUS_ICON = {"alert": "🔴", "watch": "🟡", "ok": "🟢"}
MAD_TO_SD = 1.4826


def overview(df):
    ordered = df.sort_values("timestamp")
    last = ordered.groupby("server_id").tail(1).set_index("server_id")
    stats = ordered.groupby("server_id").agg(readings=("is_anomaly", "size"), alerts=("is_anomaly", "sum"))
    recent = ordered.groupby("server_id").tail(RECENT_WINDOW).groupby("server_id")["is_anomaly"].sum()

    out = stats.join(last[["role", "timestamp", "cpu_usage", "memory_usage", "response_time", "is_anomaly"]])
    out = out.rename(columns={"is_anomaly": "latest_anomaly", "timestamp": "last_seen"})
    out["recent_alerts"] = recent
    out["alert_rate"] = (out["alerts"] / out["readings"] * 100).round(1)
    out["status"] = "ok"
    out.loc[out["recent_alerts"] > 0, "status"] = "watch"
    out.loc[out["latest_anomaly"] == 1, "status"] = "alert"
    out["_order"] = out["status"].map(STATUS_ORDER)
    out = out.sort_values(["_order", "alerts"], ascending=[True, False]).drop(columns="_order")
    return out.reset_index()


def normal_band(baselines, role, metric, k=3.0):
    if not baselines or role not in baselines or metric not in baselines[role]:
        return None
    stats = baselines[role][metric]
    spread = k * MAD_TO_SD * stats["mad"]
    return stats["median"] - spread, stats["median"] + spread


def status_counts(table):
    counts = table["status"].value_counts()
    return {s: int(counts.get(s, 0)) for s in STATUS_ORDER}


def pick_index(options, saved, fallback=0):
    return options.index(saved) if saved in options else fallback


def keep_valid(saved, options):
    return [item for item in (saved or []) if item in options]


def parse_signals(raw):
    if not raw or not isinstance(raw, str):
        return []
    return [(part.split(":")[0], float(part.split(":")[1])) for part in raw.split(";")]


def filter_history(df, types=None, metrics=None, above_only=False, hide_other=False):
    out = df
    if hide_other:
        out = out[out["alert_type"] != "Other"]
    if types:
        out = out[out["alert_type"].isin(types)]
    if metrics or above_only:
        wanted = {f"{m}_role_z" for m in metrics} if metrics else None

        def matches(raw):
            return any((wanted is None or key in wanted) and (z > 0 or not above_only)
                       for key, z in parse_signals(raw))

        out = out[out["signals_raw"].map(matches)]
    return out
