import pandas as pd

SEVERITY_RANK = {"High": 3, "Medium": 2, "Low": 1, "None": 0}
DETECTION_RANK = {"both": 3, "zscore": 2, "forest": 1}
INCIDENT_GAP = 3
INCIDENT_COLUMNS = ["incident_id", "server_id", "role", "start", "end", "readings", "alert_type",
                    "severity_level", "detection", "signals_raw", "status"]


def zscore_stats(df: pd.DataFrame, cols: list, threshold: float = 3.0) -> dict:
    mean = df[cols].mean()
    std = df[cols].std(ddof=0).replace(0, 1)
    return {"cols": list(cols), "mean": mean.to_dict(), "std": std.to_dict(), "threshold": threshold}


def zscore_flag(values: dict, stats: dict) -> bool:
    return any(abs((values[c] - stats["mean"][c]) / stats["std"][c]) > stats["threshold"] for c in stats["cols"])


def detection_source(forest_flag: bool, zscore_flagged: bool):
    if forest_flag and zscore_flagged:
        return "both"
    if zscore_flagged:
        return "zscore"
    if forest_flag:
        return "forest"
    return None


def group_incidents(alerts: pd.DataFrame, latest_reading_no: dict | None = None, gap: int = INCIDENT_GAP) -> pd.DataFrame:
    if alerts.empty:
        return pd.DataFrame(columns=INCIDENT_COLUMNS)
    latest_reading_no = latest_reading_no or {}
    rows = []
    for server, group in alerts.sort_values(["server_id", "reading_no"]).groupby("server_id", sort=False):
        run_id = group["reading_no"].diff().gt(gap).fillna(False).cumsum()
        for _, run in group.groupby(run_id):
            types = run["alert_type"]
            known = types[types != "Other"]
            dominant = (known if len(known) else types).mode().iloc[0]
            pool = run[run["alert_type"] == dominant]
            best = pool.loc[pool["anomaly_score"].idxmin()]
            last_no = int(run["reading_no"].iloc[-1])
            latest = latest_reading_no.get(server, last_no)
            rows.append({
                "incident_id": f"{server}-{int(run['reading_no'].iloc[0])}",
                "server_id": server,
                "role": run["role"].iloc[0],
                "start": run["timestamp"].min(),
                "end": run["timestamp"].max(),
                "readings": len(run),
                "alert_type": dominant,
                "severity_level": max(run["severity_level"], key=lambda s: SEVERITY_RANK.get(s, 0)),
                "detection": max(run["detection"], key=lambda d: DETECTION_RANK.get(d, 0)),
                "signals_raw": best["signals_raw"],
                "status": "ongoing" if latest - last_no <= gap else "ended",
            })
    return pd.DataFrame(rows, columns=INCIDENT_COLUMNS).sort_values("start", ascending=False).reset_index(drop=True)
