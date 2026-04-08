# utils/summary.py  (deploy branch)
# CHANGED: added high_risk_count + zone_triggers to init_summary()
#          added 1 line to log_detection() to track high risk count
#          everything else is identical to original
from datetime import datetime


def init_summary():
    return {
        "pedestrians":    0,
        "vehicles":       0,
        "total":          0,
        "alerts":         0,
        "high_risk_count": 0,   # NEW — used by road report scoring
        "zone_triggers":   0,   # NEW — danger zone hits
        "start_time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detection_log":  []
    }


def reset_summary():
    return init_summary()


def log_detection(summary, detection_type, distance=None, risk=None):
    """Log each detection event with timestamp."""
    summary["detection_log"].append({
        "time":       datetime.now().strftime("%H:%M:%S"),
        "type":       detection_type,
        "distance_m": str(round(distance, 2)) if distance is not None else "N/A",
        "risk":       risk or "low"
    })
    # NEW — track high risk for road report scoring
    if risk == "high":
        summary["high_risk_count"] += 1