# reports/road_report.py
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

SESSIONS_FILE = "reports/sessions.json"


# ── Session helpers ────────────────────────────────────────────────────────────

def start_session(road_name, username):
    return {
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "road_name": road_name.strip() or "Unknown Road",
        "username": username,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_minutes": 0,
        "total_detections": 0,
        "pedestrian_count": 0,
        "vehicle_count": 0,
        "high_risk_count": 0,
        "alert_count": 0,
        "avg_speed_kmh": 0.0,
        "max_speed_kmh": 0.0,
        "speeding_violations": 0,
        "danger_zone_triggers": 0,
    }


def end_session(session, summary, speed_stats=None):
    now = datetime.now()
    session["end_time"] = now.isoformat()
    start = datetime.fromisoformat(session["start_time"])
    session["duration_minutes"] = round((now - start).total_seconds() / 60, 1)
    session["total_detections"] = summary.get("total", 0)
    session["pedestrian_count"] = summary.get("pedestrians", 0)
    session["vehicle_count"] = summary.get("vehicles", 0)
    session["high_risk_count"] = summary.get("high_risk_count", 0)
    session["alert_count"] = summary.get("alerts", 0)
    session["danger_zone_triggers"] = summary.get("zone_triggers", 0)
    if speed_stats:
        session["avg_speed_kmh"] = speed_stats.get("avg_speed", 0.0)
        session["max_speed_kmh"] = speed_stats.get("max_speed", 0.0)
        session["speeding_violations"] = speed_stats.get("speeding_count", 0)
    _append(session)
    return session


def _append(session):
    os.makedirs("reports", exist_ok=True)
    sessions = _load_all()
    sessions.append(session)
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


def _load_all():
    if not os.path.exists(SESSIONS_FILE):
        return []
    with open(SESSIONS_FILE) as f:
        return json.load(f)


def get_all_sessions_df():
    sessions = _load_all()
    if not sessions:
        return pd.DataFrame()
    df = pd.DataFrame(sessions)
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df.sort_values("start_time", ascending=False)


def get_road_names():
    return sorted({s["road_name"] for s in _load_all()})


# ── Report generation ──────────────────────────────────────────────────────────

def generate_report(period_days=15):
    sessions = _load_all()
    if not sessions:
        return {"error": "No session data yet. Run some detection sessions first."}

    cutoff = datetime.now() - timedelta(days=period_days)
    recent = [s for s in sessions
              if datetime.fromisoformat(s["start_time"]) >= cutoff]

    if not recent:
        return {"error": f"No sessions in the last {period_days} days."}

    road_data = defaultdict(lambda: {
        "sessions": 0, "duration": 0, "detections": 0,
        "pedestrians": 0, "vehicles": 0, "high_risk": 0,
        "alerts": 0, "speeding": 0, "zone_triggers": 0,
        "max_speed": 0.0, "speed_readings": [],
    })

    for s in recent:
        r = road_data[s["road_name"]]
        r["sessions"] += 1
        r["duration"] += s.get("duration_minutes", 0)
        r["detections"] += s.get("total_detections", 0)
        r["pedestrians"] += s.get("pedestrian_count", 0)
        r["vehicles"] += s.get("vehicle_count", 0)
        r["high_risk"] += s.get("high_risk_count", 0)
        r["alerts"] += s.get("alert_count", 0)
        r["speeding"] += s.get("speeding_violations", 0)
        r["zone_triggers"] += s.get("danger_zone_triggers", 0)
        r["max_speed"] = max(r["max_speed"], s.get("max_speed_kmh", 0))
        if s.get("avg_speed_kmh", 0) > 0:
            r["speed_readings"].append(s["avg_speed_kmh"])

    roads = []
    for road, r in road_data.items():
        dur = max(r["duration"], 1)
        avg_spd = (sum(r["speed_readings"]) / len(r["speed_readings"])
                   if r["speed_readings"] else 0)
        penalty = (
            (r["high_risk"] / dur) * 15 +
            (r["alerts"] / dur) * 10 +
            (r["speeding"] / dur) * 8 +
            max(0, avg_spd - 30) * 0.5
        )
        score = round(max(0, min(100, 100 - penalty * 10)), 1)

        if score >= 75:
            rec, css = "✅ SAFE", "safe"
            advice = "Low risk detected. This road is good to use."
        elif score >= 50:
            rec, css = "⚠️ CAUTION", "caution"
            advice = "Moderate risk. Drive carefully, especially at peak hours."
        else:
            rec, css = "🚨 AVOID", "avoid"
            advice = "High risk road. Frequent danger events. Use an alternate route."

        roads.append({
            "road_name": road, "safety_score": score,
            "recommendation": rec, "css": css, "advice": advice,
            "sessions": r["sessions"],
            "duration_min": round(r["duration"], 1),
            "detections": r["detections"],
            "pedestrians": r["pedestrians"],
            "vehicles": r["vehicles"],
            "high_risk": r["high_risk"],
            "alerts": r["alerts"],
            "speeding": r["speeding"],
            "avg_speed": round(avg_spd, 1),
            "max_speed": round(r["max_speed"], 1),
        })

    roads.sort(key=lambda x: x["safety_score"], reverse=True)
    return {
        "period_days": period_days,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_sessions": len(recent),
        "roads_analyzed": len(roads),
        "roads": roads,
        "safest_road": roads[0]["road_name"] if roads else "N/A",
        "riskiest_road": roads[-1]["road_name"] if len(roads) > 1 else "N/A",
    }


def export_report_txt(report):
    lines = [
        "=" * 58,
        "   VISIONGUARD AI — ROAD SAFETY PERIOD REPORT",
        "=" * 58,
        f"  Period       : Last {report['period_days']} days",
        f"  Generated    : {report['generated_at']}",
        f"  Sessions     : {report['total_sessions']}",
        f"  Roads        : {report['roads_analyzed']}",
        f"  Safest road  : {report['safest_road']}",
        f"  Riskiest road: {report['riskiest_road']}",
        "", "=" * 58, "  ROAD DETAILS", "=" * 58,
    ]
    for r in report["roads"]:
        lines += [
            f"\n  📍 {r['road_name']}",
            f"     Score       : {r['safety_score']}/100  {r['recommendation']}",
            f"     Advice      : {r['advice']}",
            f"     Sessions    : {r['sessions']}  |  Monitor time: {r['duration_min']} min",
            f"     Detections  : {r['detections']}  (Ped: {r['pedestrians']}, Veh: {r['vehicles']})",
            f"     High risk   : {r['high_risk']}  |  Alerts: {r['alerts']}",
            f"     Speeding    : {r['speeding']}  |  Avg speed: {r['avg_speed']} km/h",
            "  " + "-" * 55,
        ]
    lines += ["", "  Generated by VisionGuard AI — github.com/Vinay-Partap/VisionGuard", "=" * 58]
    return "\n".join(lines)