# detector/yolo_detector.py  (deploy branch — PIL based)
# ORIGINAL code preserved exactly.
# ADDED: optional zone_manager, show_heatmap, heatmap_acc, pixels_per_meter args
#        tracker + speed estimator as module-level singletons
#        All new args default to None/False so existing calls still work.
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from detector.distance import estimate_distance
from detector.tracker import CentroidTracker
from detector.speed_estimator import SpeedEstimator
from utils.summary import log_detection

model = YOLO("yolov8n.pt")

PEDESTRIAN_CLASS_ID = 0
VEHICLE_CLASS_IDS   = [2, 3, 5, 7]

# Module-level singletons — persist across frames within a session
_tracker = CentroidTracker(max_disappeared=25, max_distance=80)
_speed   = SpeedEstimator(pixels_per_meter=40.0)


def reset_tracker():
    """Call when starting a new session or resetting."""
    global _tracker, _speed
    _tracker = CentroidTracker(max_disappeared=25, max_distance=80)
    _speed   = SpeedEstimator(pixels_per_meter=40.0)


def get_speed_estimator():
    return _speed


# ── Risk helpers (original, unchanged) ────────────────────────────────────────

def get_risk(distance):
    if distance is None:
        return "low"
    if distance < 4.0:
        return "high"
    elif distance < 7.0:
        return "medium"
    return "low"


def get_bbox_color(risk):
    return {
        "high":   "#FF0000",
        "medium": "#FFA500",
        "low":    "#00FF00",
    }.get(risk, "#00FF00")


# ── Main detection function ────────────────────────────────────────────────────

def detect_objects(frame, summary, confidence_threshold=0.4,
                   proximity_threshold=7.0,
                   zone_manager=None,
                   show_heatmap=False,
                   heatmap_acc=None,
                   pixels_per_meter=40.0):
    """
    frame: numpy array (H, W, 3) in RGB
    Returns: numpy array (RGB), alert bool

    New optional args (all default to None/False — original callers unaffected):
      zone_manager     : DangerZoneManager instance or None
      show_heatmap     : bool — composite heatmap onto frame
      heatmap_acc      : HeatmapAccumulator instance or None
      pixels_per_meter : float — speed calibration
    """
    global _speed
    _speed.ppm = pixels_per_meter

    h_frame, w_frame = frame.shape[:2]

    # ── Run YOLO (original) ────────────────────────────────────────────────
    pil_img = Image.fromarray(frame)
    draw    = ImageDraw.Draw(pil_img)
    results = model(frame, verbose=False, conf=confidence_threshold)[0]

    alert_triggered = False
    rects, labels   = [], []
    raw_dets        = []

    # ── First pass: collect boxes + update summary (original logic) ────────
    for box in results.boxes:
        cls_id      = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == PEDESTRIAN_CLASS_ID:
            height   = y2 - y1
            distance = estimate_distance(height)
            risk     = get_risk(distance)
            color    = get_bbox_color(risk)
            dist_str = f"{distance}m" if distance else "?"
            label_str= "person"

            summary["pedestrians"] += 1
            summary["total"]       += 1
            log_detection(summary, "Pedestrian", distance, risk)

            if distance is not None and distance < proximity_threshold:
                alert_triggered = True
                summary["alerts"] += 1

        elif cls_id in VEHICLE_CLASS_IDS:
            height   = y2 - y1
            distance = estimate_distance(height, known_height=1.5)
            risk     = get_risk(distance)
            color    = "#FFD700"
            dist_str = f"{distance}m" if distance else "?"
            label_str= "vehicle"

            summary["vehicles"] += 1
            summary["total"]    += 1
            log_detection(summary, "Vehicle", distance, risk)

        else:
            continue

        rects.append((x1, y1, x2, y2))
        labels.append(label_str)
        raw_dets.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "label": label_str, "distance": distance,
            "risk": risk, "color": color, "dist_str": dist_str,
        })

    # ── Tracker update ─────────────────────────────────────────────────────
    tracked    = _tracker.update(rects, labels)
    active_ids = set(tracked.keys())
    _speed.cleanup(active_ids)

    # ── Heatmap update ─────────────────────────────────────────────────────
    if heatmap_acc is not None:
        heatmap_acc.ensure_size(h_frame, w_frame)
        for det in raw_dets:
            cx = (det["x1"] + det["x2"]) // 2
            cy = (det["y1"] + det["y2"]) // 2
            heat_val = {"high": 2.0, "medium": 1.0, "low": 0.5}.get(det["risk"], 1.0)
            heatmap_acc.update([(cx, cy)], heat_val)

    # ── Danger zones overlay on PIL image ──────────────────────────────────
    if zone_manager:
        pil_img = zone_manager.draw_on_frame(pil_img)
        draw    = ImageDraw.Draw(pil_img)

    # ── Heatmap composite ──────────────────────────────────────────────────
    if show_heatmap and heatmap_acc is not None:
        pil_img = heatmap_acc.composite_on(pil_img)
        draw    = ImageDraw.Draw(pil_img)

    # ── Second pass: draw boxes with IDs + speeds (replaces original draw) ─
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for det in raw_dets:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        color    = det["color"]
        label    = det["label"]
        distance = det["distance"]
        risk     = det["risk"]
        dist_str = det["dist_str"]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Find object ID from tracker
        oid      = None
        best_d   = 999
        for oid_k, (tx, ty, _) in tracked.items():
            d = abs(tx - cx) + abs(ty - cy)
            if d < best_d:
                best_d, oid = d, oid_k

        # Speed update
        if oid is not None:
            _speed.update(oid, cx, cy)

        # Danger zone check
        in_zone    = False
        zone_name  = ""
        if zone_manager:
            hit = zone_manager.check(cx, cy, w_frame, h_frame)
            if hit:
                in_zone   = True
                zone_name = hit[0].name
                summary["zone_triggers"] = summary.get("zone_triggers", 0) + 1

        # Build label — same style as original + ID + speed + zone
        parts = []
        if oid is not None:
            parts.append(f"#{oid}")
        if label == "person":
            parts.append(f"Pedestrian {dist_str} [{risk.upper()}]")
        else:
            parts.append(f"Vehicle {dist_str} [{risk.upper()}]")
        if oid is not None and label == "vehicle":
            spd = _speed.get(oid)
            if spd > 1:
                parts.append(f"{spd:.0f}km/h")
        if in_zone:
            parts.append(f"⚠ {zone_name}")
        label_text = " | ".join(parts)

        # Draw trajectory for vehicles
        if oid is not None and label == "vehicle":
            traj = _tracker.get_trajectory(oid)
            if len(traj) > 2:
                hex_c = color.lstrip("#")
                rc = int(hex_c[0:2], 16)
                gc = int(hex_c[2:4], 16)
                bc = int(hex_c[4:6], 16)
                pts = [(int(p[0]), int(p[1])) for p in traj[-15:]]
                for j in range(1, len(pts)):
                    alpha = int(j / len(pts) * 200)
                    draw.line([pts[j-1], pts[j]], fill=(rc, gc, bc, alpha), width=2)

        # Bounding box (original style)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Label background + text (original style)
        text_bbox = draw.textbbox((x1, y1 - 18), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 18), label_text, fill="white", font=font)

    return np.array(pil_img), alert_triggered