# detector/yolo_detector.py  (main branch)
import cv2
from ultralytics import YOLO
from detector.distance import estimate_distance
from detector.tracker import CentroidTracker
from detector.speed_estimator import SpeedEstimator
from utils.summary import log_detection

model = YOLO("yolov8n.pt")

PEDESTRIAN_CLASS_ID = 0
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, bike, bus, truck

# Module-level tracker + speed estimator (persist across frames in a session)
_tracker = CentroidTracker(max_disappeared=25, max_distance=80)
_speed   = SpeedEstimator(pixels_per_meter=40.0)


def reset_tracker():
    """Call when starting a new session / resetting."""
    global _tracker, _speed
    _tracker = CentroidTracker(max_disappeared=25, max_distance=80)
    _speed   = SpeedEstimator(pixels_per_meter=40.0)


def get_speed_estimator():
    return _speed


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
        "high":   (0, 0, 255),
        "medium": (0, 165, 255),
        "low":    (0, 255, 0),
    }.get(risk, (0, 255, 0))


def detect_objects(frame, summary, confidence_threshold=0.4,
                   proximity_threshold=7.0,
                   zone_manager=None,
                   show_heatmap=False,
                   heatmap_acc=None,
                   pixels_per_meter=40.0):
    """
    Run YOLOv8 detection with tracking IDs, speed estimation, danger zones,
    and optional heatmap overlay.

    Returns: (annotated_frame, alert_triggered)
    """
    global _speed
    _speed.ppm = pixels_per_meter

    results = model(frame, verbose=False, conf=confidence_threshold)[0]
    h_frame, w_frame = frame.shape[:2]
    alert_triggered = False

    rects, labels = [], []

    # ── First pass: collect boxes ─────────────────────────────────────────────
    raw_detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_h = y2 - y1

        if cls_id == PEDESTRIAN_CLASS_ID:
            distance = estimate_distance(box_h)
            risk = get_risk(distance)
            label_str = "person"
            summary["pedestrians"] += 1
            summary["total"] += 1
            log_detection(summary, "Pedestrian", distance, risk)
            if distance is not None and distance < proximity_threshold:
                alert_triggered = True
                summary["alerts"] += 1
        elif cls_id in VEHICLE_CLASS_IDS:
            distance = estimate_distance(box_h, known_height=1.5)
            risk = get_risk(distance)
            label_str = "vehicle"
            summary["vehicles"] += 1
            summary["total"] += 1
            log_detection(summary, "Vehicle", distance, risk)
        else:
            continue

        rects.append((x1, y1, x2, y2))
        labels.append(label_str)
        raw_detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "label": label_str, "distance": distance, "risk": risk,
        })

    # ── Tracker update ────────────────────────────────────────────────────────
    tracked = _tracker.update(rects, labels)
    active_ids = set(tracked.keys())
    _speed.cleanup(active_ids)

    # Build centroid → object_id map
    centroid_to_oid = {(cx, cy): oid for oid, (cx, cy, _) in tracked.items()}

    # ── Danger zone overlay (drawn before boxes) ──────────────────────────────
    if zone_manager:
        frame = zone_manager.draw_on_frame(frame)

    # ── Heatmap update ────────────────────────────────────────────────────────
    if heatmap_acc is not None:
        heatmap_acc.ensure_size(h_frame, w_frame)
        centroids_this_frame = []
        for det in raw_detections:
            cx = (det["x1"] + det["x2"]) // 2
            cy = (det["y1"] + det["y2"]) // 2
            heat_val = {"high": 2.0, "medium": 1.0, "low": 0.5}.get(det["risk"], 1.0)
            centroids_this_frame.append((cx, cy))
            heatmap_acc.update([(cx, cy)], heat_val)
        if show_heatmap:
            frame = heatmap_acc.composite_on(frame, alpha=0.55, show_boxes=True, box_thresh=0.6, box_min_area=200)

    # ── Second pass: draw boxes with IDs + speeds ─────────────────────────────
    for det in raw_detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        risk = det["risk"]
        label = det["label"]
        distance = det["distance"]
        color = get_bbox_color(risk)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Find object ID
        oid = None
        best_d = 999
        for (tx, ty), tid in centroid_to_oid.items():
            d = abs(tx - cx) + abs(ty - cy)
            if d < best_d:
                best_d, oid = d, tid

        # Update speed for vehicles
        if oid is not None:
            _speed.update(oid, cx, cy)

        # ── Danger zone check ─────────────────────────────────────────────────
        in_zone = False
        zone_name = ""
        if zone_manager:
            hit_zones = zone_manager.check(cx, cy, w_frame, h_frame)
            if hit_zones:
                in_zone = True
                zone_name = hit_zones[0].name
                summary["zone_triggers"] = summary.get("zone_triggers", 0) + 1

        # ── Draw trajectory for vehicles ──────────────────────────────────────
        if oid is not None and label == "vehicle":
            traj = _tracker.get_trajectory(oid)
            if len(traj) > 2:
                for j in range(1, min(len(traj), 15)):
                    p1 = (int(traj[j-1][0]), int(traj[j-1][1]))
                    p2 = (int(traj[j][0]),   int(traj[j][1]))
                    alpha = j / min(len(traj), 15)
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(frame, p1, p2, c, 2)

        # ── Bounding box ──────────────────────────────────────────────────────
        thickness = 3 if risk == "high" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # ── Label text ────────────────────────────────────────────────────────
        parts = []
        if oid is not None:
            parts.append(f"#{oid}")
        parts.append(label.upper())
        if distance:
            parts.append(f"{distance}m")
        parts.append(f"[{risk.upper()}]")
        if oid is not None and label == "vehicle":
            spd = _speed.get(oid)
            if spd > 1:
                parts.append(f"{spd:.0f}km/h")
        if in_zone:
            parts.append(f"⚠ {zone_name}")

        text = " | ".join(parts)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    return frame, alert_triggered