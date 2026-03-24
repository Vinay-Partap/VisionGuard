# detector/yolo_detector.py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from detector.distance import estimate_distance
from utils.summary import log_detection

model = YOLO("yolov8n.pt")

PEDESTRIAN_CLASS_ID = 0
VEHICLE_CLASS_IDS = [2, 3, 5, 7]


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


def detect_objects(frame, summary, confidence_threshold=0.4, proximity_threshold=7.0):
    """
    frame: numpy array (H, W, 3) in RGB
    returns: numpy array (RGB), alert bool
    """
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    results = model(frame, verbose=False, conf=confidence_threshold)[0]
    alert_triggered = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == PEDESTRIAN_CLASS_ID:
            height = y2 - y1
            distance = estimate_distance(height)
            risk = get_risk(distance)
            color = get_bbox_color(risk)
            dist_str = f"{distance}m" if distance else "?"
            label = f"Pedestrian {dist_str} [{risk.upper()}]"
            summary["pedestrians"] += 1
            summary["total"] += 1
            log_detection(summary, "Pedestrian", distance, risk)

            if distance is not None and distance < proximity_threshold:
                alert_triggered = True
                summary["alerts"] += 1

        elif cls_id in VEHICLE_CLASS_IDS:
            color = "#FFD700"
            label = "Vehicle"
            summary["vehicles"] += 1
            summary["total"] += 1
            log_detection(summary, "Vehicle")
        else:
            continue

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label background + text
        text_bbox = draw.textbbox((x1, y1 - 18), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 18), label, fill="white")

    return np.array(pil_img), alert_triggered