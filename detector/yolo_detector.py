# detector/yolo_detector.py

import cv2
from ultralytics import YOLO
from detector.distance import estimate_distance

# Load stable YOLO model
model = YOLO("yolov8n.pt")

PEDESTRIAN_CLASS_ID = 0
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, bike, bus, truck


def detect_objects(frame, summary):
    results = model(frame, verbose=False)[0]

    alert_triggered = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        height = y2 - y1

        label = None
        color = (0, 255, 0)

        # Pedestrian
        if cls_id == PEDESTRIAN_CLASS_ID:
            distance = estimate_distance(height)
            label = f"Pedestrian {distance}m"
            color = (0, 0, 255)

            summary["pedestrians"] += 1
            summary["total"] += 1

            if distance and distance < 2.0:
                alert_triggered = True

        # Vehicle
        elif cls_id in VEHICLE_CLASS_IDS:
            label = "Vehicle"
            summary["vehicles"] += 1
            summary["total"] += 1

        if label:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    return frame, alert_triggered
