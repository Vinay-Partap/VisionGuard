import cv2
from ultralytics import YOLO

from detector.distance import estimate_distance, is_near
from utils.alerts import show_alert, should_alert

# Load YOLO model
model = YOLO("yolov8n.pt")

# Class IDs
HUMAN_CLASS = 0
VEHICLE_CLASSES = [1, 2, 3, 5, 7]


def detect_objects(frame, summary):
    results = model.predict(frame, conf=0.4)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # -------- PEDESTRIAN --------
        if cls == HUMAN_CLASS:
            distance = estimate_distance(x1, x2)
            near = is_near(distance)

            color = (0, 0, 255) if near else (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"Pedestrian {distance:.2f}m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            summary["pedestrians"] += 1

            # ✅ ALERT WITH COOLDOWN
            if near and should_alert():
                show_alert()

        # -------- VEHICLE --------
        elif cls in VEHICLE_CLASSES:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            summary["vehicles"] += 1

    summary["total"] = summary["pedestrians"] + summary["vehicles"]
    return frame, summary
