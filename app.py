import streamlit as st
import cv2
import numpy as np
import time

from detector.yolo_detector import detect_objects
from utils.alerts import should_alert, play_alert_sound
from utils.summary import init_summary, reset_summary

st.set_page_config(page_title="VisionGuard AI", layout="wide")

# ---------------- UI HEADER ----------------
st.title("👁️ VisionGuard AI")
st.subheader("Real-Time Pedestrian & Vehicle Detection System")

# ---------------- SESSION STATE ----------------
if "sound_played" not in st.session_state:
    st.session_state.sound_played = False

if "last_frame_time" not in st.session_state:
    st.session_state.last_frame_time = time.time()

# ---------------- INPUT TYPE ----------------
input_type = st.radio(
    "Select Input Type",
    ["Upload Image", "Upload Video", "Live Camera"]
)

frame_placeholder = st.empty()
alert_box = st.empty()
summary_box = st.empty()

# ---------------- IMAGE MODE ----------------
if input_type == "Upload Image":
    image_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "png", "jpeg"]
    )

    if image_file:
        image = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

        summary = init_summary()
        frame, alert = detect_objects(frame, summary)

        frame_placeholder.image(frame, channels="BGR")
        summary_box.json(summary)

        if alert and should_alert():
            alert_box.warning("🚨 Pedestrian too close!")
            play_alert_sound()

# ---------------- VIDEO MODE ----------------
elif input_type == "Upload Video":
    video_file = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture("temp_video.mp4")
        summary = init_summary()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, alert = detect_objects(frame, summary)

            frame_placeholder.image(frame, channels="BGR")
            summary_box.json(summary)

            if alert and should_alert():
                alert_box.warning("🚨 Pedestrian too close!")
                play_alert_sound()

            time.sleep(0.03)  # smooth playback

        cap.release()

# ---------------- LIVE CAMERA MODE ----------------
elif input_type == "Live Camera":
    st.info("Click **Start Camera** to begin live detection")

    start_cam = st.checkbox("Start Camera")

    if start_cam:
        cap = cv2.VideoCapture(0)
        summary = init_summary()

        if not cap.isOpened():
            st.error("Camera not accessible")
        else:
            while start_cam:
                ret, frame = cap.read()
                if not ret:
                    break

                frame, alert = detect_objects(frame, summary)

                frame_placeholder.image(frame, channels="BGR")
                summary_box.json(summary)

                if alert and should_alert():
                    alert_box.warning("🚨 Pedestrian too close!")
                    play_alert_sound()

                time.sleep(0.03)

            cap.release()
