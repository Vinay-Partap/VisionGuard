import streamlit as st
import tempfile
import cv2
import numpy as np
if "video_running" not in st.session_state:
    st.session_state.video_running = False

from detector.yolo_detector import detect_objects
from utils.summary import init_summary, update_summary

st.set_page_config(page_title="VISIONGUARD", layout="wide")

# Title
st.markdown("""
<h1 style='text-align:center;'>👁️‍🗨️ VISIONGUARD</h1>
<p style='text-align:center;'>Real-Time Pedestrian & Vehicle Detection System</p>
""", unsafe_allow_html=True)

# Initialize summary
summary = init_summary()

option = st.radio(
    "Select Input Type",
    ["Upload Image", "Upload Video"]
)

if option == "Upload Image":
    img_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if img_file:
        image = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(image, 1)

        processed_frame, summary = detect_objects(frame, summary)

        st.image(
            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

elif option == "Upload Video":
    video_file = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if video_file and not st.session_state.video_running:
        st.session_state.video_running = True

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        summary = {
            "pedestrians": 0,
            "vehicles": 0,
            "total": 0
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, summary = detect_objects(frame, summary)

            stframe.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        cap.release()

        st.markdown("## 📊 Video Detection Summary")
        st.json(summary)

        # ✅ STOP LOOPING
        st.session_state.video_running = False



st.markdown("## 📊 Detection Summary")
st.json(summary)
