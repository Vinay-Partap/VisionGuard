import streamlit as st
import cv2
import numpy as np
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
    st.info("Video support will be enabled in next phase.")

st.markdown("## 📊 Detection Summary")
st.json(summary)
