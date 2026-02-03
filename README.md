🚦 VisionGuard AI

Real-Time Pedestrian & Vehicle Detection and Alert System

VisionGuard AI is a deep learning–based computer vision system designed to detect pedestrians and vehicles in real-time from images and videos. The system estimates the distance of pedestrians from the camera and generates intelligent alerts when a pedestrian is detected too close, helping improve road safety and situational awareness.

🎯 Problem Statement

Road accidents involving pedestrians and vehicles are a major safety concern, especially in urban areas. Drivers often fail to notice nearby pedestrians due to blind spots, poor visibility, or delayed reaction times. VisionGuard AI aims to address this issue by providing real-time pedestrian detection and proximity-based alerts.

🧠 Solution Overview

VisionGuard AI uses a YOLO (You Only Look Once) deep learning model to detect objects in real time. It processes images and videos uploaded by the user, identifies pedestrians and vehicles, estimates pedestrian distance, and triggers alerts only when necessary using a cooldown-based alert mechanism.

✨ Key Features

📸 Image Upload Detection
🎥 Real-Time Video Detection
🧍 Pedestrian & Vehicle Identification
📏 Distance Estimation for Pedestrians
🚨 Proximity-Based Alerts
⏱ Cooldown-Based Alert System (prevents repeated alerts in videos)
📊 Detection Summary (Pedestrians, Vehicles, Total)


🛠️ Technologies Used
Python
YOLO (Ultralytics)
OpenCV
Streamlit
NumPy
PyTorch


📂 Project Structure
VisionGuard/
│
├── app.py                 # Streamlit application
├── detector/
│   ├── yolo_detector.py   # YOLO detection logic
│   ├── distance.py        # Distance estimation
│
├── utils/
│   ├── alerts.py          # Alert & cooldown logic
│   ├── summary.py         # Detection summary handling
│
├── assets/                # Images / videos for testing
├── requirements.txt
├── README.md
└── .gitignore


▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/Vinay-Partap/VisionGuard.git
cd VisionGuard

2️⃣ Create & Activate Virtual Environment
python -m venv .venv
.venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run app.py

🧪 Usage
Select Image Upload or Video Upload
Upload an image or short video
The system will detect pedestrians and vehicles
Alerts will be shown only when a pedestrian is too close


📈 Project Category
DeepTech & System-Based Project


📌 Future Enhancements
Real-time camera feed support
Sound-based alerts
Vehicle speed estimation
Lane detection integration
Deployment on edge devices


📜 License
This project is developed for academic and educational purposes.

✅ READY FOR:
Teacher evaluation
Synopsis presentation
GitHub review
Future commits