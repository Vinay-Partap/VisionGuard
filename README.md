# VisionGuard AI
### Real-Time Pedestrian & Vehicle Detection System

---

## Overview
VisionGuard AI is a real-time computer vision system designed to detect pedestrians and vehicles from images and video streams using deep learning. The system estimates pedestrian proximity and triggers alerts only when safety thresholds are breached, reducing unnecessary alert noise during continuous video processing.

---

##  Problem Statement
Pedestrian safety is a critical challenge in urban traffic environments. Traditional driver-assistance systems often fail to provide timely warnings due to poor visibility, blind spots, or delayed human response. There is a need for an automated vision-based system capable of detecting pedestrians in real time and issuing proximity-aware alerts without overwhelming the user.

---

##  Solution Overview
VisionGuard AI uses the YOLO (You Only Look Once) deep learning model to detect objects in real time. The system:
- Processes images and videos
- Identifies pedestrians and vehicles
- Estimates pedestrian distance
- Generates alerts only when required
- Prevents alert flooding in video streams

---

## Core Features
- Real-time pedestrian and vehicle detection
- Image and video input support
- Distance-based pedestrian proximity estimation
- Cooldown-controlled alert system for video streams
- Detection summary analytics
- Modular and scalable architecture  

---

## System Architecture

```text
User Input (Image / Video)
        ↓
YOLO Object Detection
        ↓
Pedestrian Distance Estimation
        ↓
Alert Decision Engine (Cooldown Logic)
        ↓
Streamlit UI + Detection Summary
```
---

## Tech Stack
- **Language:** Python
- **Deep Learning:** YOLO (Ultralytics)
- **Computer Vision:** OpenCV
- **Web Interface:** Streamlit
- **Numerical Processing:** NumPy
- **Model Backend:** PyTorch

---

## Project Structure

```text
VisionGuard/
│── app.py                 # Streamlit application
│── detector/
│   ├── yolo_detector.py   # YOLO detection logic
│   ├── distance.py        # Distance estimation
│   └── tracker.py         # Object tracking (optional)
│── utils/
│   ├── alerts.py          # Alert & cooldown logic
│   └── summary.py         # Detection summary handling
│── assets/                # Images & videos for testing
│── requirements.txt
│── README.md
│── .gitignore

```


##  How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Vinay-Partap/VisionGuard.git
cd VisionGuard
```
### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the Application
```bash
streamlit run app.py
```
## Usage
- Launch the application
- Select Image or Video input
- Upload a file
- View detections, proximity alerts, and summary metrics in real time

## Alert Logic
- Alerts are triggered only when pedestrian distance falls below a predefined threshold
- Cooldown mechanism prevents repeated alerts in video frames
- Designed for continuous video processing without alert flooding

## Project Category
DeepTech / System-Based AI Project

## Future Scope
- Live camera feed integration
- Audio-based alerts
- Vehicle speed estimation
- Edge device deployment
- Integration with smart traffic systems

## License
This project is developed for academic and research purposes.