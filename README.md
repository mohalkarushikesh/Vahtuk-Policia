# WORK IN PROGRESS

# 🚦 Vāhatūk Polīsia: AI Traffic Violation Detection System

**Vahtuk Policia** is a real-time computer vision pipeline designed to monitor traffic intersections. It automatically detects traffic signals, identifies zebra crossings, and flags vehicles that violate "Stop" protocols using a fair-entry logic system.

## ✨ Key Features

* **Auto-Zebra Detection:** Uses morphological operations and Convex Hull to identify pedestrian crossings without manual ROI (Region of Interest) mapping.
* **HSV Signal Intelligence:** Employs precise color-space filtering to distinguish between **RED**, **YELLOW**, and **GREEN** lights, even in varying lighting conditions.
* **"Fair-Play" Violation Logic:** Only catches vehicles that *enter* the zebra zone after the light has turned red. Vehicles already stopped on the line or those that passed before the red light are ignored.
* **Persistent Tracking:** Uses YOLOv8 ByteTrack to maintain vehicle IDs across frames, ensuring a violator is logged even if they speed away.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Vision Core:** OpenCV
* **Inference Engine:** Ultralytics YOLOv8
* **Numeric Processing:** NumPy

---

## 🚀 How It Works

### 1. Signal State Machine

The system crops the traffic light detected by YOLO and converts it to **HSV (Hue, Saturation, Value)**.

* **Red:** Detected across two hue ranges ( and ).
* **Yellow:** Detected in the  range.
* **Green:** Detected in the  range.

### 2. Zebra Crossing Geometry

The system applies a **Morphological Closing** kernel to "glue" white stripes into a single solid parallelogram.

This creates a mathematical boundary used for the `pointPolygonTest`, which checks if a vehicle's tire (bottom-center of the bounding box) has made contact with the crossing.

### 3. Violation Trigger Logic

To prevent "fake tickets," the system follows this sequence:

1. **Monitor Entry:** Record when a vehicle ID first touches the zebra crossing.
2. **Verify Signal:** Check the `current_signal` at that exact moment of entry.
3. **Flag:** If `Signal == RED` during the transition from "Road" to "Zebra," the vehicle ID is added to the `violated_ids` set.

---

## 📦 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Vahtuk-Policia.git
cd Vahtuk-Policia

```


2. **Install dependencies:**
```bash
pip install opencv-python numpy ultralytics

```


3. **Run the detector:**
Ensure your video file is named `traffic_video.mp4` or update the path in the script.
```bash
python main.py

```



---

## 📊 Performance Tuning

* **Sensitivity:** Adjust the `conf` parameter in `model.track` (default `0.20`) to catch more pedestrians or small vehicles.
* **Zebra Detection:** If the crossing is not being detected, lower the threshold value in `detect_zebra_zone` from `150` to `120`.

---
