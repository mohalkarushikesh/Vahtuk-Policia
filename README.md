# 🚦 Traffic Violation Detection System
**YOLOv8n + ByteTrack | MP4 Video | Python 3.9+**

Detects violations in traffic footage automatically:

| Violation | Severity |
|---|---|
| 🔴 Red-light running | HIGH |
| ⚡ Speeding (configurable limit) | HIGH / MEDIUM |
| ↩️ Wrong-way driving | HIGH |
| 🛑 Stopped in intersection | MEDIUM |

---

## ⚙️ Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
# YOLOv8n weights (~6 MB) download automatically on first run
```

---

## 🚀 Run

### Basic
```bash
python src/detector.py path/to/your_video.mp4
```

### With options
```bash
python src/detector.py traffic.mp4 \
  --conf 0.5 \
  --speed-limit 60 \
  --device cuda          # GPU (much faster)
```

| Flag | Default | Description |
|---|---|---|
| `--conf` | 0.45 | YOLO confidence threshold |
| `--speed-limit` | 50 | Speed limit in km/h |
| `--device` | cpu | `cpu`, `cuda`, or `mps` |
| `--no-zones` | off | Hide zone overlays |

### Generate HTML dashboard
```bash
python src/dashboard.py output/violation_report.json
# Opens output/dashboard.html in your browser
```

---

## 📂 Output

```
output/
├── annotated_output.mp4      # Annotated video with bounding boxes + labels
├── violation_report.json     # Machine-readable structured report
├── dashboard.html            # Interactive HTML dashboard with charts
└── snapshots/
    └── v_<frame>_<id>_<type>.jpg   # One frame capture per violation event
```

---

## 🗺️ Zone Calibration (important!)

In `src/detector.py` → `class Config`, adjust these to match your video:

```python
RED_LIGHT_ZONE = (0.3, 0.4, 0.7, 0.6)   # intersection box (fractions 0–1)
STOP_LINE_Y    = 0.55                     # horizontal stop line position
WRONG_WAY_ZONE = (0.0, 0.0, 0.4, 1.0)   # lane for wrong-way check
SPEED_LIMIT_KMH = 50
```

Use this helper to find coordinates visually:
```python
import cv2
cap = cv2.VideoCapture("your_video.mp4")
ret, frame = cap.read()
h, w = frame.shape[:2]
# Click on frame to get pixel coords, then divide by w or h
cv2.imshow("frame", frame); cv2.waitKey(0)
```

---

## 🔧 Advanced: Real Traffic Light Detection

Replace the simulated signal in `detector.py` with actual color detection:

```python
def _detect_real_red_light(self, frame, signal_roi):
    """
    signal_roi: (x1, y1, x2, y2) pixels of the traffic light crop
    """
    crop = frame[signal_roi[1]:signal_roi[3], signal_roi[0]:signal_roi[2]]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Red hue mask
    m1 = cv2.inRange(hsv, (0, 120, 100),  (10, 255, 255))
    m2 = cv2.inRange(hsv, (160, 120, 100),(179, 255, 255))
    red_pixels = cv2.countNonZero(m1 | m2)
    return red_pixels > 200   # threshold
```

---

## 📦 Requirements

- Python 3.9+
- ultralytics ≥ 8.0  
- opencv-python ≥ 4.8  
- numpy ≥ 1.24  
- GPU optional (CUDA 11.8+ for `--device cuda`)

---

## 📊 Sample Report JSON

```json
{
  "total_violations": 12,
  "by_type": {
    "Red Light Running": 4,
    "Speeding (63 km/h)": 6,
    "Stopped in Intersection": 2
  },
  "by_vehicle": { "car": 9, "truck": 3 },
  "violations": [
    {
      "frame_no": 142,
      "track_id": 3,
      "vehicle_class": "car",
      "violation_type": "Red Light Running",
      "severity": "HIGH",
      "confidence": 0.87,
      "snapshot_path": "output/snapshots/v_142_3_Red_Light_Running.jpg"
    }
  ]
}
```