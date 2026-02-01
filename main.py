import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture('traffic_video.mp4')

violated_ids = set()
current_signal = "GREEN"
car_entry_status = {} # {id: bool} Tracks if the car was already on the zebra when it turned red

# HSV Ranges
RED_LOWER1, RED_UPPER1 = np.array([0, 120, 70]), np.array([10, 255, 255])
RED_LOWER2, RED_UPPER2 = np.array([160, 120, 70]), np.array([180, 255, 255])
YELLOW_LOWER, YELLOW_UPPER = np.array([15, 100, 100]), np.array([35, 255, 255])

def get_signal_color(img):
    if img.size == 0: return "UNKNOWN"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    red_mask = cv2.add(cv2.inRange(hsv, RED_LOWER1, RED_UPPER1), cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    
    if np.count_nonzero(red_mask) > 30: return "RED"
    if np.count_nonzero(yellow_mask) > 30: return "YELLOW"
    return "GREEN"

def detect_zebra_zone(frame):
    # (Same as your previous stable detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 20))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    h, w = morphed.shape
    roi = morphed[int(h*0.5):, :] 
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.intp)
            box[:, 1] += int(h*0.5)
            return box
    return None

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    zebra_box = detect_zebra_zone(frame)
    results = model.track(frame, persist=True, conf=0.20, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        # 1. Update Signal
        for box, cls in zip(boxes, clss):
            if cls == 9:
                current_signal = get_signal_color(frame[box[1]:box[3], box[0]:box[2]])

        # 2. Check Violations
        for box, cls, obj_id in zip(boxes, clss, ids):
            if cls in [2, 3, 5, 7]:
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, y2
                
                if zebra_box is not None:
                    on_zebra = cv2.pointPolygonTest(zebra_box, (float(cx), float(cy)), False) >= 0
                    
                    # LOGIC: CAR ENTERS ZEBRA
                    if on_zebra:
                        # If the car was NOT on the zebra before, and now it is:
                        if obj_id not in car_entry_status:
                            car_entry_status[obj_id] = True # Mark as "Entered"
                            
                            # ONLY catch them if the signal is ALREADY red when they enter
                            if current_signal == "RED":
                                violated_ids.add(obj_id)
                    else:
                        # Car is not on zebra, remove from entry status to reset for next time
                        car_entry_status.pop(obj_id, None)

                # Draw
                color = (0, 0, 255) if obj_id in violated_ids else (0, 255, 0)
                label = "CAUGHT!" if obj_id in violated_ids else f"ID:{obj_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), 0, 0.6, color, 2)

    # UI
    cv2.putText(frame, f"Signal: {current_signal}", (20, 50), 0, 1, (255,255,255), 2)
    cv2.putText(frame, f"Total Violators: {len(violated_ids)}", (20, 100), 0, 1, (0,0,255), 2)

    cv2.imshow("Fair Violation System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
