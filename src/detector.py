"""
Traffic Violation Detection System using YOLOv8n
Detects: Red-light running, wrong-way driving, speeding (estimated),
         lane violations, illegal parking, and stopped-in-intersection violations.
"""

import cv2
import numpy as np
import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class VehicleTrack:
    track_id: int
    bbox: tuple          # (x1, y1, x2, y2)
    center: tuple        # (cx, cy)
    class_name: str
    confidence: float
    speed_kmh: float = 0.0
    direction: tuple = (0, 0)
    frames_seen: int = 1
    position_history: list = field(default_factory=list)
    violations: list = field(default_factory=list)

    def update_position(self, cx, cy, fps):
        self.position_history.append((cx, cy, time.time()))
        if len(self.position_history) > 30:
            self.position_history.pop(0)
        if len(self.position_history) >= 2:
            p1 = self.position_history[-2]
            p2 = self.position_history[-1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            self.direction = (dx, dy)
            pixel_dist = np.sqrt(dx**2 + dy**2)
            # Approximate: 1 pixel ≈ 0.05 m (adjust PIXEL_TO_METER in config)
            self.speed_kmh = pixel_dist * fps * 0.05 * 3.6


@dataclass
class Violation:
    frame_no: int
    timestamp: float
    track_id: int
    vehicle_class: str
    violation_type: str
    severity: str          # LOW / MEDIUM / HIGH
    bbox: tuple
    confidence: float
    snapshot_path: str = ""


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

class Config:
    # YOLOv8n settings
    MODEL_PATH = "yolov8n.pt"          # auto-downloaded on first run
    CONFIDENCE_THRESHOLD = 0.45
    IOU_THRESHOLD = 0.45
    DEVICE = "cpu"                      # "cuda" if GPU available

    # Zone definitions (relative 0–1 coords, scaled to frame size)
    # Adjust these for your specific video / intersection layout
    RED_LIGHT_ZONE = (0.3, 0.4, 0.7, 0.6)   # (x1, y1, x2, y2) as fractions
    STOP_LINE_Y    = 0.55                     # horizontal stop line (fraction)
    WRONG_WAY_ZONE = (0.0, 0.0, 0.4, 1.0)   # left lane for oncoming check

    # Violation thresholds
    SPEED_LIMIT_KMH = 50
    STOPPED_IN_INTERSECTION_FRAMES = 45   # ~1.5 s at 30 fps
    WRONG_WAY_DX_THRESHOLD = 3           # pixels/frame rightward in left lane

    # Vehicle classes from COCO that count as "vehicles"
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

    # Output
    OUTPUT_VIDEO = "output/annotated_output.mp4"
    OUTPUT_REPORT = "output/violation_report.json"
    SNAPSHOT_DIR = "output/snapshots"
    DRAW_TRACKS = True
    DRAW_ZONES = True
    DRAW_SPEED = True


# ─────────────────────────────────────────────
#  Violation rule engine
# ─────────────────────────────────────────────

class ViolationEngine:
    def __init__(self, frame_w: int, frame_h: int, cfg: Config):
        self.fw = frame_w
        self.fh = frame_h
        self.cfg = cfg
        # Convert fractional zones to pixels
        rz = cfg.RED_LIGHT_ZONE
        self.red_zone_px = (
            int(rz[0]*frame_w), int(rz[1]*frame_h),
            int(rz[2]*frame_w), int(rz[3]*frame_h)
        )
        wz = cfg.WRONG_WAY_ZONE
        self.wrong_zone_px = (
            int(wz[0]*frame_w), int(wz[1]*frame_h),
            int(wz[2]*frame_w), int(wz[3]*frame_h)
        )
        self.stop_line_y_px = int(cfg.STOP_LINE_Y * frame_h)
        self._stopped_counters: dict[int, int] = {}

    def _in_box(self, cx, cy, box):
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    def check(self, track: VehicleTrack, frame_no: int,
              red_light_active: bool) -> list[Violation]:
        violations = []
        cx, cy = track.center
        ts = time.time()

        # ── 1. Red-light running ──────────────────────────────────
        if (red_light_active
                and self._in_box(cx, cy, self.red_zone_px)
                and cy > self.stop_line_y_px):
            violations.append(Violation(
                frame_no=frame_no, timestamp=ts,
                track_id=track.track_id,
                vehicle_class=track.class_name,
                violation_type="Red Light Running",
                severity="HIGH",
                bbox=track.bbox,
                confidence=track.confidence
            ))

        # ── 2. Speeding ──────────────────────────────────────────
        if track.speed_kmh > self.cfg.SPEED_LIMIT_KMH:
            violations.append(Violation(
                frame_no=frame_no, timestamp=ts,
                track_id=track.track_id,
                vehicle_class=track.class_name,
                violation_type=f"Speeding ({track.speed_kmh:.0f} km/h)",
                severity="HIGH" if track.speed_kmh > 80 else "MEDIUM",
                bbox=track.bbox,
                confidence=track.confidence
            ))

        # ── 3. Wrong-way driving ─────────────────────────────────
        dx, dy = track.direction
        if (self._in_box(cx, cy, self.wrong_zone_px)
                and dx > self.cfg.WRONG_WAY_DX_THRESHOLD):
            violations.append(Violation(
                frame_no=frame_no, timestamp=ts,
                track_id=track.track_id,
                vehicle_class=track.class_name,
                violation_type="Wrong-Way Driving",
                severity="HIGH",
                bbox=track.bbox,
                confidence=track.confidence
            ))

        # ── 4. Stopped in intersection ────────────────────────────
        if self._in_box(cx, cy, self.red_zone_px):
            cnt = self._stopped_counters.get(track.track_id, 0)
            if track.speed_kmh < 2:
                cnt += 1
            else:
                cnt = 0
            self._stopped_counters[track.track_id] = cnt
            if cnt == self.cfg.STOPPED_IN_INTERSECTION_FRAMES:
                violations.append(Violation(
                    frame_no=frame_no, timestamp=ts,
                    track_id=track.track_id,
                    vehicle_class=track.class_name,
                    violation_type="Stopped in Intersection",
                    severity="MEDIUM",
                    bbox=track.bbox,
                    confidence=track.confidence
                ))
        else:
            self._stopped_counters.pop(track.track_id, None)

        return violations


# ─────────────────────────────────────────────
#  Main detector
# ─────────────────────────────────────────────

class TrafficViolationDetector:
    def __init__(self, video_path: str, cfg: Optional[Config] = None):
        self.video_path = video_path
        self.cfg = cfg or Config()
        self.model = YOLO(self.cfg.MODEL_PATH)
        self.tracks: dict[int, VehicleTrack] = {}
        self.all_violations: list[Violation] = []
        os.makedirs(self.cfg.SNAPSHOT_DIR, exist_ok=True)
        os.makedirs("output", exist_ok=True)

    # ── Simulated traffic-light state ─────────────────────────────
    #   Replace with real signal detection if needed
    def _red_light_active(self, frame_no: int, fps: float) -> bool:
        cycle = int(fps * 6)        # 6-second cycle
        phase = frame_no % cycle
        return phase < int(fps * 3)  # red for first 3 s

    # ── Draw overlay ──────────────────────────────────────────────
    def _draw_overlay(self, frame, track: VehicleTrack,
                      violations_this_frame: list[Violation],
                      engine: ViolationEngine):
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        has_violation = len(violations_this_frame) > 0
        color = (0, 0, 255) if has_violation else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track.track_id} {track.class_name}"
        if self.cfg.DRAW_SPEED:
            label += f" {track.speed_kmh:.0f}km/h"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if has_violation:
            vtype = violations_this_frame[0].violation_type
            cv2.putText(frame, f"! {vtype}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if self.cfg.DRAW_TRACKS and len(track.position_history) > 1:
            pts = [(int(p[0]), int(p[1])) for p in track.position_history]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (255, 200, 0), 1)

    def _draw_zones(self, frame, engine: ViolationEngine,
                    red_active: bool):
        rz = engine.red_zone_px
        zone_color = (0, 0, 255) if red_active else (0, 255, 0)
        cv2.rectangle(frame, (rz[0], rz[1]), (rz[2], rz[3]),
                      zone_color, 2)
        cv2.putText(frame, "INTERSECTION ZONE",
                    (rz[0] + 4, rz[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone_color, 1)
        # Stop line
        cv2.line(frame, (0, engine.stop_line_y_px),
                 (frame.shape[1], engine.stop_line_y_px),
                 (0, 165, 255), 2)

    def _draw_hud(self, frame, frame_no, fps, red_active,
                  total_violations):
        signal = "RED" if red_active else "GREEN"
        sig_color = (0, 0, 255) if red_active else (0, 255, 0)
        cv2.rectangle(frame, (8, 8), (280, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {frame_no}  FPS: {fps:.1f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.putText(frame, f"Signal: {signal}",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    sig_color, 2)
        cv2.putText(frame, f"Violations: {total_violations}",
                    (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 255), 1)

    # ── Save snapshot ─────────────────────────────────────────────
    def _save_snapshot(self, frame, violation: Violation):
        fname = (f"{self.cfg.SNAPSHOT_DIR}/"
                 f"v_{violation.frame_no}_{violation.track_id}_"
                 f"{violation.violation_type.replace(' ', '_')}.jpg")
        cv2.imwrite(fname, frame)
        violation.snapshot_path = fname

    # ── Main processing loop ──────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            self.cfg.OUTPUT_VIDEO,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (fw, fh)
        )

        engine = ViolationEngine(fw, fh, self.cfg)
        frame_no = 0
        violation_set: set[tuple] = set()   # deduplicate per frame+track+type

        print(f"[INFO] Video: {fw}x{fh} @ {fps:.1f} fps  ({total_frames} frames)")
        print("[INFO] Processing…")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            red_active = self._red_light_active(frame_no, fps)

            # ── Run YOLOv8n with ByteTrack ─────────────────────────
            results = self.model.track(
                frame,
                persist=True,
                conf=self.cfg.CONFIDENCE_THRESHOLD,
                iou=self.cfg.IOU_THRESHOLD,
                device=self.cfg.DEVICE,
                classes=[2, 3, 5, 7],   # car, motorcycle, bus, truck
                verbose=False
            )

            frame_violations: list[Violation] = []

            if results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    if box.id is None:
                        continue
                    tid   = int(box.id.item())
                    cls   = int(box.cls.item())
                    conf  = float(box.conf.item())
                    cname = self.model.names[cls]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    # Update or create track
                    if tid not in self.tracks:
                        self.tracks[tid] = VehicleTrack(
                            track_id=tid, bbox=(x1,y1,x2,y2),
                            center=(cx,cy), class_name=cname,
                            confidence=conf
                        )
                    t = self.tracks[tid]
                    t.bbox = (x1, y1, x2, y2)
                    t.confidence = conf
                    t.frames_seen += 1
                    t.update_position(cx, cy, fps)
                    t.center = (cx, cy)

                    # Check violations
                    new_violations = engine.check(t, frame_no, red_active)

                    # Deduplicate (one per track+type per 30 frames)
                    deduped = []
                    for v in new_violations:
                        key = (tid, v.violation_type,
                               frame_no // 30)
                        if key not in violation_set:
                            violation_set.add(key)
                            deduped.append(v)
                            self._save_snapshot(frame, v)

                    frame_violations.extend(deduped)
                    self.all_violations.extend(deduped)

                    self._draw_overlay(frame, t, deduped, engine)

            if self.cfg.DRAW_ZONES:
                self._draw_zones(frame, engine, red_active)
            self._draw_hud(frame, frame_no, fps, red_active,
                           len(self.all_violations))

            writer.write(frame)

            if frame_no % 100 == 0:
                pct = frame_no / total_frames * 100 if total_frames else 0
                print(f"  {pct:.1f}%  frame {frame_no}  "
                      f"violations so far: {len(self.all_violations)}")

        cap.release()
        writer.release()
        self._save_report()
        print(f"\n[DONE] Annotated video → {self.cfg.OUTPUT_VIDEO}")
        print(f"       Report          → {self.cfg.OUTPUT_REPORT}")
        print(f"       Total violations detected: {len(self.all_violations)}")

    # ── JSON report ───────────────────────────────────────────────
    def _save_report(self):
        summary = {
            "total_violations": len(self.all_violations),
            "by_type": {},
            "by_vehicle": {},
            "violations": []
        }
        for v in self.all_violations:
            summary["by_type"][v.violation_type] = \
                summary["by_type"].get(v.violation_type, 0) + 1
            summary["by_vehicle"][v.vehicle_class] = \
                summary["by_vehicle"].get(v.vehicle_class, 0) + 1
            d = asdict(v)
            d["bbox"] = list(v.bbox)
            summary["violations"].append(d)

        with open(self.cfg.OUTPUT_REPORT, "w") as f:
            json.dump(summary, f, indent=2)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Traffic Violation Detector (YOLOv8n)")
    parser.add_argument("video", help="Path to input MP4 video")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="Detection confidence threshold")
    parser.add_argument("--speed-limit", type=float, default=50,
                        help="Speed limit in km/h")
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu | cuda | mps")
    parser.add_argument("--no-zones", action="store_true",
                        help="Hide zone overlays")
    args = parser.parse_args()

    cfg = Config()
    cfg.CONFIDENCE_THRESHOLD = args.conf
    cfg.SPEED_LIMIT_KMH = args.speed_limit
    cfg.DEVICE = args.device
    cfg.DRAW_ZONES = not args.no_zones

    detector = TrafficViolationDetector(args.video, cfg)
    detector.run()