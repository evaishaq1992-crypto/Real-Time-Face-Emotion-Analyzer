import cv2
import numpy as np
import csv
import os
import math
import time
from datetime import datetime
from typing import Tuple, List, Dict, Optional

# ══════════════════════════════════════════════
#  Math & Geometry
# ══════════════════════════════════════════════

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_angle(p1, p2):
    """Get angle in degrees between two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# ══════════════════════════════════════════════
#  Signal Smoothing
# ══════════════════════════════════════════════

class MovingAverage:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, v):
        if self.value is None: self.value = v
        else: self.value = self.alpha * v + (1 - self.alpha) * self.value
        return self.value

# ══════════════════════════════════════════════
#  CSV Logger
# ══════════════════════════════════════════════

class CSVLogger:
    def __init__(self, path="logs/data_log.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.file = open(path, 'a', newline='')
        self.writer = csv.writer(self.file)
        if os.stat(path).st_size == 0:
            self.writer.writerow(["Timestamp", "Face_ID", "Age", "Gender", "Gender_Conf", "Emotion"])

    def log(self, fid, age, gender, gconf, emotion):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.writer.writerow([ts, fid, age, gender, f"{gconf:.2f}", emotion])
        self.file.flush()

    def close(self):
        self.file.close()

# ══════════════════════════════════════════════
#  Dark Mode UI Drawing
# ══════════════════════════════════════════════

def draw_dark_text(frame, text, pos, font_scale=0.5, color=(220, 220, 220), thickness=1, bg_opacity=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 3, y - h - 3), (x + w + 3, y + 3), (30, 30, 30), -1)
    cv2.addWeighted(overlay, bg_opacity, frame, 1 - bg_opacity, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def draw_region_contours(frame, landmarks, regions, colors, w, h):
    for name, indices in regions.items():
        pts = [(int(landmarks[i][0] * w), int(landmarks[i][1] * h)) for i in indices]
        if len(pts) > 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], isClosed=(name != 'jawline'), 
                          color=colors.get(name, (100,100,100)), thickness=1, lineType=cv2.LINE_AA)

def draw_landmarks(frame, landmarks, indices, w, h, color=(0, 255, 0), r=1):
    for i in indices:
        x, y = int(landmarks[i][0] * w), int(landmarks[i][1] * h)
        cv2.circle(frame, (x, y), r, color, -1)