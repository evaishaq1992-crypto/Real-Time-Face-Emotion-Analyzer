import cv2
import numpy as np
from dataclasses import dataclass
from utils import euclidean, get_angle

# MediaPipe 478 Landmark Indices
IDX = {
    'left_eye_ear': [33, 160, 158, 133, 153, 144],
    'right_eye_ear': [362, 385, 387, 263, 373, 380],
    'mouth': {'top': 13, 'bottom': 14, 'left': 61, 'right': 291},
    'left_eyebrow': [70, 63, 105, 66, 107],
    'right_eyebrow': [300, 293, 334, 296, 336],
    'face_width': (234, 454),
    'forehead': {'top': 10, 'left': 54, 'right': 284, 'lbrow': 70, 'rbrow': 300}
}

JAWLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

@dataclass
class Features:
    ear: float = 0.0
    mar: float = 0.0
    smile: float = 0.0
    brow_dist: float = 0.0
    fhd_var: float = 0.0
    w_h_ratio: float = 0.0
    tilt: float = 0.0

class LandmarkExtractor:
    def extract(self, landmarks, frame):
        f = Features()
        h, w = frame.shape[:2]
        
        # EAR
        f.ear = (self._calc_ear(landmarks, IDX['left_eye_ear']) + 
                 self._calc_ear(landmarks, IDX['right_eye_ear'])) / 2.0
        
        # MAR & Smile
        top, bot = landmarks[IDX['mouth']['top']], landmarks[IDX['mouth']['bottom']]
        left, right = landmarks[IDX['mouth']['left']], landmarks[IDX['mouth']['right']]
        f.mar = euclidean(top, bot) / euclidean(left, right) if euclidean(left, right) > 0 else 0
        
        fw_l, fw_r = landmarks[IDX['face_width'][0]], landmarks[IDX['face_width'][1]]
        face_w = euclidean(fw_l, fw_r)
        f.smile = euclidean(left, right) / face_w if face_w > 0 else 0
        
        # Eyebrow Distance
        lb_in, rb_in = landmarks[IDX['left_eyebrow'][-1]], landmarks[IDX['right_eyebrow'][-1]]
        f.brow_dist = euclidean(lb_in, rb_in) / face_w if face_w > 0 else 0
        
        # Face W/H Ratio & Head Tilt
        xs = landmarks[:, 0] * w
        ys = landmarks[:, 1] * h
        face_h = ys.max() - ys.min()
        f.w_h_ratio = face_w / face_h if face_h > 0 else 0
        
        le_center = landmarks[IDX['left_eye_ear'][0]]
        re_center = landmarks[IDX['right_eye_ear'][0]]
        f.tilt = get_angle(le_center, re_center)
        
        # Forehead Variance (Sobel/Laplacian)
        fh = IDX['forehead']
        x1 = int(max(0, landmarks[fh['left']][0] * w))
        x2 = int(min(w, landmarks[fh['right']][0] * w))
        y1 = int(max(0, landmarks[fh['top']][1] * h))
        y2 = int(min(h, (landmarks[fh['lbrow']][1] + landmarks[fh['rbrow']][1])/2 * h))
        
        if y2 > y1 and x2 > x1:
            roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if roi.size > 0:
                f.fhd_var = cv2.Laplacian(roi, cv2.CV_64F).var()
                
        return f

    def _calc_ear(self, lm, idx):
        p = [lm[i] for i in idx]
        v1 = euclidean(p[1], p[5])
        v2 = euclidean(p[2], p[4])
        h = euclidean(p[0], p[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0