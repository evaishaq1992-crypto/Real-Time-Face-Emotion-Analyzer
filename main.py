import cv2
import time
import numpy as np
from typing import Dict

from video_capture import VideoCapture
from face_detector import FaceDetector
from landmark_extractor import LandmarkExtractor
from emotion_classifier import EmotionClassifier
from utils import (draw_dark_text, draw_landmarks)

# ── Constants ─────────────────────────────────
EMO_COLORS = {
    'happy': (50, 205, 50),
    'sad': (255, 100, 100),
    'angry': (50, 50, 255),
    'surprise': (50, 255, 255),
    'neutral': (180, 180, 180)
}


class FacialAnalysisSystem:
    def __init__(self):
        self.cap = VideoCapture(src=0, width=640, height=480)
        self.detector = FaceDetector(max_faces=3)
        self.extractor = LandmarkExtractor()
        self.emotion_cls = EmotionClassifier(alpha=0.3)

        self.debug = False
        self.save_ss = False
        self.fps = 0.0
        self._fc, self._ft = 0, time.time()
        self._fi = 0

    def _get_bbox(self, lm, shape, pad=0.05):
        h, w = shape[:2]
        xs, ys = lm[:, 0] * w, lm[:, 1] * h
        x1, y1 = int(xs.min() - pad * w), int(ys.min() - pad * h)
        x2, y2 = int(xs.max() + pad * w), int(ys.max() + pad * h)
        return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

    def process(self, frame):
        self._fi += 1
        lms_list = self.detector.detect(frame)
        if not lms_list:
            return frame

        h, w = frame.shape[:2]

        for lm in lms_list:
            feat = self.extractor.extract(lm, frame)
            er = self.emotion_cls.classify(feat)

            bbox = self._get_bbox(lm, frame.shape)
            col = EMO_COLORS.get(er.name, (180, 180, 180))

            # Face box
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                col,
                2,
                cv2.LINE_AA
            )

            # Draw key landmarks
            key_indices = [33, 133, 362, 263, 1, 61]
            draw_landmarks(frame, lm, key_indices, w, h,
                           (200, 200, 200), 2)

            # Facial part labels
            labels_pos = {
                'L.Eye': 33,
                'R.Eye': 362,
                'Nose': 1,
                'Lips': 61,
                'Chin': 152
            }

            for name, i in labels_pos.items():
                lx = int(lm[i][0] * w) - 15
                ly = int(lm[i][1] * h) - 12
                draw_dark_text(
                    frame,
                    name,
                    (lx, ly),
                    0.35,
                    (180, 180, 180),
                    1,
                    0.4
                )

            # Emotion text only
            px, py = bbox[0], bbox[1] - 8
            draw_dark_text(
                frame,
                f"{er.emoji} {er.name.capitalize()}",
                (px, py),
                0.6,
                col
            )

            # Debug overlay
            if self.debug:
                dx = bbox[2] + 10
                if dx < w - 150:
                    draw_dark_text(frame, f"EAR: {feat.ear:.3f}",
                                   (dx, py), 0.4, (0, 200, 255))
                    draw_dark_text(frame, f"MAR: {feat.mar:.3f}",
                                   (dx, py - 16), 0.4, (0, 200, 255))
                    draw_dark_text(frame, f"Smile: {feat.smile:.3f}",
                                   (dx, py - 32), 0.4, (0, 200, 255))
                    draw_dark_text(frame, f"Brow: {feat.brow_dist:.3f}",
                                   (dx, py - 48), 0.4, (0, 200, 255))
                    draw_dark_text(frame, f"FHD Var: {feat.fhd:.0f}",
                                   (dx, py - 64), 0.4, (0, 200, 255))
                    draw_dark_text(frame, f"Tilt: {feat.tilt:.1f} deg",
                                   (dx, py - 80), 0.4, (0, 200, 255))

        return frame

    def run(self):
        print("[INFO] Starting System... Press 'Q' to Quit.")

        if not self.cap.cap.isOpened():
            print("[ERROR] Webcam not available.")
            return

        cv2.namedWindow("Facial Analysis Pro", cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue

                frame = self.process(frame)

                # FPS Counter
                self._fc += 1
                if time.time() - self._ft >= 1.0:
                    self.fps = self._fc / (time.time() - self._ft)
                    self._fc, self._ft = 0, time.time()

                draw_dark_text(
                    frame,
                    f"FPS: {self.fps:.1f}",
                    (10, 25),
                    0.7,
                    (0, 255, 0)
                )

                if self.debug:
                    draw_dark_text(
                        frame,
                        "[DEBUG]",
                        (10, 48),
                        0.5,
                        (0, 200, 255)
                    )

                draw_dark_text(
                    frame,
                    "D:Debug | S:Screenshot | R:Reset | Q:Quit",
                    (10, frame.shape[0] - 15),
                    0.4,
                    (100, 100, 100)
                )

                cv2.imshow("Facial Analysis Pro", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('d'):
                    self.debug = not self.debug
                elif key == ord('r'):
                    self.emotion_cls.reset()

        except KeyboardInterrupt:
            pass

        finally:
            self.cap.cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    FacialAnalysisSystem().run()