import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

def _download_file(url, path):
    """Downloads a file bypassing 403 Forbidden bot-protection."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(path):
        return True
        
    print(f"[INFO] Downloading {os.path.basename(path)}...")
    try:
        # Spoof a normal Chrome browser to prevent 403 errors
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, path)
        print("[INFO] Download complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download: {e}")
        print(f"Please manually download from:\n{url}\nAnd save it to: {path}")
        return False

class FaceDetector:
    def __init__(self, max_faces=3, det_conf=0.5, track_conf=0.5):
        if not _download_file(MODEL_URL, MODEL_PATH):
            raise FileNotFoundError("Face model missing. Cannot start.")
            
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        
        faces = []
        if result.face_landmarks:
            for lm in result.face_landmarks:
                arr = np.array([[p.x, p.y, p.z] for p in lm])
                faces.append(arr)
        return faces