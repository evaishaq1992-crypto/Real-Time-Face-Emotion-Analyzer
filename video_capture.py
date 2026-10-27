import cv2
import threading
import time

class VideoCapture:
    """Multi-threaded webcam capture to prevent I/O blocking."""
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret = False
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            self.ret = ret
            self.frame = frame
            time.sleep(0.01) # Yield to reduce CPU load

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        if self.thread.is_alive(): self.thread.join()
        self.cap.release()