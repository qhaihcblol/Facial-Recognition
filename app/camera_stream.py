import cv2


class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

    def is_opened(self):
        return self.cap.isOpened()

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
