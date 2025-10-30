from app.utils import draw_detections
from app.face_detection.detector.retina_detector import RetinaFaceDetector
from src.face_detection.config.retinaface_config import get_config
# từ từ sau này sẽ import tracker và recognizer
cfg = get_config("mobilenet_v2")

class FacePipeline:
    def __init__(self):

        self.detector = RetinaFaceDetector(cfg=cfg, weight_path="./src/face_detection/weights/mobilenet_v2_last.pth")
        # self.recognizer = ArcFaceRecognizer()

    def process_frame(self, frame):
        detections = self.detector.detect(frame)
        # recognized = self.recognizer.identify(tracks)
        frame = draw_detections(frame, detections)
        # frame = draw_tracks(frame, tracks)
        return frame
