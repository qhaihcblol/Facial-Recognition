from face_detection.detector.scrfd_detector import SCRFDDetector
from face_detection.detector.utils import draw_detections
from face_tracking.tracker.bytetrack_tracker import ByteTrackTracker
# từ từ sau này sẽ import tracker và recognizer


class FacePipeline:
    def __init__(self):
        self.detector = SCRFDDetector(model_path="face_detection/models/scrfd.onnx")
        # self.tracker = ByteTrackTracker()
        # self.recognizer = ArcFaceRecognizer()

    def process_frame(self, frame):
        detections = self.detector.detect_faces(frame)
        # tracks = self.tracker.update(detections, frame.shape)
        # recognized = self.recognizer.identify(tracks)
        frame = draw_detections(frame, detections)
        # frame = draw_tracks(frame, tracks)
        return frame
