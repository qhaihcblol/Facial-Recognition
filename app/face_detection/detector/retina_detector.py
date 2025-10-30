import numpy as np
import torch
import cv2
from models.face_detection.retinaface.retinaface import RetinaFace
from src.face_detection.utils.box_utils import decode, decode_landmarks, nms
from src.face_detection.layer.prior_box import PriorBox


class RetinaFaceDetector:
    def __init__(
        self, cfg, weight_path, device=None, confidence_threshold=0.5, nms_threshold=0.4
    ):
        self.cfg = cfg
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Load model
        self.model = RetinaFace(cfg).to(self.device)
        self.model.eval()

        state_dict = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded from {weight_path}")

    def preprocess(self, img: np.ndarray):
        """Chuẩn hóa ảnh đầu vào."""
        img = np.asarray(img, dtype=np.float32)
        im_height, im_width, _ = img.shape
        scale = torch.tensor(
            [im_width, im_height, im_width, im_height], dtype=torch.float32
        )

        img -= (104, 117, 123)  # BGR mean
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return tensor, scale

    def detect(self, img: np.ndarray):
        """Detect faces và landmarks."""
        img_tensor, scale = self.preprocess(img)

        with torch.no_grad():
            loc, conf, landms = self.model(img_tensor)

            # Prior boxes
            priorbox = PriorBox(self.cfg, image_size=(img.shape[0], img.shape[1]))
            priors = priorbox.generate_anchors().to(self.device)

            # Decode boxes, scores, landmarks
            boxes = decode(loc.squeeze(0), priors, self.cfg["variance"])
            boxes = boxes * scale.to(self.device)
            scores = conf.squeeze(0)[:, 1]
            landms = decode_landmarks(landms.squeeze(0), priors, self.cfg["variance"])

            # Chỉ scale theo (W, H)
            scale1 = torch.tensor([img.shape[1], img.shape[0]], dtype=torch.float32)
            landms = landms * scale1.repeat(5).to(self.device)

            # Convert sang numpy
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            landms = landms.cpu().numpy()

            # Filter theo confidence
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            if len(scores) == 0:
                return np.empty((0, 15))

            # NMS
            dets = np.hstack((boxes, scores[:, None])).astype(np.float32)
            keep = nms(dets, self.nms_threshold)
            dets = dets[keep, :]
            landms = landms[keep]

            detections = np.concatenate((dets, landms), axis=1)
            return detections
