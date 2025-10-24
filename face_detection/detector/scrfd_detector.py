from insightface.model_zoo.scrfd import SCRFD


class SCRFDDetector:
    def __init__(self, model_path: str, ctx_id=0):
        self.model = SCRFD(model_path)
        self.model.prepare(ctx_id=ctx_id)


    def detect_faces(self, frame):
        det, kpss = self.model.detect(frame)
        results = []
        if det is not None and len(det) > 0:
            for i, bbox in enumerate(det):
                x1, y1, x2, y2, score = bbox
                results.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "score": float(score),
                        "kps": kpss[i] if kpss is not None else None,
                    }
                )
        return results
