import cv2


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Hiển thị score
        text = f"{score:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        # Vẽ keypoints
        if det["kps"] is not None:
            for x, y in det["kps"]:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

    return frame
