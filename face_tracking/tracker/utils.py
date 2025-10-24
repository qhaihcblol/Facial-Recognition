import cv2


def draw_tracks(frame, tracks):
    for tr in tracks:
        x1, y1, x2, y2 = tr["bbox"]
        tid = tr["id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
    return frame
