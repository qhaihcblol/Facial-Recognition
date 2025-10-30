
import cv2
def draw_detections(original_image, detections, vis_threshold=0.5):
    """
    Draw bounding boxes and landmarks on the image.
    detections: np.ndarray [N, 15] = [x1, y1, x2, y2, score, lmk1_x, lmk1_y, ..., lmk5_x, lmk5_y]
    """
    if detections is None or len(detections) == 0:
        return original_image

    for det in detections:
        x1, y1, x2, y2 = det[0:4].astype(int)
        score = det[4]
        if score < vis_threshold:
            continue

        # Draw box
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            original_image,
            f"{score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Draw 5 landmarks
        landmarks = det[5:15].reshape((5, 2)).astype(int)
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]
        for (x, y), c in zip(landmarks, colors):
            cv2.circle(original_image, (x, y), 2, c, -1)

    return original_image
