import cv2
from app.camera_stream import CameraStream
from app.pipeline import FacePipeline


def main():
    cam = CameraStream(0)
    if not cam.is_opened():
        print("Không mở được webcam")
        return

    pipeline = FacePipeline()
    print("Face Recognition system running... Nhấn 'q' để thoát")

    while True:
        frame = cam.read()
        if frame is None:
            break

        processed = pipeline.process_frame(frame)
        cv2.imshow("Face System", processed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
