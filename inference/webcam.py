from pathlib import Path

from ultralytics import YOLO
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "accessedu_cricket_v1"
    / "weights"
    / "best.pt"
)
EXPECTED_CLASSES = {"No_Action", "No_Ball", "Out", "Six", "Wide"}


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Cricket model not found at {MODEL_PATH}. Train the model first or "
            "restore runs/detect/accessedu_cricket_v1/weights/best.pt."
        )

    model = YOLO(str(MODEL_PATH))
    class_names = set(model.names.values())
    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            f"{MODEL_PATH} is not the AccessEdu cricket model. "
            f"Expected classes {sorted(EXPECTED_CLASSES)}, got {sorted(class_names)}. "
            "Do not use the root-level best.pt; it is a legacy sign-language model."
        )

    print(f"Using cricket model: {MODEL_PATH}")
    print("Classes:")
    for class_id, name in model.names.items():
        print(f"{class_id}: {name}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam. Check camera permissions and connection.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read a webcam frame.")
                break

            result = model(frame, conf=0.5, verbose=False)[0]
            cv2.imshow("AccessEdu - Cricket Gesture Recognition", result.plot())

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()