from collections import Counter, deque
from pathlib import Path

from ultralytics import YOLO
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "classify"
    / "accessedu_cricket_four_class"
    / "weights"
    / "best.pt"
)
EXPECTED_CLASSES = {"No_Ball", "Out", "Six", "Wide"}


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Cricket model not found at {MODEL_PATH}. Train the model first or "
            "restore runs/classify/accessedu_cricket_four_class/weights/best.pt."
        )

    model = YOLO(str(MODEL_PATH))
    class_names = set(model.names.values())
    if model.task != "classify" or class_names != EXPECTED_CLASSES:
        raise ValueError(
            f"{MODEL_PATH} is not the AccessEdu four-class cricket classifier. "
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

    recent_predictions = deque(maxlen=7)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read a webcam frame.")
                break

            result = model(frame, imgsz=224, verbose=False)[0]
            probabilities = result.probs
            if probabilities is not None:
                recent_predictions.append(int(probabilities.top1))
                stable_id = Counter(recent_predictions).most_common(1)[0][0]
                signal = model.names[stable_id].replace("_", " ").upper()
                label = signal
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale, thickness = 0.9, 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, scale, thickness
                )
                x = max(0, (frame.shape[1] - text_width) // 2)
                y = 42
                cv2.rectangle(
                    frame,
                    (x - 18, y - text_height - 12),
                    (x + text_width + 18, y + baseline + 8),
                    (27, 48, 38),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (x, y),
                    font,
                    scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
            cv2.imshow("AccessEdu - Cricket Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
