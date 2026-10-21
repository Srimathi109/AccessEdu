"""Prepare the combined umpire images and fine-tune the four-class classifier.

The uploaded dataset is arranged in image-classification folders, so this
script preserves that format. Existing YOLO-labeled images supplement the four
action classes. No_Action samples are excluded. The app checkpoint is never
replaced automatically.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "Cricket Umpire Action Dataset"
ORIGINAL = ROOT / "dataset"
PREPARED = ROOT / "dataset" / "umpire_classifier_four_class"
PROJECT = ROOT / "runs" / "classify"
INITIAL_WEIGHTS = PROJECT / "accessedu_cricket_four_class" / "weights" / "best.pt"
RUN_NAME = "accessedu_cricket_four_class"
NAMES = ["No_Ball", "Out", "Six", "Wide"]
ORIGINAL_CLASS_IDS = {1: "No_Ball", 2: "Out", 3: "Six", 4: "Wide"}
FOLDER_CLASS = {"noball": "No_Ball", "out": "Out", "six": "Six", "wide": "Wide"}
ADDED_SPLITS = {"train": "train", "validate": "val", "test": "test"}
ORIGINAL_SPLITS = {"train": "train", "valid": "val", "test": "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def copy_original_split(source_split: str, output_split: str) -> None:
    src_images = ORIGINAL / source_split / "images"
    src_labels = ORIGINAL / source_split / "labels"
    for image in sorted(src_images.glob("*")):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = src_labels / f"{image.stem}.txt"
        if not label.is_file():
            continue
        image_data = cv2.imread(str(image))
        if image_data is None:
            continue
        height, width = image_data.shape[:2]
        lines = label.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            if class_id == 0:
                continue
            if class_id not in ORIGINAL_CLASS_IDS:
                raise ValueError(f"Unknown class id {class_id} in {label}")
            class_name = ORIGINAL_CLASS_IDS[class_id]
            destination = PREPARED / output_split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            # The original images are mosaics with multiple differently labeled
            # umpires. Crop each labeled person so one training sample has one class.
            cx, cy, box_w, box_h = (float(value) for value in parts[1:5])
            pad_x, pad_y = box_w * width * 0.12, box_h * height * 0.12
            x1 = max(0, int(cx * width - box_w * width / 2 - pad_x))
            y1 = max(0, int(cy * height - box_h * height / 2 - pad_y))
            x2 = min(width, int(cx * width + box_w * width / 2 + pad_x))
            y2 = min(height, int(cy * height + box_h * height / 2 + pad_y))
            crop = image_data[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            name = f"original_{image.stem}_{index}.jpg"
            if not cv2.imwrite(str(destination / name), crop):
                raise OSError(f"Could not write cropped training image: {destination / name}")


def copy_added_split(source_split: str, output_split: str) -> None:
    for folder, class_name in FOLDER_CLASS.items():
        src_dir = SOURCE / source_split / folder
        if not src_dir.is_dir():
            # Dataset exports may omit a class from a split. The existing YOLO
            # set still contributes examples for that split.
            continue
        images = [
            image for image in sorted(src_dir.iterdir())
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for image in images:
            destination = PREPARED / output_split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination / f"added_{folder}_{image.name}")


def prepare_dataset() -> Path:
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Added umpire dataset not found: {SOURCE}")
    # PREPARED is a generated output folder; rebuild it so repeated runs never
    # accumulate stale crops or augmented images.
    if PREPARED.exists():
        shutil.rmtree(PREPARED)
    for source_split, output_split in ORIGINAL_SPLITS.items():
        copy_original_split(source_split, output_split)
    for source_split, output_split in ADDED_SPLITS.items():
        copy_added_split(source_split, output_split)
    return PREPARED


def main() -> None:
    data_dir = prepare_dataset()
    print(f"Prepared combined classification dataset: {data_dir}")
    print(
        "Training image counts: "
        + ", ".join(
            f"{class_name}={len(list((data_dir / 'train' / class_name).glob('*')))}"
            for class_name in NAMES
        )
    )
    print(f"Fine-tuning from the current four-class checkpoint: {INITIAL_WEIGHTS}")
    model = YOLO(str(INITIAL_WEIGHTS))
    model.train(
        data=str(data_dir),
        epochs=12,
        imgsz=224,
        batch=16,
        patience=5,
        lr0=0.0003,
        seed=42,
        deterministic=True,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=True,
    )

    candidate_path = PROJECT / RUN_NAME / "weights" / "best.pt"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Training did not produce a checkpoint: {candidate_path}")
    candidate = YOLO(str(candidate_path))
    names = [candidate.names[index] for index in sorted(candidate.names)]
    if names != NAMES:
        raise RuntimeError(f"Unexpected candidate classes: {names}")

    print(f"Candidate checkpoint: {candidate_path}")
    print("Validation results:")
    candidate.val(data=str(data_dir), split="val", imgsz=224)
    print("Held-out test results:")
    candidate.val(data=str(data_dir), split="test", imgsz=224)
    print(
        "Candidate evaluation is complete. The app checkpoint was not replaced; "
        "inspect per-class accuracy before promotion."
    )


if __name__ == "__main__":
    main()
