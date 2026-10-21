"""Create a clearly labeled stress set from training images.

This is not an independent accuracy test: every generated image is derived from
an image the model may have seen during training. It is only a basic robustness
check under controlled visual changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["No_Ball", "Out", "Six", "Wide"]
ORIGINAL_CLASS_IDS = {1: 0, 2: 1, 3: 2, 4: 3}


def read_labels(path: Path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Expected YOLO detection row at {path}:{line_number}")
        class_id = int(values[0])
        cx, cy, width, height = map(float, values[1:])
        if class_id == 0:
            continue
        if class_id not in ORIGINAL_CLASS_IDS:
            raise ValueError(f"Unexpected class id {class_id} in {path}:{line_number}")
        class_id = ORIGINAL_CLASS_IDS[class_id]
        rows.append((class_id, cx, cy, width, height))
    return rows


def encode_labels(rows):
    return "".join(
        f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n"
        for class_id, cx, cy, width, height in rows
    )


def affine_rows(rows, matrix, image_width, image_height):
    output = []
    for class_id, cx, cy, box_width, box_height in rows:
        x1 = (cx - box_width / 2) * image_width
        y1 = (cy - box_height / 2) * image_height
        x2 = (cx + box_width / 2) * image_width
        y2 = (cy + box_height / 2) * image_height
        corners = np.array(
            [[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32
        )
        transformed = corners @ matrix.T
        left = max(0.0, float(transformed[:, 0].min()))
        top = max(0.0, float(transformed[:, 1].min()))
        right = min(float(image_width), float(transformed[:, 0].max()))
        bottom = min(float(image_height), float(transformed[:, 1].max()))
        if right - left < 2 or bottom - top < 2:
            continue
        output.append(
            (
                class_id,
                ((left + right) / 2) / image_width,
                ((top + bottom) / 2) / image_height,
                (right - left) / image_width,
                (bottom - top) / image_height,
            )
        )
    return output


def make_variants(image, rows, rng):
    height, width = image.shape[:2]

    flipped = cv2.flip(image, 1)
    flipped_rows = [(class_id, 1 - cx, cy, box_width, box_height) for class_id, cx, cy, box_width, box_height in rows]

    exposure_alpha = float(rng.uniform(0.72, 1.28))
    exposure_beta = float(rng.uniform(-24, 24))
    exposure = cv2.convertScaleAbs(image, alpha=exposure_alpha, beta=exposure_beta)

    angle = float(rng.uniform(-12, 12))
    scale = float(rng.uniform(0.90, 1.10))
    tx = float(rng.uniform(-0.04, 0.04) * width)
    ty = float(rng.uniform(-0.04, 0.04) * height)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    rotated = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114)
    )
    rotated_rows = affine_rows(rows, matrix, width, height)

    small = cv2.resize(image, (max(1, width // 2), max(1, height // 2)), interpolation=cv2.INTER_AREA)
    low_resolution = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    low_resolution = cv2.GaussianBlur(low_resolution, (3, 3), sigmaX=0.7)

    noise = rng.normal(0, 7, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    noisy = cv2.convertScaleAbs(noisy, alpha=float(rng.uniform(0.82, 1.18)), beta=float(rng.uniform(-12, 12)))

    return [
        ("flip", flipped, flipped_rows),
        ("exposure", exposure, rows),
        ("rotate", rotated, rotated_rows),
        ("lowres_blur", low_resolution, rows),
        ("sensor_noise", noisy, rows),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dataset" / "synthetic_train_stress")
    args = parser.parse_args()
    source_images = ROOT / "dataset" / "train" / "images"
    source_labels = ROOT / "dataset" / "train" / "labels"
    output_images = args.output / "images"
    output_labels = args.output / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in source_images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise RuntimeError(f"No training images found in {source_images}")
    generated = 0
    for image_index, image_path in enumerate(images):
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        rows = read_labels(label_path)
        rng = np.random.default_rng(seed=image_index + 20261020)
        for variant, altered_image, altered_rows in make_variants(image, rows, rng):
            filename = f"{image_path.stem}.{variant}.jpg"
            if not cv2.imwrite(str(output_images / filename), altered_image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError(f"Could not write image: {output_images / filename}")
            (output_labels / f"{Path(filename).stem}.txt").write_text(
                encode_labels(altered_rows), encoding="utf-8"
            )
            generated += 1

    names = ", ".join(f"{i}: {name}" for i, name in enumerate(NAMES))
    data_yaml = (
        f'path: "{args.output.resolve().as_posix()}"\n'
        "train: images\nval: images\ntest: images\n"
        f"nc: {len(NAMES)}\nnames: {{{names}}}\n"
    )
    (args.output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    training_yaml = (
        f'path: "{ROOT.joinpath("dataset").resolve().as_posix()}"\n'
        "train:\n  - train/images\n  - synthetic_train_stress/images\n"
        "val: valid/images\ntest: test/images\n"
        f"nc: {len(NAMES)}\nnames: {{{names}}}\n"
    )
    (ROOT / "dataset" / "augmented_training.yaml").write_text(training_yaml, encoding="utf-8")
    readme = (
        "# Synthetic training-image stress set\n\n"
        f"Generated {generated} altered images from {len(images)} original training images. "
        "This is a leakage-prone robustness check, not an independent test set or a measure "
        "of expected real-world accuracy. Every source image may already have been used in training.\n"
    )
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(f"Generated {generated} images from {len(images)} training images in {args.output}")
    print(f"Dataset config: {args.output / 'data.yaml'}")
    print(f"Fine-tuning config: {ROOT / 'dataset' / 'augmented_training.yaml'}")


if __name__ == "__main__":
    main()
