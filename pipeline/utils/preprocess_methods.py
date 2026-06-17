from pathlib import Path

import cv2
import numpy as np

BLACK_PIXEL_THRESHOLD = 0.93
BLUR_THRESHOLD = 40
MIN_BOX_WH = 0.015  # drop YOLO boxes narrower/shorter than this (normalized w/h)
IMAGE_SIZE = 1024


def classify_label_line(parts, min_wh=MIN_BOX_WH):
    """Classify a YOLO label line: 'keep', 'too_small', or 'invalid'."""
    if len(parts) != 5:
        return "invalid"

    try:
        int(parts[0])
        x, y, w, h = map(float, parts[1:])
    except ValueError:
        return "invalid"

    if not (
        0 <= x <= 1 and
        0 <= y <= 1 and
        0 < w <= 1 and
        0 < h <= 1
    ):
        return "invalid"

    if w < min_wh or h < min_wh:
        return "too_small"

    return "keep"


def validate_label_line(parts, min_wh=MIN_BOX_WH):
    return classify_label_line(parts, min_wh=min_wh) == "keep"


def normalize_thermal(img):
    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    return img.astype(np.uint8)


def apply_clahe(img):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(img)


def is_mostly_black(image_path: Path, threshold=BLACK_PIXEL_THRESHOLD) -> bool:
    img = cv2.imread(str(image_path))
    if img is None:
        return True  # treat unreadable images as invalid
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    black_pixels = np.sum(gray < 10)
    total_pixels = gray.size

    ratio = black_pixels / total_pixels
    return ratio >= threshold


def is_blurry(image_path, threshold=BLUR_THRESHOLD):
    img = cv2.imread(str(image_path))

    if img is None:
        return True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return variance < threshold
