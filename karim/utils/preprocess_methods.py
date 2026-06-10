from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil

BLACK_PIXEL_THRESHOLD = 0.93
BLUR_THRESHOLD = 40


def normalize_thermal(img):
    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return img.astype(np.uint8)


def apply_clahe(img):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(img)


def validate_label_line(parts):

    if len(parts) != 5:
        return False

    cls = int(parts[0])
    x, y, w, h = map(float, parts[1:])

    if not (
        0 <= x <= 1 and
        0 <= y <= 1 and
        0 < w <= 1 and
        0 < h <= 1
    ):
        return False

    # only warn, don't reject
    if w < 0.01 or h < 0.01:
        print(
            f"Very small box detected "
            f"(w={w:.4f}, h={h:.4f})"
        )

    return True




def is_mostly_black(image_path: Path, threshold=BLACK_PIXEL_THRESHOLD) -> bool:
    img = cv2.imread(str(image_path))
    if img is None:
        return True  # treat unreadable images as invalid
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # count near-black pixels
    black_pixels = np.sum(gray < 10)  # intensity < 10 = almost black
    total_pixels = gray.size

    ratio = black_pixels / total_pixels
    return ratio >= threshold


def is_blurry(image_path, threshold=BLUR_THRESHOLD):
    img = cv2.imread(str(image_path))
    
    if img is None:
        return True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Laplacian variance (sharpness measure)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return variance < threshold