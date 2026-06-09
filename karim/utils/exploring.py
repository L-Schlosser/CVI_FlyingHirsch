from pathlib import Path
import shutil
import cv2
import numpy as np

# DATA_RAW = Path("datasets/raw")
# DATA_PROCESSED = Path("datasets/processed")
# IMAGES_DIR = "images"
# LABELS_DIR = "labels"
# SPLITS = ["train", "val", "test"]



BLACK_PIXEL_THRESHOLD = 0.93
BLUR_THRESHOLD = 40

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


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)



def is_blurry(image_path, threshold=BLUR_THRESHOLD):
    img = cv2.imread(str(image_path))
    
    if img is None:
        return True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Laplacian variance (sharpness measure)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return variance < threshold