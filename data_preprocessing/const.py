from pathlib import Path

DATA_PATH = Path(__file__).parent / ".." / "datasets" / "raw"
IMAGES_SUBDIR = "images"
LABELS_SUBDIR = "labels"

KEEP_BACK_PERCENT = 0.2 # keep 20% of images without labels, remove 80%
BLURRY_THRESHOLD = 0.3 # threshold for blurry image detection, higher means more blurry images will be removed