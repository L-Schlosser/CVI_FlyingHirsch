from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
PIPELINE_DIR = Path(__file__).parent

DATA_PATH = PROJECT_ROOT / "datasets"
DATA_PATH_RAW = DATA_PATH / "raw"
DATA_PATH_PROCESSED = DATA_PATH / "processed"
IMAGES_SUBDIR = "images"
LABELS_SUBDIR = "labels"

ALFS_SUBDIR = "alfs_data"
THERMAL_SUBDIR = "thermal_data"
RGB_SUBDIR = "rgb_data"
ANNOTATED_IMAGES_SUBDIR = "annotated_thermal"


KEEP_BACK_PERCENT = 0.2 # keep 20% of images without labels, remove 80%
BLURRY_THRESHOLD = 0.3 # threshold for blurry image detection, higher means more blurry images will be removed

# Training plots (val_batch0_labels.jpg, val_batch0_pred.jpg, results.png, etc.)
TRAIN_PLOTS = False  # set False to save ~1-2 min/epoch

# Validation: train at 640-800, validate at native resolution
VAL_IMGSZ = 1024
VAL_BATCH = 8

# SAHI – tuned for ~15-40 px animals in 1024x1024 thermal frames
SAHI_SLICE_SIZE = 512
SAHI_OVERLAP = 0.25
SAHI_CONF = 0.15
SAHI_USE_CLAHE = True  # contrast boost per tile at inference only

ANNOTATED_DATASET_PATH = DATA_PATH_PROCESSED / ANNOTATED_IMAGES_SUBDIR
THERMAL_DATASET_PATH = DATA_PATH_PROCESSED / THERMAL_SUBDIR
THERMAL_IMAGES_PATH = THERMAL_DATASET_PATH / IMAGES_SUBDIR
DATA_YAML = ANNOTATED_DATASET_PATH / "data.yaml"
PREDICT_SOURCE = THERMAL_IMAGES_PATH / "test2"

RUNS_DIR = PIPELINE_DIR / "runs"
YOLO_RUNS_DIR = RUNS_DIR / "detect"
VALIDATION_RUNS_DIR = YOLO_RUNS_DIR / "validate"
PREDICT_RUNS_DIR = YOLO_RUNS_DIR / "predict"
TRACK_SOURCE = THERMAL_IMAGES_PATH / "test2"
TRACK_RUNS_DIR = YOLO_RUNS_DIR / "track"
TRACKER_CONFIG = PIPELINE_DIR / "trackers" / "botsort.yaml"
YOLO_EXPERIMENT_NAME = "ir_animal_detection"
BEST_WEIGHTS_DIR = PIPELINE_DIR / "best_weights"
BEST_WEIGHTS = BEST_WEIGHTS_DIR / "best.pt"
MODEL_NAME = YOLO_EXPERIMENT_NAME
