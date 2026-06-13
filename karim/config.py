"""Shared training and inference settings."""

PROFILE = "balanced"  # "fast" | "balanced" | "quality"

PROFILES = {
    "fast": {
        "weights": "yolo26s.pt",
        "imgsz": 640,
        "batch": 16,
        "epochs": 80,
        "patience": 20,
        "mosaic": 0.3,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "degrees": 10.0,
        "translate": 0.05,
        "scale": 0.35,
        "close_mosaic": 10,
    },
    "balanced": {
        "weights": "yolo26m.pt",
        "imgsz": 1024,
        "batch": 4,
        "epochs": 120,
        "patience": 30,
        "mosaic": 0.5, #0.35
        "mixup": 0.05,
        "copy_paste": 0.0, #0.1
        "erasing": 0.1,
        "degrees": 15.0,
        "translate": 0.08,
        "scale": 0.4,
        "close_mosaic": 15,
    },
    "quality": {
        "weights": "yolo26l.pt",
        "imgsz": 1024,
        "batch": 5,
        "epochs": 150,
        "patience": 40,
        "mosaic": 0.4,
        "mixup": 0.05,
        "copy_paste": 0.15,
        "erasing": 0.1,
        "degrees": 15.0,
        "translate": 0.08,
        "scale": 0.45,
        "close_mosaic": 15,
    },
}

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

DATA_YAML = "data/alfs.yaml"
TEST_SOURCE = "datasets/processed/images/test2"


##BEST WEIGHTS:
BEST_WEIGHTS = "best_weights/best.pt"
MODEL_NAME = "yolo26l_annotated"

def run_name(profile: str | None = None) -> str:
    profile = profile or PROFILE
    weights = PROFILES[profile]["weights"].replace(".pt", "_pt")
    return f"{weights}_{profile}_Annotated"


def best_weights(profile: str | None = None) -> str:
    # return f"runs/detect/train/{run_name(profile)}/weights/best.pt"
    return f"best_weights/best.pt"


