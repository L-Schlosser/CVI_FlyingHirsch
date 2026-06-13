from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image, visualize_object_predictions
from tqdm import tqdm

from config import (
    PROFILE,
    SAHI_CONF,
    SAHI_OVERLAP,
    SAHI_SLICE_SIZE,
    SAHI_USE_CLAHE,
    TEST_SOURCE,
    best_weights,
    run_name,
    BEST_WEIGHTS,
    MODEL_NAME,
)


def preprocess_thermal(image_bgr):
    """Optional CLAHE contrast boost for faint thermal signatures."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def main():
    weights = BEST_WEIGHTS
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"Model not found: {weights}\n"
            f"Train first: python 05_train.py (PROFILE='{PROFILE}' in config.py)"
        )

    source = Path(TEST_SOURCE)
    image_paths = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source}")

    out_dir = Path("runs/detect/predict") / f"sahi_{MODEL_NAME}"
    out_dir.mkdir(parents=True, exist_ok=True)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=SAHI_CONF,
        device="cuda:0",
    )

    print(f"SAHI predict: {len(image_paths)} images")
    print(f"  slice={SAHI_SLICE_SIZE}, overlap={SAHI_OVERLAP}, conf={SAHI_CONF}")
    print(f"  CLAHE={'on' if SAHI_USE_CLAHE else 'off'}")
    print(f"  output -> {out_dir}")

    for image_path in tqdm(image_paths, desc="SAHI inference"):
        image = read_image(str(image_path))
        if SAHI_USE_CLAHE:
            image = preprocess_thermal(image)

        result = get_sliced_prediction(
            image,
            detection_model,
            slice_height=SAHI_SLICE_SIZE,
            slice_width=SAHI_SLICE_SIZE,
            overlap_height_ratio=SAHI_OVERLAP,
            overlap_width_ratio=SAHI_OVERLAP,
        )

        visualize_object_predictions(
            image=image,
            object_prediction_list=result.object_prediction_list,
            output_dir=str(out_dir),
            file_name=image_path.stem,
            export_format="jpg",
        )

    print(f"Done. Visualizations saved to {out_dir}")


if __name__ == "__main__":
    main()
