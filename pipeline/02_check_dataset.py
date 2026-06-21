from pathlib import Path

from config import (
    ANNOTATED_DATASET_PATH,
    ALFS_SUBDIR,
    DATA_PATH_RAW,
    IMAGES_SUBDIR,
    LABELS_SUBDIR,
    RGB_SUBDIR,
    THERMAL_SUBDIR,
)

def check(img_dir: Path, lbl_dir: Path):
    imgs = list(img_dir.glob("*.jpg"))
    lbls = list(lbl_dir.glob("*.txt"))

    print("Images:", len(imgs))
    print("Labels:", len(lbls))

    missing = []
    for img in imgs:
        label_path = lbl_dir / f"{img.stem}.txt"
        if not label_path.exists():
            missing.append(img.name)

    print("Missing labels:", len(missing))
    if missing[:5]:
        print("Examples:", missing[:5])


if __name__ == "__main__":
    print("ALFS:")
    check(
        DATA_PATH_RAW / ALFS_SUBDIR / IMAGES_SUBDIR / "train",
        DATA_PATH_RAW / ALFS_SUBDIR / LABELS_SUBDIR / "train",
    )
    
    print("\nThermal:")
    check(
        DATA_PATH_RAW / THERMAL_SUBDIR / IMAGES_SUBDIR / "train",
        DATA_PATH_RAW / THERMAL_SUBDIR / LABELS_SUBDIR / "train",
    )

    print("\nRGB:")
    check(
        DATA_PATH_RAW / RGB_SUBDIR / IMAGES_SUBDIR / "train",
        DATA_PATH_RAW / RGB_SUBDIR / LABELS_SUBDIR / "train",
    )

    print("\nAnnotated:")
    check(
        ANNOTATED_DATASET_PATH / IMAGES_SUBDIR / "train",
        ANNOTATED_DATASET_PATH / LABELS_SUBDIR / "train",
    )
    
