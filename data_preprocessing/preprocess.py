import os
import requests
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm

from const import DATA_PATH

def remove_images_without_labels(data_dir: Path, subdir: Path) -> int:
    images_root = data_dir / subdir / "images"
    labels_root = data_dir / subdir / "labels"

    image_extensions = {".jpg", ".jpeg"}
    removed = 0

    for img_path in images_root.rglob("*"):
        if img_path.suffix.lower() not in image_extensions:
            continue

        rel_path = img_path.relative_to(images_root)
        label_path = (labels_root / rel_path).with_suffix(".txt")

        if not label_path.exists():
            img_path.unlink()
            removed += 1

    print(f"Removed {removed} images without corresponding labels.") 
    return removed

if __name__ == "__main__":
    remove_images_without_labels(DATA_PATH, "alfs_data")
    remove_images_without_labels(DATA_PATH, "rgb_data")
    remove_images_without_labels(DATA_PATH, "thermal_data")