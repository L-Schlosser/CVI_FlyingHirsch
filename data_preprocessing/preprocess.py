from collections import defaultdict
import os
import requests
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm
import cv2
import os
import random
from pathlib import Path

from const import DATA_PATH, IMAGES_SUBDIR, LABELS_SUBDIR, KEEP_BACK_PERCENT, BLURRY_THRESHOLD

random.seed(42)

def remove_images_without_labels(data_dir: Path, subdir: Path, keep_percent: float=0.2) -> None:
    images_root = data_dir / subdir / IMAGES_SUBDIR
    labels_root = data_dir / subdir / LABELS_SUBDIR

    image_extensions = {".jpg", ".jpeg"}
    images_to_remove = []

    for img_path in images_root.rglob("*"):
        if img_path.suffix.lower() not in image_extensions:
            continue

        rel_path = img_path.relative_to(images_root)
        label_path = (labels_root / rel_path).with_suffix(".txt")

        if not label_path.exists():
            images_to_remove.append(img_path)

    random.shuffle(images_to_remove)
    total_images_without_labels = len(images_to_remove)
    num_of_images_to_remove = int(len(images_to_remove) * (1 - keep_percent))

    for i in range(num_of_images_to_remove):
        img_to_remove = images_to_remove.pop()
        img_to_remove.unlink()

    print(f"Removed {num_of_images_to_remove} images of {total_images_without_labels} without corresponding labels.") 

def _delete_image_and_label(img_path: Path, images_root: Path, labels_root: Path, dry_run=True):
    if dry_run:
        print(f"[DELETE IMG] {img_path}")
    else:
        os.remove(img_path)

    label_path = labels_root / img_path.relative_to(images_root)
    label_path = label_path.with_suffix(".txt")

    if label_path.exists():
        if dry_run:
            print(f"[DELETE LABEL] {label_path}")
        else:
            os.remove(label_path)

def _is_blurry(image_path, threshold=1.0):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return False
    
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance < (threshold * 100)

def delete_blurry_images(data_dir: Path, subdir: Path, threshold: float=1.0, dry_run: bool=True) -> None:
    images_root = data_dir / subdir / IMAGES_SUBDIR
    labels_root = data_dir / subdir / LABELS_SUBDIR
    image_extensions = {".jpg", ".jpeg"}
    
    deleted = 0
    checked = 0

    for img_path in images_root.rglob("*"):
        if img_path.suffix.lower() not in image_extensions:
            continue

        checked += 1

        if _is_blurry(img_path, threshold):
            _delete_image_and_label(img_path, images_root, labels_root, dry_run)
            deleted += 1

    print(f"Removed {deleted} blurry images out of {checked} checked. Blurry threshold: {threshold}")

def _increase_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def enhance_contrast_clahe(data_dir: Path, subdir: Path) -> None:
    images_root = data_dir / subdir / IMAGES_SUBDIR
    image_extensions = {".jpg", ".jpeg", ".png"}

    processed = 0

    for img_path in images_root.rglob("*"):
        if img_path.suffix.lower() not in image_extensions:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        enhanced = _increase_contrast(img)
        cv2.imwrite(str(img_path), enhanced)
        processed += 1

    print(f"Images processed: {processed}")

def get_flight_id(filename: str) -> str:
    return filename.split("_")[0]

def move_files(flight_set, target_split, flights, images_root, labels_root, dry_run=True):
    for flight_id in flight_set:
        for img_path in flights[flight_id]:
            rel_path = img_path.relative_to(images_root)
            label_path = (labels_root / rel_path).with_suffix(".txt")

            target_img_dir = images_root.parent / target_split / IMAGES_SUBDIR
            target_lbl_dir = labels_root.parent / target_split / LABELS_SUBDIR

            target_img_path = target_img_dir / rel_path
            target_lbl_path = target_lbl_dir / rel_path.with_suffix(".txt")

            if dry_run:
                print(f"[{target_split.upper()}] {img_path}")
            else:
                target_img_path.parent.mkdir(parents=True, exist_ok=True)
                target_lbl_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.move(str(img_path), str(target_img_path))

                if label_path.exists():
                    shutil.move(str(label_path), str(target_lbl_path))

def update_split(data_dir: Path, subdir: Path, train_percentage: float=0.75, dry_run: bool=True) -> None:
    images_root = data_dir / subdir / IMAGES_SUBDIR
    labels_root = data_dir / subdir / LABELS_SUBDIR
    image_extensions = {".jpg", ".jpeg"}

    flights = defaultdict(list)

    for img_path in images_root.rglob("*"):
        if img_path.suffix.lower() not in image_extensions:
            continue

        flight_id = get_flight_id(img_path.name)
        flights[flight_id].append(img_path)

    print(f"Found {len(flights)} flights")

    flight_ids = list(flights.keys())
    random.shuffle(flight_ids)

    split_idx = int(len(flight_ids) * train_percentage)

    train_flights = set(flight_ids[:split_idx])
    val_flights = set(flight_ids[split_idx:])

    print(f"Train flights: {len(train_flights)}")
    print(f"Val flights: {len(val_flights)}")

    move_files(train_flights, "train", flights, images_root, labels_root, dry_run)
    move_files(val_flights, "val", flights, images_root, labels_root, dry_run)

    print("\n--- SUMMARY ---")
    print("Split completed (flight-safe)")
    print(f"Mode: {'DRY RUN' if dry_run else 'MOVED'}")



def preprocess_data_before_labeling(data_dir: Path, subdir: str) -> None:
    delete_blurry_images(data_dir, subdir, BLURRY_THRESHOLD, dry_run=False)
    #remove_images_without_labels(data_dir, subdir, KEEP_BACK_PERCENT)
    #enhance_contrast_clahe(data_dir, subdir)
    #update_split(data_dir, subdir, train_percentage=0.75, dry_run=False)

def preprocess_data_after_labeling(data_dir: Path, subdir: str) -> None:
    enhance_contrast_clahe(data_dir, subdir)

if __name__ == "__main__":
    # preprocess_data(DATA_PATH, "alfs_data")
    # preprocess_data(DATA_PATH, "rgb_data")
    preprocess_data_before_labeling(DATA_PATH, "thermal_data")
    #preprocess_data_after_labeling(DATA_PATH, "thermal_data_labeled")