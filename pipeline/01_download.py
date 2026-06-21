import os
import requests
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm

from config import IMAGES_SUBDIR, LABELS_SUBDIR, ANNOTATED_IMAGES_SUBDIR, ALFS_SUBDIR, DATA_PATH_RAW, DATA_PATH_PROCESSED, THERMAL_SUBDIR, RGB_SUBDIR

ALFS_URL = "https://zenodo.org/records/18772136/files/dataset.zip"

RGB_URL = "https://zenodo.org/records/19034999/files/images_rgb.zip"
LABEL_RGB_URL = "https://zenodo.org/records/19034999/files/labels_rgb.zip"

THERMAL_URL = "https://zenodo.org/records/19034999/files/images_thermal.zip"
LABELS_THERMAL_URL = "https://zenodo.org/records/19034999/files/labels_thermal_merged.zip"

LABELLED_IMAGES_URL = "https://zenodo.org/records/20728879/files/bambi_thermal_labelled.zip"

def download_zip(zip_url: str, save_path: Path, skip_if_exists: bool = True):
    if save_path.exists() and skip_if_exists:
        print(f"File already exists: {save_path}")
        return
    
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with (
        open(save_path, "wb") as f,
        tqdm(response.iter_content(chunk_size=8192), desc=f"Downloading {save_path.name}", total=total_size) as pbar
    ):
        for chunk in pbar:
            f.write(chunk)
            pbar.update(len(chunk))

def move_train_to_root(root: Path, use_base_folder_as_root: bool = False):
    target_parent = None

    root = root.parent if use_base_folder_as_root else root
    subfolder = (root / "data").rglob("images") if use_base_folder_as_root else root.rglob("train")

    for path in subfolder:
        if path.is_dir():
            target_parent = path.parent
            break

    if not target_parent:
        print("Could not find a 'train' folder anywhere in the directory tree.")
        return

    if target_parent == root:
        return

    print(f"Found dataset split folders inside: {target_parent}")
    for item in target_parent.iterdir():
        target_path = root / item.name
        shutil.move(str(item), str(target_path))

    keep_folders = {'images', 'labels'} if use_base_folder_as_root else {'train', 'val', 'test'}

    for item in root.iterdir():
        if item.is_dir() and item.name not in keep_folders:
            shutil.rmtree(item)
            print(f"Cleaned up empty wrapper folder: {item.name}")

def extract_zip(zip_file_path: Path, extract_to: Path, skip_if_exists: bool = True):
    images_path = extract_to.parent / "images" if zip_file_path.name.startswith("images") else extract_to.parent / "labels"
    if images_path.exists() and images_path.is_dir() and skip_if_exists:
        print(f"Directory already exists: {extract_to}")
        return

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        total_size = sum(info.file_size for info in zip_ref.infolist())

        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Extracting") as pbar:
            for info in zip_ref.infolist():
                zip_ref.extract(info, extract_to)
                pbar.update(info.file_size)
    
def download_and_extract_zip(data_dir: Path, subdir: str, image_url: str, label_url: str, skip_if_exists: bool = True):
    file_name = IMAGES_SUBDIR if label_url is not None else "data"
    os.makedirs(data_dir / subdir, exist_ok=True)
    
    uses_single_zip = label_url is None
    zip_file_path = data_dir / subdir / f"{file_name}_file.zip"
    extracted_dir_path = data_dir / subdir / f"{file_name}"

    download_zip(image_url, zip_file_path, skip_if_exists)
    extract_zip(zip_file_path, extracted_dir_path, skip_if_exists)
    move_train_to_root(extracted_dir_path, use_base_folder_as_root=uses_single_zip)
    
    if not uses_single_zip:
        label_zip_file_path = data_dir / subdir / f"{LABELS_SUBDIR}_file.zip"
        label_extracted_dir_path = data_dir / subdir / LABELS_SUBDIR

        download_zip(label_url, label_zip_file_path, skip_if_exists)
        extract_zip(label_zip_file_path, label_extracted_dir_path, skip_if_exists)
        move_train_to_root(label_extracted_dir_path)

if __name__ == "__main__":
    download_and_extract_zip(DATA_PATH_RAW, THERMAL_SUBDIR, THERMAL_URL, LABELS_THERMAL_URL)
    download_and_extract_zip(DATA_PATH_RAW, RGB_SUBDIR, RGB_URL, LABEL_RGB_URL)
    download_and_extract_zip(DATA_PATH_RAW, ALFS_SUBDIR, ALFS_URL, None)
    download_and_extract_zip(DATA_PATH_PROCESSED, ANNOTATED_IMAGES_SUBDIR, LABELLED_IMAGES_URL, None)