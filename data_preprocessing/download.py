import os
import requests
import zipfile
from pathlib import Path

DATA_PATH = Path(__file__).parent / ".." / "datasets" / "raw"

ALFS_URL = "https://zenodo.org/records/18772136/files/dataset.zip"

RGB_URL = "https://zenodo.org/records/19034999/files/images_rgb.zip"
LABEL_RGB_URL = "https://zenodo.org/records/19034999/files/labels_rgb.zip"

THERMAL_URL = "https://zenodo.org/records/19034999/files/images_thermal.zip"
LABELS_THERMAL_URL = "https://zenodo.org/records/19034999/files/labels_thermal.zip"


def download_zip(zip_url: str, save_path: Path, skip_if_exists: bool = True):
    if save_path.exists() and skip_if_exists:
        print(f"File already exists: {save_path}")
        return
    
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded: {save_path}")
    
def extract_zip(zip_file_path: Path, extract_to: Path, skip_if_exists: bool = True):
    if extract_to.exists() and skip_if_exists:
        print(f"Directory already exists: {extract_to}")
        return

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to: {extract_to}")

def download_and_extract_zip(data_dir: Path, subdir: str, image_url: str, label_url: str, skip_if_exists: bool = True):
    file_name = "images" if label_url is None else "data"
    
    zip_file_path = data_dir / subdir / f"{file_name}_file.zip"
    extracted_dir_path = data_dir / subdir / f"{file_name}_files"

    os.makedirs(data_dir / subdir, exist_ok=True)
    download_zip(image_url, zip_file_path, skip_if_exists)

    os.makedirs(extracted_dir_path, exist_ok=True)
    extract_zip(zip_file_path, extracted_dir_path, skip_if_exists)
    
    if label_url:
        label_zip_file_path = data_dir / subdir / f"labels_file.zip"
        label_extracted_dir_path = data_dir / subdir / f"labels_files"

        download_zip(label_url, label_zip_file_path, skip_if_exists)
        os.makedirs(label_extracted_dir_path, exist_ok=True)
        extract_zip(label_zip_file_path, label_extracted_dir_path, skip_if_exists)

download_and_extract_zip(DATA_PATH, "alfs_data", ALFS_URL, None)
download_and_extract_zip(DATA_PATH, "rgb_data", RGB_URL, LABEL_RGB_URL)
download_and_extract_zip(DATA_PATH, "thermal_data", THERMAL_URL, LABELS_THERMAL_URL)