import os
import requests
import zipfile
from pathlib import Path

ZIP_URL = "https://zenodo.org/records/18772136/files/dataset.zip"
DATA_PATH = Path(__file__).parent / ".." / "datasets" / "raw"

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

def download_and_extract_zip(data_dir: Path, zip_url: str, skip_if_exists: bool = True):
    zip_file_path = data_dir / "downloaded_file.zip"
    extracted_dir_path = data_dir / "extracted_files"
    
    os.makedirs(data_dir, exist_ok=True)
    download_zip(zip_url, zip_file_path, skip_if_exists)

    os.makedirs(extracted_dir_path, exist_ok=True)
    extract_zip(zip_file_path, extracted_dir_path, skip_if_exists)

download_and_extract_zip(DATA_PATH, ZIP_URL)