import os
import requests
import zipfile
from pathlib import Path

ZIP_URL = "https://zenodo.org/records/18772136/files/dataset.zip"
DATA_PATH = Path(__file__).parent / ".." / "datasets" / "raw"

def download_and_extract_zip(data_dir: Path, zip_url: str):
    os.makedirs(data_dir, exist_ok=True)
    
    zip_file_path = data_dir / "downloaded_file.zip"
    extracted_dir_path = data_dir / "extracted_files"
    
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()

    with open(zip_file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded: {zip_file_path}")

    os.makedirs(extracted_dir_path, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dir_path)

    print(f"Extracted to: {extracted_dir_path}")

download_and_extract_zip(DATA_PATH, ZIP_URL)