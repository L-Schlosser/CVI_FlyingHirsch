from pathlib import Path
import shutil

from config import (
    ALFS_SUBDIR,
    DATA_PATH_RAW,
    IMAGES_SUBDIR,
    LABELS_SUBDIR,
    THERMAL_SUBDIR,
)

# =========================
# INFO -> takes the dataset (thermal images combined with rgb) and copies the thermal images / label combination into train/test/val 
# = it reproduces the same split as before - but with other image quality and other labels (only 1 class)
# =========================
SOURCE_LABELS_ROOT = DATA_PATH_RAW / ALFS_SUBDIR / LABELS_SUBDIR
BT_LABELS_ROOT = DATA_PATH_RAW / THERMAL_SUBDIR / LABELS_SUBDIR
BT_IMAGES_ROOT = DATA_PATH_RAW / THERMAL_SUBDIR / IMAGES_SUBDIR

BT_LABELS_OUT = BT_LABELS_ROOT
BT_IMAGES_OUT = BT_IMAGES_ROOT

SPLITS = ["train", "val", "test"]

def build_file_index(root: Path, pattern: str, splits: list[str]):
    index = {}
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for path in split_dir.rglob(pattern):
            index[path.name] = path
    return index


def main():
    bt_label_index = build_file_index(BT_LABELS_ROOT, "*.txt", SPLITS)
    bt_image_index = build_file_index(BT_IMAGES_ROOT, "*.jpg", SPLITS)

    for split in SPLITS:
        split_helper = split + "2"
        src_dir = SOURCE_LABELS_ROOT / split
        out_label_dir = BT_LABELS_OUT / split_helper
        out_img_dir = BT_IMAGES_OUT / split_helper

        out_label_dir.mkdir(parents=True, exist_ok=True)
        out_img_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"[WARN] Missing source split: {src_dir}")
            continue

        for src_label_path in src_dir.glob("*.txt"):
            filename = src_label_path.name

            bt_label_path = bt_label_index.get(filename)
            bt_img_path_jpg = bt_image_index.get(f"{src_label_path.stem}.jpg")

            # must exist in BOTH datasets, otherwise skip
            if bt_label_path is None:
                continue
            if bt_img_path_jpg is None:
                continue


            out_label_path = out_label_dir / bt_label_path.name
            out_img_path = out_img_dir / bt_img_path_jpg.name

            if bt_label_path.resolve() != out_label_path.resolve():
                shutil.copy2(bt_label_path, out_label_path)
            # copy image
            if bt_img_path_jpg.resolve() != out_img_path.resolve():
                shutil.copy2(bt_img_path_jpg, out_img_path)

        print(f"[DONE] {split}")


if __name__ == "__main__":
    main()
