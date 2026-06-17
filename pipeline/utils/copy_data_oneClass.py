from pathlib import Path
import shutil

# =========================
# INFO -> takes the dataset (thermal images combined with rgb) and copies the thermal images / label combination into train/test/val 
# = it reproduces the same split as before - but with other image quality and other labels (only 1 class)
# =========================
SOURCE_LABELS_ROOT = Path("datasets/raw_old/labels")
BT_LABELS_ALL = Path("datasets/raw/labels/all")
BT_IMAGES_ALL = Path("datasets/raw/images/all")

BT_LABELS_OUT = Path("datasets/raw/labels")
BT_IMAGES_OUT = Path("datasets/raw/images")

SPLITS = ["train", "val", "test"]


def read_lines(path: Path):
    return path.read_text().strip().splitlines()


def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    for split in SPLITS:
        split_helper = split + "2"
        src_dir = SOURCE_LABELS_ROOT / split
        out_label_dir = BT_LABELS_OUT / split_helper
        out_img_dir = BT_IMAGES_OUT / split_helper

        out_label_dir.mkdir(parents=True, exist_ok=True)
        out_img_dir.mkdir(parents=True, exist_ok=True)

        for src_label_path in src_dir.glob("*.txt"):
            filename = src_label_path.name

            bt_label_path = BT_LABELS_ALL / filename
            bt_img_path_jpg = BT_IMAGES_ALL / (src_label_path.stem + ".jpg")

            # must exist in BOTH datasets, otherwise skip
            if not bt_label_path.exists():
                continue
            if not bt_img_path_jpg.exists():
                continue


            shutil.copy2(bt_label_path, out_label_dir / bt_label_path.name)
            # copy image
            shutil.copy2(bt_img_path_jpg, out_img_dir / bt_img_path_jpg.name)

        print(f"[DONE] {split}")


if __name__ == "__main__":
    main()