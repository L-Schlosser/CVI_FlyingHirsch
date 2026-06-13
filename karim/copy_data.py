from pathlib import Path
import shutil

# =========================
# INFO -> takes the dataset (thermal images combined with rgb) and copies the thermal images / label combination into train/test/val
# =========================
SOURCE_LABELS_ROOT = Path("datasets/raw_old/labels")
BT_LABELS_ALL = Path("datasets/raw/labels/all")
BT_IMAGES_ALL = Path("datasets/raw/images/all")

BT_LABELS_OUT = Path("datasets/raw/labels")
BT_IMAGES_OUT = Path("datasets/raw/images")

SPLITS = ["train2", "val2", "test2"]


def read_lines(path: Path):
    return path.read_text().strip().splitlines()


def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    for split in SPLITS:
        src_dir = SOURCE_LABELS_ROOT / split
        out_label_dir = BT_LABELS_OUT / split
        out_img_dir = BT_IMAGES_OUT / split

        out_label_dir.mkdir(parents=True, exist_ok=True)
        out_img_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"[WARN] Missing source split: {src_dir}")
            continue

        for src_label_path in src_dir.glob("*.txt"):
            filename = src_label_path.name

            bt_label_path = BT_LABELS_ALL / filename
            bt_img_path_jpg = BT_IMAGES_ALL / (src_label_path.stem + ".jpg")

            # must exist in BOTH datasets, otherwise skip
            if not bt_label_path.exists():
                continue
            if not bt_img_path_jpg.exists():
                continue

            src_lines = read_lines(src_label_path)
            bt_lines = read_lines(bt_label_path)

            # safety check: same number of boxes
            if len(src_lines) != len(bt_lines):
                print(f"[SKIP mismatch lines] {filename}")
                continue

            new_lines = []

            for src_line, bt_line in zip(src_lines, bt_lines):
                src_parts = src_line.split()
                bt_parts = bt_line.split()

                # replace class id only
                bt_parts[0] = src_parts[0]

                new_lines.append(" ".join(bt_parts))

            # write output label
            write_lines(out_label_dir / filename, new_lines)

            # shutil.copy2(bt_label_path, out_label_dir / bt_label_path.name)
            # copy image
            shutil.copy2(bt_img_path_jpg, out_img_dir / bt_img_path_jpg.name)

        print(f"[DONE] {split}")


if __name__ == "__main__":
    main()