from pathlib import Path

def check(img_dir, lbl_dir):
    
    imgs = list(Path(img_dir).glob("*.jpg"))
    lbls = list(Path(lbl_dir).glob("*.txt"))

    print("Images:", len(imgs))
    print("Labels:", len(lbls))

    missing = []
    for img in imgs:
        label_path = Path(lbl_dir) / (img.stem + ".txt")
        if not label_path.exists():
            missing.append(img.name)

    print("Missing labels:", len(missing))
    if missing[:5]:
        print("Examples:", missing[:5])


if __name__ == "__main__":
    print("raw:")
    check(
        "datasets/raw/images/train",
        "datasets/raw/labels/train"  
    )
    print("processed:")
    check(
        "datasets/processed/images/train",
        "datasets/processed/labels/train"  
    )