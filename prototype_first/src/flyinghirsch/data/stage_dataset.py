from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


@dataclass(frozen=True)
class StageReport:
    src_root: Path
    dst_root: Path
    splits: tuple[str, ...]
    images_copied: int
    labels_copied: int
    missing_label_for_image: int
    missing_image_for_label: int
    unreadable_images: int
    class_histogram: dict[int, int]


def _iter_images(images_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _safe_read_class_ids(label_path: Path) -> list[int]:
    try:
        txt = label_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        txt = label_path.read_text(encoding="latin-1").strip()
    if not txt:
        return []
    ids: list[int] = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            ids.append(int(parts[0]))
        except ValueError:
            # If this isn't YOLO-format, skip counting but still allow staging.
            continue
    return ids


def stage_yolo_like_dataset(
    *,
    src_root: Path,
    dst_root: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    overwrite: bool = False,
    verify_images: bool = True,
) -> StageReport:
    """
    Stage a dataset that is already split into train/val/test with YOLO-style labels.

    Supports both common layouts:

    A) split-first:
       <src_root>/<split>/images/*
       <src_root>/<split>/labels/*.txt

    B) images-first (Ultralytics default):
       <src_root>/images/<split>/*
       <src_root>/labels/<split>/*.txt

    Output is always images-first (layout B) under dst_root:
       <dst_root>/images/<split>/*
       <dst_root>/labels/<split>/*.txt

    This is copy-only by default; it will not modify your source dataset.
    """
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    images_copied = 0
    labels_copied = 0
    missing_label_for_image = 0
    missing_image_for_label = 0
    unreadable_images = 0
    class_hist: dict[int, int] = {}

    # Auto-detect input layout.
    images_first = (src_root / images_subdir).exists() and (src_root / labels_subdir).exists()

    for split in splits:
        if images_first:
            src_images = src_root / images_subdir / split
            src_labels = src_root / labels_subdir / split
        else:
            src_images = src_root / split / images_subdir
            src_labels = src_root / split / labels_subdir
        if not src_images.exists():
            raise FileNotFoundError(f"Missing images directory: {src_images}")
        if not src_labels.exists():
            raise FileNotFoundError(f"Missing labels directory: {src_labels}")

        dst_images = dst_root / images_subdir / split
        dst_labels = dst_root / labels_subdir / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        # Copy images, track missing labels.
        for img_path in tqdm(list(_iter_images(src_images)), desc=f"staging {split} images"):
            rel = img_path.relative_to(src_images)
            out_path = dst_images / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            label_path = src_labels / rel.with_suffix(".txt")
            if not label_path.exists():
                missing_label_for_image += 1

            if out_path.exists() and not overwrite:
                continue

            shutil.copy2(img_path, out_path)
            images_copied += 1

            if verify_images:
                # Lazy import so staging works even without opencv installed.
                import cv2  # type: ignore

                img = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
                if img is None:
                    unreadable_images += 1

        # Copy labels, track missing images + class histogram.
        for lbl_path in tqdm(list(src_labels.rglob("*.txt")), desc=f"staging {split} labels"):
            rel = lbl_path.relative_to(src_labels)
            out_path = dst_labels / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            img_jpg = src_images / rel.with_suffix(".jpg")
            img_png = src_images / rel.with_suffix(".png")
            img_jpeg = src_images / rel.with_suffix(".jpeg")
            if not (img_jpg.exists() or img_png.exists() or img_jpeg.exists()):
                missing_image_for_label += 1

            for cid in _safe_read_class_ids(lbl_path):
                class_hist[cid] = class_hist.get(cid, 0) + 1

            if out_path.exists() and not overwrite:
                continue
            shutil.copy2(lbl_path, out_path)
            labels_copied += 1

    return StageReport(
        src_root=src_root,
        dst_root=dst_root,
        splits=splits,
        images_copied=images_copied,
        labels_copied=labels_copied,
        missing_label_for_image=missing_label_for_image,
        missing_image_for_label=missing_image_for_label,
        unreadable_images=unreadable_images,
        class_histogram=dict(sorted(class_hist.items(), key=lambda kv: kv[0])),
    )


def write_stage_report(report: StageReport, *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "src_root": str(report.src_root),
        "dst_root": str(report.dst_root),
        "splits": list(report.splits),
        "images_copied": report.images_copied,
        "labels_copied": report.labels_copied,
        "missing_label_for_image": report.missing_label_for_image,
        "missing_image_for_label": report.missing_image_for_label,
        "unreadable_images": report.unreadable_images,
        "class_histogram": report.class_histogram,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

