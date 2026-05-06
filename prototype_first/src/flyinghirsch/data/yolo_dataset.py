from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class YoloDatasetLayout:
    """
    Expected on-disk dataset structure (images-first):

      root/
        images/{train,val,test}/...
        labels/{train,val,test}/...(.txt)
    """

    root: Path
    images_dirname: str = "images"
    labels_dirname: str = "labels"

    def images_dir(self, split: str) -> Path:
        return self.root / self.images_dirname / split

    def labels_dir(self, split: str) -> Path:
        return self.root / self.labels_dirname / split


def discover_class_count_from_labels(labels_root: Path) -> int:
    """
    Infer number of classes from YOLO txt labels by scanning all files and
    taking max(class_id)+1. If no class ids found, returns 0.
    """
    max_id = -1
    for p in labels_root.rglob("*.txt"):
        try:
            txt = p.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            txt = p.read_text(encoding="latin-1").strip()
        if not txt:
            continue
        for line in txt.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if cid > max_id:
                max_id = cid
    return max_id + 1


def write_ultralytics_dataset_yaml(
    *,
    path: Path,
    dataset_root: Path,
    class_names: list[str] | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    """
    Write an Ultralytics-compatible dataset YAML.

    If class_names is None, it will write `nc` only.
    If class_names is provided, it will write both `nc` and `names`.
    """
    import yaml

    root = dataset_root.resolve()
    payload: dict[str, object] = {"path": str(root)}
    if "train" in splits:
        payload["train"] = "images/train"
    if "val" in splits:
        payload["val"] = "images/val"
    if "test" in splits:
        payload["test"] = "images/test"

    if class_names is not None:
        payload["nc"] = len(class_names)
        payload["names"] = class_names

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_class_names_if_present(dataset_root: Path) -> list[str] | None:
    """
    Looks for a `classes.txt` at dataset_root (one class name per line).
    Returns None if not found.
    """
    p = dataset_root / "classes.txt"
    if not p.exists():
        return None
    names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return names or None

