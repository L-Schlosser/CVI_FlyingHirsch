from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainResult:
    artifacts_dir: Path
    metrics: dict[str, float]


def train(cfg: dict[str, Any], *, artifacts_dir: Path) -> TrainResult:
    """
    Train entrypoint (YOLOv8).

    Uses an Ultralytics dataset YAML produced by `scripts/preprocess.py`
    (default: `artifacts/dataset.yaml`).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset_yaml = artifacts_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Missing dataset yaml at {dataset_yaml}. Run: python -m scripts.preprocess"
        )

    from flyinghirsch.models.yolo import train_yolov8, validate_yolov8

    train_out = train_yolov8(cfg, dataset_yaml=dataset_yaml, artifacts_dir=artifacts_dir)
    best = train_out.get("best_weights")
    metrics_any: dict[str, Any] = {"train": train_out.get("metrics", {}), "paths": train_out}

    if best:
        val_out = validate_yolov8(
            weights=Path(best),
            dataset_yaml=dataset_yaml,
            split="val",
            imgsz=int(cfg.get("yolo", {}).get("imgsz", 640)),
            device=cfg.get("yolo", {}).get("device", None),
        )
        metrics_any["val"] = val_out.get("metrics", {})

    # Flatten numeric metrics for TrainResult, keep the rest in artifacts.
    flat: dict[str, float] = {}
    for group, md in metrics_any.items():
        if isinstance(md, dict):
            for k, v in md.items():
                if isinstance(v, (int, float)):
                    flat[f"{group}.{k}"] = float(v)

    # Persist full metrics payload.
    (artifacts_dir / "train_metrics.json").write_text(
        __import__("json").dumps(metrics_any, indent=2), encoding="utf-8"
    )

    return TrainResult(artifacts_dir=artifacts_dir, metrics=flat)

