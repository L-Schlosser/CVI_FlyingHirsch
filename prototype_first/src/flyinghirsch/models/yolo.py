from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class YoloTrainConfig:
    model: str = "yolov8n.pt"
    imgsz: int = 640
    epochs: int = 50
    batch: int = 16
    device: str | None = None  # e.g. "0" or "cpu"
    workers: int = 4
    seed: int = 42
    patience: int = 20
    project: str = "artifacts/yolo"
    name: str = "run"


def _get(d: dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def train_yolov8(
    cfg: dict[str, Any],
    *,
    dataset_yaml: Path,
    artifacts_dir: Path,
) -> dict[str, Any]:
    """
    Train YOLOv8 using Ultralytics. Returns a dict with useful paths + metrics.
    """
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency 'ultralytics'. Install with: pip install -r requirements.txt"
        ) from e

    train_cfg = YoloTrainConfig(
        model=str(_get(cfg, ["yolo", "model"], "yolov8n.pt")),
        imgsz=int(_get(cfg, ["yolo", "imgsz"], 640)),
        epochs=int(_get(cfg, ["yolo", "epochs"], 50)),
        batch=int(_get(cfg, ["yolo", "batch"], 16)),
        device=_get(cfg, ["yolo", "device"], None),
        workers=int(_get(cfg, ["yolo", "workers"], 4)),
        seed=int(_get(cfg, ["train", "seed"], 42)),
        patience=int(_get(cfg, ["yolo", "patience"], 20)),
        project=str(_get(cfg, ["yolo", "project"], str(artifacts_dir / "yolo"))),
        name=str(_get(cfg, ["yolo", "name"], "run")),
    )

    model = YOLO(train_cfg.model)
    results = model.train(
        data=str(dataset_yaml),
        imgsz=train_cfg.imgsz,
        epochs=train_cfg.epochs,
        batch=train_cfg.batch,
        device=train_cfg.device,
        workers=train_cfg.workers,
        seed=train_cfg.seed,
        patience=train_cfg.patience,
        project=train_cfg.project,
        name=train_cfg.name,
        exist_ok=True,
    )

    # Ultralytics returns a Results object-like; keep output stable as dict.
    out_dir = Path(train_cfg.project) / train_cfg.name
    weights_best = out_dir / "weights" / "best.pt"
    weights_last = out_dir / "weights" / "last.pt"
    metrics = getattr(results, "results_dict", None) or {}

    return {
        "run_dir": str(out_dir),
        "best_weights": str(weights_best) if weights_best.exists() else None,
        "last_weights": str(weights_last) if weights_last.exists() else None,
        "metrics": metrics,
    }


def validate_yolov8(
    *,
    weights: Path,
    dataset_yaml: Path,
    split: str = "val",
    imgsz: int = 640,
    device: str | None = None,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency 'ultralytics'. Install with: pip install -r requirements.txt"
        ) from e

    model = YOLO(str(weights))
    results = model.val(data=str(dataset_yaml), split=split, imgsz=imgsz, device=device)
    metrics = getattr(results, "results_dict", None) or {}
    return {"metrics": metrics}

