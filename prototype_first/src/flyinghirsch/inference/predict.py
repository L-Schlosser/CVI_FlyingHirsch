from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


def predict_one(cfg: dict[str, Any], *, artifacts_dir: Path, image_path: Path) -> Prediction:
    """
    Image inference entrypoint.

    Loads YOLO weights from `artifacts_dir/yolo/<name>/weights/best.pt` if present,
    otherwise falls back to `artifacts_dir/yolo/<name>/weights/last.pt`.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency 'ultralytics'. Install with: pip install -r requirements.txt"
        ) from e

    run_dir = Path(cfg.get("yolo", {}).get("project", str(artifacts_dir / "yolo"))) / str(
        cfg.get("yolo", {}).get("name", "run")
    )
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    weights = best if best.exists() else last
    if not weights.exists():
        raise FileNotFoundError(
            f"Could not find YOLO weights at {best} or {last}. Run: python -m scripts.train"
        )

    model = YOLO(str(weights))
    res = model.predict(
        source=str(image_path),
        imgsz=int(cfg.get("yolo", {}).get("imgsz", 640)),
        conf=float(cfg.get("inference", {}).get("score_threshold", 0.5)),
        device=cfg.get("yolo", {}).get("device", None),
        verbose=False,
    )[0]

    # For a single image, return the highest-confidence detection as a simple Prediction.
    if res.boxes is None or len(res.boxes) == 0:
        return Prediction(label="none", score=0.0)

    conf = res.boxes.conf.detach().cpu().numpy()
    cls = res.boxes.cls.detach().cpu().numpy().astype(int)
    i = int(conf.argmax())
    class_id = int(cls[i])
    score = float(conf[i])
    name_map = getattr(res, "names", None) or {}
    label = str(name_map.get(class_id, f"class_{class_id}"))
    return Prediction(label=label, score=score)

