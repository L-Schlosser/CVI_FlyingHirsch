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
    Placeholder inference entrypoint.

    Load your model from artifacts_dir and run inference on image_path.
    """
    _ = (cfg, artifacts_dir, image_path)
    return Prediction(label="unknown", score=0.0)

