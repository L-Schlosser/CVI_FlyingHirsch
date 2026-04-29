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
    Placeholder training entrypoint.

    Replace this with your actual training pipeline (classical CV or DL).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"todo_accuracy": 0.0}
    return TrainResult(artifacts_dir=artifacts_dir, metrics=metrics)

