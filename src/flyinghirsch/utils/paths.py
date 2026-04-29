from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str) -> Path | None:
    val = os.getenv(name)
    return Path(val) if val else None


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    artifacts_dir: Path
    reports_dir: Path

    @staticmethod
    def from_root(root: Path) -> "ProjectPaths":
        data_dir = _env_path("FLYINGHIRSCH_DATA_DIR") or (root / "data")
        artifacts_dir = _env_path("FLYINGHIRSCH_ARTIFACTS_DIR") or (root / "artifacts")
        reports_dir = root / "reports"
        return ProjectPaths(
            root=root,
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
            reports_dir=reports_dir,
        )
