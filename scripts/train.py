from __future__ import annotations

from pathlib import Path

from scripts._bootstrap import bootstrap_src

bootstrap_src()

from flyinghirsch.models.train import train
from flyinghirsch.utils.config import load_config
from flyinghirsch.utils.paths import ProjectPaths


def main() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    paths = ProjectPaths.from_root(Path.cwd())
    result = train(cfg, artifacts_dir=paths.artifacts_dir)
    print("metrics:", result.metrics)
    print("artifacts_dir:", result.artifacts_dir)


if __name__ == "__main__":
    main()

