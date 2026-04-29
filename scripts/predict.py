from __future__ import annotations

import argparse
from pathlib import Path

from scripts._bootstrap import bootstrap_src

bootstrap_src()

from flyinghirsch.inference.predict import predict_one
from flyinghirsch.utils.config import load_config
from flyinghirsch.utils.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    args = parser.parse_args()

    cfg = load_config(Path("configs/default.yaml"))
    paths = ProjectPaths.from_root(Path.cwd())
    pred = predict_one(cfg, artifacts_dir=paths.artifacts_dir, image_path=Path(args.image))
    print({"label": pred.label, "score": pred.score})


if __name__ == "__main__":
    main()

