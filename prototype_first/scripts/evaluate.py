from __future__ import annotations

import argparse
from pathlib import Path

from scripts._bootstrap import bootstrap_src

bootstrap_src()

from flyinghirsch.models.yolo import validate_yolov8
from flyinghirsch.utils.config import load_config
from flyinghirsch.utils.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--weights", type=str, default=None, help="Path to weights .pt. Default: best.pt from config run.")
    args = parser.parse_args()

    cfg = load_config(Path("configs/default.yaml"))
    paths = ProjectPaths.from_root(Path.cwd())

    dataset_yaml = paths.artifacts_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing {dataset_yaml}. Run: python -m scripts.preprocess")

    if args.weights is not None:
        weights = Path(args.weights)
    else:
        run_dir = Path(cfg.get("yolo", {}).get("project", str(paths.artifacts_dir / "yolo"))) / str(
            cfg.get("yolo", {}).get("name", "run")
        )
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        weights = best if best.exists() else last

    if not weights.exists():
        raise FileNotFoundError(f"Could not find weights at {weights}. Run: python -m scripts.train")

    out = validate_yolov8(
        weights=weights,
        dataset_yaml=dataset_yaml,
        split=args.split,
        imgsz=int(cfg.get("yolo", {}).get("imgsz", 640)),
        device=cfg.get("yolo", {}).get("device", None),
    )
    print({"split": args.split, "weights": str(weights), "metrics": out.get("metrics", {})})


if __name__ == "__main__":
    main()

