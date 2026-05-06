from __future__ import annotations

import argparse
from pathlib import Path

from scripts._bootstrap import bootstrap_src

bootstrap_src()

from flyinghirsch.inference.predict import predict_one
from flyinghirsch.inference.video_pipeline import run_video_detection_and_tracking
from flyinghirsch.utils.config import load_config
from flyinghirsch.utils.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to an image or a video.")
    parser.add_argument("--outdir", type=str, default="reports/predictions", help="Output directory for videos/CSV/JSONL.")
    args = parser.parse_args()

    cfg = load_config(Path("configs/default.yaml"))
    paths = ProjectPaths.from_root(Path.cwd())
    in_path = Path(args.input)

    # weights resolution consistent with inference.predict_one
    run_dir = Path(cfg.get("yolo", {}).get("project", str(paths.artifacts_dir / "yolo"))) / str(
        cfg.get("yolo", {}).get("name", "run")
    )
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    weights = best if best.exists() else last

    if in_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        outdir = Path(args.outdir)
        out_video = outdir / f"{in_path.stem}_annotated.mp4"
        out_jsonl = outdir / f"{in_path.stem}_detections.jsonl"
        out_csv = outdir / f"{in_path.stem}_detections.csv"
        result = run_video_detection_and_tracking(
            cfg,
            weights=weights,
            input_path=in_path,
            output_video_path=out_video,
            output_jsonl_path=out_jsonl,
            output_csv_path=out_csv,
        )
        print(result)
    else:
        pred = predict_one(cfg, artifacts_dir=paths.artifacts_dir, image_path=in_path)
        print({"label": pred.label, "score": pred.score})


if __name__ == "__main__":
    main()

