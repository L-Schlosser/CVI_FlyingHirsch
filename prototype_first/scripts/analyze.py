from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._bootstrap import bootstrap_src

bootstrap_src()

from flyinghirsch.analysis.movement import MovementConfig, compute_movement_tables, save_metadata, save_tables
from flyinghirsch.visualization.plots import plot_spatial_heatmap, plot_speed_over_time, plot_trajectories
from flyinghirsch.utils.config import load_config
from flyinghirsch.utils.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=str, required=True, help="CSV produced by scripts.predict (video mode).")
    parser.add_argument("--outdir", type=str, default="reports/analysis", help="Output directory.")
    args = parser.parse_args()

    cfg = load_config(Path("configs/default.yaml"))
    paths = ProjectPaths.from_root(Path.cwd())

    det_path = Path(args.detections)
    outdir = Path(args.outdir)

    df = pd.read_csv(det_path)
    fps = float(cfg.get("analysis", {}).get("fps", 30.0))
    ppm = cfg.get("analysis", {}).get("pixels_per_meter", None)
    mcfg = MovementConfig(fps=fps, pixels_per_meter=float(ppm) if ppm else None)

    per_frame, per_track = compute_movement_tables(df, cfg=mcfg)
    save_tables(per_frame, per_track, out_dir=outdir)

    # plots
    plot_trajectories(per_frame, out_path=outdir / "trajectories.png")
    plot_spatial_heatmap(per_frame, out_path=outdir / "spatial_heatmap.png")
    plot_speed_over_time(per_frame, out_path=outdir / "speed_over_time.png")

    save_metadata(
        {
            "detections_csv": str(det_path),
            "fps": mcfg.fps,
            "pixels_per_meter": mcfg.pixels_per_meter,
        },
        out_dir=outdir,
    )

    print("analysis_outdir:", outdir.resolve())


if __name__ == "__main__":
    main()

