from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MovementConfig:
    fps: float = 30.0
    pixels_per_meter: float | None = None  # if set, converts px->meters


def _centroids(xyxy: np.ndarray) -> np.ndarray:
    # xyxy: (N,4) -> (N,2)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0], axis=1)


def compute_movement_tables(
    detections: pd.DataFrame,
    *,
    cfg: MovementConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input dataframe columns (minimum):
      frame, track_id, class_id, x1,y1,x2,y2, confidence

    Returns:
      - per-frame table with centroid + speed
      - per-track summary stats
    """
    req = {"frame", "track_id", "class_id", "x1", "y1", "x2", "y2"}
    missing = req - set(detections.columns)
    if missing:
        raise ValueError(f"detections missing columns: {sorted(missing)}")

    df = detections.copy()
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)

    # centroids
    xyxy = df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    c = _centroids(xyxy)
    df["cx"] = c[:, 0]
    df["cy"] = c[:, 1]

    # delta per track
    df["prev_cx"] = df.groupby("track_id")["cx"].shift(1)
    df["prev_cy"] = df.groupby("track_id")["cy"].shift(1)
    df["prev_frame"] = df.groupby("track_id")["frame"].shift(1)

    dx = df["cx"] - df["prev_cx"]
    dy = df["cy"] - df["prev_cy"]
    dframe = (df["frame"] - df["prev_frame"]).fillna(1.0)
    dist_px = np.sqrt(dx * dx + dy * dy)

    # speed in px/s then optionally m/s
    dt = dframe / float(cfg.fps)
    df["speed_px_s"] = (dist_px / dt).fillna(0.0)

    if cfg.pixels_per_meter:
        ppm = float(cfg.pixels_per_meter)
        df["speed_m_s"] = df["speed_px_s"] / ppm
    else:
        df["speed_m_s"] = np.nan

    summary = (
        df.groupby("track_id")
        .agg(
            class_id=("class_id", "first"),
            frames=("frame", "count"),
            mean_speed_px_s=("speed_px_s", "mean"),
            max_speed_px_s=("speed_px_s", "max"),
        )
        .reset_index()
    )
    if cfg.pixels_per_meter:
        summary["mean_speed_m_s"] = summary["mean_speed_px_s"] / float(cfg.pixels_per_meter)
        summary["max_speed_m_s"] = summary["max_speed_px_s"] / float(cfg.pixels_per_meter)
    else:
        summary["mean_speed_m_s"] = np.nan
        summary["max_speed_m_s"] = np.nan

    return df, summary


def save_tables(
    per_frame: pd.DataFrame,
    per_track: pd.DataFrame,
    *,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_frame.to_csv(out_dir / "detections_with_speed.csv", index=False)
    per_track.to_csv(out_dir / "track_summary.csv", index=False)


def save_metadata(payload: dict[str, Any], *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

