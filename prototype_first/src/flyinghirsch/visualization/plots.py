from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_trajectories(per_frame: pd.DataFrame, *, out_path: Path, max_tracks: int = 50) -> None:
    """
    Plot 2D trajectories (cx,cy) for up to max_tracks tracks.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = per_frame.dropna(subset=["cx", "cy"]).copy()
    tracks = list(df["track_id"].dropna().unique())[:max_tracks]

    plt.figure(figsize=(10, 8))
    for tid in tracks:
        t = df[df["track_id"] == tid].sort_values("frame")
        plt.plot(t["cx"].to_numpy(), t["cy"].to_numpy(), linewidth=1.5, label=str(tid))
    plt.gca().invert_yaxis()  # image coordinates
    plt.title("Trajectories (image space)")
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    if len(tracks) <= 15:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_spatial_heatmap(per_frame: pd.DataFrame, *, out_path: Path, bins: int = 60) -> None:
    """
    2D histogram heatmap of centroids across all frames.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = per_frame.dropna(subset=["cx", "cy"]).copy()
    x = df["cx"].to_numpy(dtype=float)
    y = df["cy"].to_numpy(dtype=float)
    if len(x) == 0:
        return

    plt.figure(figsize=(10, 8))
    plt.hist2d(x, y, bins=bins, cmap="magma")
    plt.gca().invert_yaxis()
    plt.title("Spatial distribution heatmap (centroids)")
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_speed_over_time(per_frame: pd.DataFrame, *, out_path: Path, max_tracks: int = 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = per_frame.copy()
    tracks = list(df["track_id"].dropna().unique())[:max_tracks]

    plt.figure(figsize=(12, 6))
    for tid in tracks:
        t = df[df["track_id"] == tid].sort_values("frame")
        plt.plot(t["frame"].to_numpy(), t["speed_px_s"].to_numpy(), linewidth=1.2, label=str(tid))
    plt.title("Speed over time (px/s)")
    plt.xlabel("frame")
    plt.ylabel("speed (px/s)")
    if len(tracks) <= 10:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

