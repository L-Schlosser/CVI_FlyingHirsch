from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrackConfig:
    track_thresh: float = 0.25
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 30


class ByteTrackWrapper:
    """
    Thin wrapper around Supervision's ByteTrack to keep the rest of the codebase stable.
    """

    def __init__(self, cfg: TrackConfig | None = None) -> None:
        self.cfg = cfg or TrackConfig()
        try:
            import supervision as sv
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "Missing dependency 'supervision'. Install with: pip install -r requirements.txt"
            ) from e

        self._sv = sv
        self._tracker = sv.ByteTrack(
            track_thresh=self.cfg.track_thresh,
            match_thresh=self.cfg.match_thresh,
            track_buffer=self.cfg.track_buffer,
            frame_rate=self.cfg.frame_rate,
        )

    def update(self, *, xyxy: np.ndarray, confidence: np.ndarray, class_id: np.ndarray) -> Any:
        sv = self._sv
        det = sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=confidence.astype(np.float32),
            class_id=class_id.astype(int),
        )
        return self._tracker.update_with_detections(det)

