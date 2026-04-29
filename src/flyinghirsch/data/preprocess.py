from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ResizeConfig:
    width: int
    height: int


def resize(image: np.ndarray, cfg: ResizeConfig) -> np.ndarray:
    return cv2.resize(image, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

