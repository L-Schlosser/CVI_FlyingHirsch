from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, *, color: bool = True) -> np.ndarray:
    p = str(path)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(p, flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    return img


def imwrite(path: str | Path, image: np.ndarray) -> None:
    p = str(path)
    ok = cv2.imwrite(p, image)
    if not ok:
        raise OSError(f"Could not write image: {p}")

