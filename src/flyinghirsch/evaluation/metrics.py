from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    accuracy: float


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if len(y_true) == 0:
        raise ValueError("y_true must be non-empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / len(y_true)

