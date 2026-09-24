"""Causal three-class labels for the aligned Hybrid direction heads."""

from __future__ import annotations

import numpy as np


CLASSES = np.array([-1, 0, 1], dtype=np.int8)
TARGET_SHARES = np.full(3, 1.0 / 3.0)


def labels(move_ticks: np.ndarray, flat_threshold_ticks: float) -> np.ndarray:
    """Classify a future tick move using an inclusive flat interval."""
    move = np.asarray(move_ticks)
    return np.where(move > flat_threshold_ticks, 1,
                    np.where(move < -flat_threshold_ticks, -1, 0)).astype(np.int8)


def class_counts(y: np.ndarray) -> list[int]:
    return [int(np.count_nonzero(y == value)) for value in CLASSES]


def prior_flat_threshold(move_ticks: np.ndarray) -> float:
    """Choose the closest three-way split on the half-tick grid using past moves."""
    moves = np.asarray(move_ticks, dtype=np.float64)
    if moves.ndim != 1 or not len(moves) or not np.isfinite(moves).all():
        raise ValueError("training moves must be a nonempty finite vector")
    if np.max(np.abs(moves * 2 - np.rint(moves * 2))) > 0.01:
        raise ValueError("training moves must lie on the half-tick grid")
    candidates = np.unique(np.abs(moves))
    candidates = candidates[candidates < np.max(np.abs(moves))]
    if not len(candidates):
        raise ValueError("training moves cannot form three classes")
    choices = []
    for threshold in candidates:
        counts = np.asarray(class_counts(labels(moves, float(threshold))))
        if np.any(counts == 0):
            continue
        shares = counts / counts.sum()
        choices.append((float(np.max(np.abs(shares - TARGET_SHARES))),
                        float(np.sum(np.abs(shares - TARGET_SHARES))),
                        float(threshold)))
    if not choices:
        raise ValueError("no threshold gives three training classes")
    return min(choices)[2]
