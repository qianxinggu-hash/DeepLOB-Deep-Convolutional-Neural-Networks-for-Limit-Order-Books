#!/usr/bin/env python3
"""Adapt Hybrid DeepLOB probabilities to AS and GP drift inputs.

The neural model predicts down/stationary/up probabilities.  Market-making
models need a signed price move (AS) or price drift per unit time (GP).  This
module keeps those units explicit and requires calibration targets from data
strictly earlier than the replay period.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


CLASS_DOWN = 0
CLASS_STATIONARY = 1
CLASS_UP = 2


@dataclass(frozen=True)
class ProbabilityMoveCalibrator:
    """Ridge map from three class probabilities to an expected move in ticks."""

    intercept_ticks: float
    coefficients_ticks: np.ndarray
    clip_ticks: float
    ridge: float
    samples: int

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        values = _validate_probabilities(probabilities)
        prediction = self.intercept_ticks + values @ self.coefficients_ticks
        return np.clip(prediction, -self.clip_ticks, self.clip_ticks)


@dataclass(frozen=True)
class DriftForecast:
    indices: np.ndarray
    probabilities: np.ndarray
    expected_move_ticks: np.ndarray
    # Elapsed time over the preceding snapshot horizon, known at each decision.
    horizon_seconds: np.ndarray
    drift_hkd_per_second: np.ndarray


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("probabilities must have shape (samples, 3)")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = values.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("each probability row must have positive mass")
    return values / row_sums[:, None]


def fit_probability_move_calibrator(
    probabilities: np.ndarray,
    future_move_ticks: np.ndarray,
    clip_ticks: float = 2.0,
    ridge: float = 1e-3,
) -> ProbabilityMoveCalibrator:
    """Fit a stable probability-to-price calibration on past-only samples.

    The three probabilities sum to one, so the fit is expressed as a centered
    ridge regression.  The intercept is the calibration-period mean move and
    is not penalized.  This uses more information than only ``p(up)-p(down)``
    while remaining a small, auditable calibration layer.
    """

    values = _validate_probabilities(probabilities)
    target = np.asarray(future_move_ticks, dtype=np.float64)
    if target.ndim != 1 or len(target) != len(values) or len(target) < 2:
        raise ValueError("future moves must align with at least two predictions")
    if not np.all(np.isfinite(target)):
        raise ValueError("future moves must be finite")
    if clip_ticks <= 0.0 or ridge < 0.0:
        raise ValueError("clip_ticks must be positive and ridge non-negative")

    clipped = np.clip(target, -5.0 * clip_ticks, 5.0 * clip_ticks)
    probability_mean = values.mean(axis=0)
    target_mean = float(clipped.mean())
    centered_x = values - probability_mean
    centered_y = clipped - target_mean
    gram = centered_x.T @ centered_x + ridge * np.eye(3)
    coefficients = np.linalg.solve(gram, centered_x.T @ centered_y)
    # The all-ones direction is unidentified because probabilities sum to one.
    coefficients -= coefficients.mean()
    intercept = target_mean - float(probability_mean @ coefficients)
    return ProbabilityMoveCalibrator(
        intercept_ticks=intercept,
        coefficients_ticks=coefficients.astype(np.float64),
        clip_ticks=float(clip_ticks),
        ridge=float(ridge),
        samples=int(len(values)),
    )


def compact_time_to_day_ms(values: np.ndarray) -> np.ndarray:
    """Convert YYYYMMDDhhmmssfff integers to milliseconds since midnight."""

    compact = np.asarray(values, dtype=np.int64) % 1_000_000_000
    hour = compact // 10_000_000
    minute = (compact // 100_000) % 100
    second = (compact // 1_000) % 100
    millisecond = compact % 1_000
    return ((hour * 60 + minute) * 60 + second) * 1_000 + millisecond


def prediction_horizon_seconds(
    send_times: np.ndarray,
    sessions: np.ndarray,
    indices: np.ndarray,
    horizon_snapshots: int,
) -> np.ndarray:
    """Return realized future clock durations for labels and diagnostics only."""

    indices = np.asarray(indices, dtype=np.int64)
    if horizon_snapshots < 1 or indices.ndim != 1:
        raise ValueError("invalid prediction horizon or indices")
    future = indices + horizon_snapshots
    if np.any(indices < 0) or np.any(future >= len(send_times)):
        raise ValueError("prediction horizon is outside the day")
    sessions = np.asarray(sessions)
    if np.any(sessions[future] != sessions[indices]):
        raise ValueError("prediction horizon crosses a session boundary")
    milliseconds = compact_time_to_day_ms(send_times)
    duration = (milliseconds[future] - milliseconds[indices]) / 1000.0
    if np.any(duration <= 0.0):
        raise ValueError("prediction horizon must have positive clock duration")
    return duration.astype(np.float64)


def causal_prediction_horizon_seconds(
    send_times: np.ndarray,
    sessions: np.ndarray,
    indices: np.ndarray,
    horizon_snapshots: int,
) -> np.ndarray:
    """Estimate a snapshot forecast's clock horizon from past snapshots only.

    At a quote decision the arrival time of ``index + horizon_snapshots`` is
    unknown.  The elapsed time since ``index - horizon_snapshots`` is known,
    and provides a causal rate for converting expected ticks to HKD/second.
    """

    indices = np.asarray(indices, dtype=np.int64)
    sessions = np.asarray(sessions)
    if horizon_snapshots < 1 or indices.ndim != 1:
        raise ValueError("invalid prediction horizon or indices")
    if sessions.ndim != 1 or len(sessions) != len(send_times):
        raise ValueError("sessions must align with send times")
    past = indices - horizon_snapshots
    if np.any(past < 0) or np.any(indices >= len(send_times)):
        raise ValueError("prediction lacks the required past snapshots")
    segment_starts = np.maximum.accumulate(
        np.where(
            np.r_[True, sessions[1:] != sessions[:-1]],
            np.arange(len(sessions)),
            0,
        )
    )
    if np.any(past < segment_starts[indices]):
        raise ValueError("prediction history crosses a session boundary")
    milliseconds = compact_time_to_day_ms(send_times)
    duration = (milliseconds[indices] - milliseconds[past]) / 1000.0
    if np.any(duration <= 0.0):
        raise ValueError("past snapshot horizon must have positive clock duration")
    return duration.astype(np.float64)


def future_mid_move_ticks(
    book: np.ndarray,
    sessions: np.ndarray,
    indices: np.ndarray,
    horizon_snapshots: int,
    tick_hkd: float,
) -> np.ndarray:
    """Compute the executable-horizon mid-price markout used for calibration."""

    if tick_hkd <= 0.0:
        raise ValueError("tick_hkd must be positive")
    indices = np.asarray(indices, dtype=np.int64)
    future = indices + horizon_snapshots
    if np.any(future >= len(book)) or np.any(
        np.asarray(sessions)[future] != np.asarray(sessions)[indices]
    ):
        raise ValueError("future move crosses a day/session boundary")
    mid = (book[:, 0].astype(np.float64) + book[:, 2]) / 2.0
    return (mid[future] - mid[indices]) / tick_hkd


class HybridDriftAdapter:
    """Load one Hybrid DeepLOB checkpoint and emit calibrated drift forecasts."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
    ) -> None:
        # Keep calibration and time-conversion helpers importable in lightweight
        # environments. Full model/reconstruction dependencies are required
        # only when checkpoint inference is actually instantiated.
        from train_7709_hybrid_deeplob import (
            HybridDeepLOB,
            causal_features,
            transform_window,
        )
        from train_pytorch_mac import choose_device

        self._causal_features = causal_features
        self._transform_window = transform_window
        self.device = choose_device(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        preprocessing = checkpoint["preprocessing"]
        self.sequence_length = int(checkpoint["config"]["sequence_length"])
        self.horizon_snapshots = int(checkpoint["labeling"]["k"])
        self.book_mean = np.asarray(preprocessing["book_mean"], dtype=np.float32)
        self.book_std = np.asarray(preprocessing["book_std"], dtype=np.float32)
        self.auxiliary_mean = np.asarray(
            preprocessing["auxiliary_mean"], dtype=np.float32
        )
        self.auxiliary_std = np.asarray(
            preprocessing["auxiliary_std"], dtype=np.float32
        )
        self.model = HybridDeepLOB(int(preprocessing["auxiliary_features"])).to(
            self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def probabilities(
        self,
        day: Any,
        indices: np.ndarray,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Run causal Hybrid DeepLOB inference for selected prediction indices."""

        indices = np.asarray(indices, dtype=np.int64)
        if batch_size < 1 or indices.ndim != 1:
            raise ValueError("invalid batch size or indices")
        if np.any(indices < self.sequence_length - 1):
            raise ValueError("an index lacks the required input history")
        raw_book = getattr(day, "book", None)
        if raw_book is None:
            raw_book = getattr(day, "features", None)
        book = np.asarray(raw_book)
        if book.ndim != 2 or book.shape[1] != 40:
            raise ValueError("day must expose a 40-column features/book array")
        class _DayView:
            pass

        auxiliary_day = _DayView()
        auxiliary_day.features = book
        auxiliary = self._causal_features(auxiliary_day, indices)
        auxiliary = np.ascontiguousarray(
            (auxiliary - self.auxiliary_mean) / self.auxiliary_std,
            dtype=np.float32,
        )

        output: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                windows = np.stack(
                    [
                        self._transform_window(
                            book[
                                index - self.sequence_length + 1 : index + 1
                            ],
                            self.book_mean,
                            self.book_std,
                        )
                        for index in batch_indices
                    ]
                )
                book_tensor = torch.from_numpy(windows).unsqueeze(1).to(self.device)
                auxiliary_tensor = torch.from_numpy(
                    auxiliary[start : start + len(batch_indices)]
                ).to(self.device)
                logits = self.model(book_tensor, auxiliary_tensor)
                output.append(torch.softmax(logits, dim=1).cpu().numpy())
        if not output:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(output).astype(np.float64)

    def forecast(
        self,
        day: Any,
        indices: np.ndarray,
        calibrator: ProbabilityMoveCalibrator,
        tick_hkd: float,
        batch_size: int = 512,
    ) -> DriftForecast:
        probabilities = self.probabilities(day, indices, batch_size)
        expected_ticks = calibrator.transform(probabilities)
        raw_times = getattr(day, "send_times", None)
        if raw_times is None and hasattr(day, "path"):
            with np.load(Path(day.path), allow_pickle=False) as archive:
                raw_times = archive["send_times"]
        if raw_times is None:
            raise ValueError("day must expose send_times or a source npz path")
        send_times = np.asarray(raw_times)
        raw_sessions = getattr(day, "sessions", None)
        if raw_sessions is None:
            raw_sessions = getattr(day, "segments", None)
        if raw_sessions is None:
            raise ValueError("day must expose sessions or segments")
        durations = causal_prediction_horizon_seconds(
            send_times,
            np.asarray(raw_sessions),
            np.asarray(indices),
            self.horizon_snapshots,
        )
        drift = expected_ticks * tick_hkd / durations
        return DriftForecast(
            indices=np.asarray(indices, dtype=np.int64),
            probabilities=probabilities,
            expected_move_ticks=expected_ticks,
            horizon_seconds=durations,
            drift_hkd_per_second=drift,
        )
