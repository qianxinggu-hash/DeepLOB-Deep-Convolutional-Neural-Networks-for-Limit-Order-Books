from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_drift_adapter import (  # noqa: E402
    HybridDriftAdapter,
    ProbabilityMoveCalibrator,
    causal_prediction_horizon_seconds,
    fit_probability_move_calibrator,
    future_mid_move_ticks,
    prediction_horizon_seconds,
)


class HybridDriftAdapterTests(unittest.TestCase):
    def test_probability_calibration_preserves_direction(self) -> None:
        probabilities = np.asarray(
            [
                [0.90, 0.05, 0.05],
                [0.80, 0.10, 0.10],
                [0.10, 0.80, 0.10],
                [0.05, 0.05, 0.90],
                [0.10, 0.10, 0.80],
            ]
        )
        future_ticks = np.asarray([-1.2, -0.8, 0.0, 1.2, 0.8])
        calibrator = fit_probability_move_calibrator(
            probabilities, future_ticks, ridge=1e-6
        )
        down, flat, up = calibrator.transform(np.eye(3))
        self.assertLess(down, flat)
        self.assertLess(flat, up)

    def test_snapshot_horizon_is_converted_to_real_seconds(self) -> None:
        times = np.asarray(
            [
                20260807093000000,
                20260807093000100,
                20260807093000350,
                20260807093000900,
            ],
            dtype=np.int64,
        )
        sessions = np.ones(4, dtype=np.int8)
        durations = prediction_horizon_seconds(
            times, sessions, np.asarray([0, 1]), horizon_snapshots=2
        )
        np.testing.assert_allclose(durations, [0.35, 0.8])

    def test_causal_horizon_and_forecast_ignore_future_timestamps(self) -> None:
        observed = np.asarray(
            [
                20260807093000000,
                20260807093000100,
                20260807093000350,
                20260807093000500,
                20260807093000900,
            ],
            dtype=np.int64,
        )
        changed_future = observed.copy()
        changed_future[3:] = [20260807093000900, 20260807093002000]
        sessions = np.ones(5, dtype=np.int8)
        indices = np.asarray([2])
        actual = prediction_horizon_seconds(
            observed, sessions, indices, horizon_snapshots=2
        )
        revised_actual = prediction_horizon_seconds(
            changed_future, sessions, indices, horizon_snapshots=2
        )
        self.assertNotEqual(float(actual[0]), float(revised_actual[0]))
        for times in (observed, changed_future):
            causal = causal_prediction_horizon_seconds(
                times, sessions, indices, horizon_snapshots=2
            )
            np.testing.assert_allclose(causal, [0.35])

        adapter = object.__new__(HybridDriftAdapter)
        adapter.horizon_snapshots = 2
        adapter.probabilities = lambda day, quote_indices, batch_size: np.asarray(
            [[0.0, 0.0, 1.0]]
        )
        calibrator = ProbabilityMoveCalibrator(
            intercept_ticks=0.0,
            coefficients_ticks=np.asarray([-1.0, 0.0, 1.0]),
            clip_ticks=2.0,
            ridge=0.0,
            samples=3,
        )
        forecasts = [
            adapter.forecast(
                SimpleNamespace(send_times=times, segments=sessions),
                indices,
                calibrator,
                tick_hkd=0.02,
            )
            for times in (observed, changed_future)
        ]
        np.testing.assert_allclose(
            forecasts[0].horizon_seconds, forecasts[1].horizon_seconds
        )
        np.testing.assert_allclose(
            forecasts[0].drift_hkd_per_second,
            forecasts[1].drift_hkd_per_second,
        )

    def test_causal_horizon_rejects_cross_session_history(self) -> None:
        times = np.asarray(
            [20260807093000000, 20260807093000100, 20260807130000000],
            dtype=np.int64,
        )
        for sessions in (np.asarray([0, 0, 1]), np.asarray([0, 1, 0])):
            with self.assertRaisesRegex(ValueError, "session boundary"):
                causal_prediction_horizon_seconds(
                    times,
                    sessions,
                    np.asarray([2]),
                    horizon_snapshots=2,
                )

    def test_future_mid_move_is_reported_in_ticks(self) -> None:
        book = np.asarray(
            [
                [10.02, 1.0, 10.00, 1.0],
                [10.04, 1.0, 10.02, 1.0],
                [10.08, 1.0, 10.06, 1.0],
            ]
        )
        move = future_mid_move_ticks(
            book,
            np.ones(3),
            np.asarray([0]),
            horizon_snapshots=2,
            tick_hkd=0.02,
        )
        np.testing.assert_allclose(move, [3.0])


if __name__ == "__main__":
    unittest.main()
