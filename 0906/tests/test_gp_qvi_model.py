from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "0906", ROOT / "0824"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gp_qvi_model import (  # noqa: E402
    ACTION_IMPROVE,
    GPConfig,
    GPInputs,
    infer_tick_hkd,
    solve_gp_policy,
)


class GPQVIModelTests(unittest.TestCase):
    def day(self, prices: list[tuple[float, float]]) -> SimpleNamespace:
        book = np.asarray([[ask, 1.0, bid, 1.0] for ask, bid in prices])
        return SimpleNamespace(book=book)

    def inputs(self) -> GPInputs:
        m = 6
        rho = np.full((m, m), 1.0 / (m - 1))
        np.fill_diagonal(rho, 0.0)
        intensity = np.full((2, 2, m), 0.2)
        intensity[:, 1, :] = 0.4
        return GPInputs(
            transition_probabilities=rho,
            clock_intensity_per_second=np.ones(6),
            execution_intensity_per_second={
                "through": intensity,
                "touch": intensity * 1.2,
            },
            calibration_dates=("2026-07-02",),
            calibration_reference_price_hkd=50.0,
            mean_lifecycle_seconds=3.0,
            audit={},
        )

    def test_infers_daily_tick_band(self) -> None:
        self.assertEqual(
            infer_tick_hkd(self.day([(103.55, 103.50), (103.60, 103.50)])),
            0.05,
        )
        self.assertEqual(
            infer_tick_hkd(self.day([(28.16, 28.14), (28.18, 28.14)])),
            0.02,
        )

    def test_one_tick_spread_never_uses_improved_limit_quote(self) -> None:
        config = GPConfig(time_steps=4, horizon_seconds=12.0)
        policy = solve_gp_policy(
            self.inputs(), 0.02, "touch", 0, config, market_orders_enabled=True
        )
        self.assertFalse(np.any(policy.bid_action[:, 0, :] == ACTION_IMPROVE))
        self.assertFalse(np.any(policy.ask_action[:, 0, :] == ACTION_IMPROVE))

    def test_market_impulse_reduces_extreme_inventory(self) -> None:
        config = GPConfig(
            time_steps=4,
            horizon_seconds=12.0,
            inventory_penalty_gamma=5.0,
        )
        policy = solve_gp_policy(
            self.inputs(), 0.02, "through", 0, config, market_orders_enabled=True
        )
        cap = config.max_inventory_lots
        positive_target = int(policy.impulse_target[-1, 1, -1]) - cap
        negative_target = int(policy.impulse_target[-1, 1, 0]) - cap
        self.assertLess(positive_target, cap)
        self.assertGreater(negative_target, -cap)

    def test_womo_policy_never_changes_inventory_by_impulse(self) -> None:
        config = GPConfig(time_steps=4, horizon_seconds=12.0)
        policy = solve_gp_policy(
            self.inputs(), 0.02, "through", 0, config, market_orders_enabled=False
        )
        expected = np.broadcast_to(
            np.arange(2 * config.max_inventory_lots + 1),
            policy.impulse_target.shape,
        )
        np.testing.assert_array_equal(policy.impulse_target, expected)


if __name__ == "__main__":
    unittest.main()
