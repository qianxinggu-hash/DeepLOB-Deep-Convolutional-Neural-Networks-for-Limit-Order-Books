from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "0906", ROOT / "0824"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gp_qvi_model import (  # noqa: E402
    ACTION_BEST,
    ACTION_IMPROVE,
    ACTION_NONE,
    ALL_IN_FEE_RATE,
    GPConfig,
    GPInputs,
    LOT_SIZE,
    TradeData,
    assign_gp_drift_states,
    fit_gp_drift_regime,
    infer_tick_hkd,
    make_gp_drift_regime,
    rescale_gp_drift_regime,
    solve_gp_drift_policy,
    solve_gp_policy,
    simulate_gp_day,
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

    def test_segment_flatten_closes_inventory_before_feed_gap(self) -> None:
        day = SimpleNamespace(
            date="2026-07-03",
            book=np.asarray([
                [28.16, 1, 28.14, 1],
                [28.16, 1, 28.14, 1],
                [28.14, 1, 28.12, 1],
                [28.18, 1, 28.16, 1],
                [28.18, 1, 28.14, 1],
            ], dtype=np.float32),
            send_times=np.asarray([
                20260703093000000, 20260703093001000,
                20260703093002000, 20260703093500000,
                20260703093501000,
            ], dtype=np.int64),
            segments=np.asarray([0, 0, 0, 1, 1], dtype=np.int16),
            eligible=np.asarray([1, 3], dtype=np.int64),
            metadata={"gap_events": [{
                "previous_send_time": 20260703093003000,
                "resume_send_time": 20260703093459000,
            }]},
        )
        trades = TradeData(
            trade_time_ms=np.empty(0, dtype=np.int64),
            send_time_ms=np.empty(0, dtype=np.int64),
            price_hkd=np.empty(0), quantity=np.empty(0), metadata={},
        )
        config = GPConfig(quote_horizon_snapshots=1)
        carry = simulate_gp_day(day, trades, None, config, "touch", "test")
        flat = simulate_gp_day(
            day, trades, None, config, "touch", "test",
            flatten_on_segment_change=True,
        )
        self.assertEqual(carry["segment_flatten_orders"], 0)
        self.assertEqual(carry["terminal_flatten_orders"], 1)
        self.assertEqual(flat["segment_flatten_orders"], 1)
        self.assertEqual(flat["segment_flatten_lots"], 1)
        self.assertEqual(flat["terminal_flatten_orders"], 0)
        self.assertEqual(flat["end_inventory_lots"], 0)

    def test_lunch_segment_does_not_trigger_gap_flatten(self) -> None:
        day = SimpleNamespace(
            date="2026-07-03",
            book=np.asarray([
                [28.16, 1, 28.14, 1],
                [28.16, 1, 28.14, 1],
                [28.14, 1, 28.12, 1],
                [28.18, 1, 28.16, 1],
                [28.18, 1, 28.14, 1],
            ], dtype=np.float32),
            send_times=np.asarray([
                20260703115900000, 20260703115901000,
                20260703115902000, 20260703130000000,
                20260703130001000,
            ], dtype=np.int64),
            segments=np.asarray([0, 0, 0, 1, 1], dtype=np.int16),
            eligible=np.asarray([1, 3], dtype=np.int64),
            metadata={"gap_events": []},
        )
        trades = TradeData(
            trade_time_ms=np.empty(0, dtype=np.int64),
            send_time_ms=np.empty(0, dtype=np.int64),
            price_hkd=np.empty(0), quantity=np.empty(0), metadata={},
        )
        config = GPConfig(quote_horizon_snapshots=1)
        carry = simulate_gp_day(day, trades, None, config, "touch", "test")
        flat = simulate_gp_day(
            day, trades, None, config, "touch", "test",
            flatten_on_segment_change=True,
        )
        self.assertEqual(flat["segment_flatten_orders"], 0)
        self.assertEqual(flat["terminal_flatten_orders"], 1)
        self.assertEqual(flat["net_pnl_hkd"], carry["net_pnl_hkd"])

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

    def test_zero_drift_policy_matches_martingale_gp(self) -> None:
        config = GPConfig(time_steps=4, horizon_seconds=12.0)
        baseline = solve_gp_policy(
            self.inputs(), 0.02, "through", 0, config, market_orders_enabled=True
        )
        regime = make_gp_drift_regime([0.0], np.ones((1, 1)), ["neutral"])
        directional = solve_gp_drift_policy(
            self.inputs(), 0.02, "through", 0, config, regime,
            market_orders_enabled=True,
        )
        np.testing.assert_allclose(directional.values[:, 0], baseline.values)
        np.testing.assert_array_equal(
            directional.bid_action[:, 0], baseline.bid_action
        )
        np.testing.assert_array_equal(
            directional.ask_action[:, 0], baseline.ask_action
        )
        np.testing.assert_array_equal(
            directional.impulse_target[:, 0], baseline.impulse_target
        )

    def test_zero_maker_fill_alpha_preserves_existing_solvers(self) -> None:
        inputs = self.inputs()
        with_zero_alpha = replace(
            inputs,
            maker_fill_alpha_hkd_per_lot={
                "through": np.zeros((2, 2, 6)),
            },
        )
        config = GPConfig(time_steps=2, horizon_seconds=6.0)
        regime = make_gp_drift_regime([0.01], np.ones((1, 1)))
        for solve, extra in (
            (solve_gp_policy, {}),
            (solve_gp_drift_policy, {"drift_regime": regime}),
        ):
            baseline = solve(inputs, 0.02, "through", 0, config, **extra)
            zero_alpha = solve(
                with_zero_alpha, 0.02, "through", 0, config, **extra
            )
            np.testing.assert_array_equal(zero_alpha.values, baseline.values)
            np.testing.assert_array_equal(zero_alpha.bid_action, baseline.bid_action)
            np.testing.assert_array_equal(zero_alpha.ask_action, baseline.ask_action)
            np.testing.assert_array_equal(
                zero_alpha.impulse_target, baseline.impulse_target
            )

    def test_maker_fill_alpha_affects_only_selected_fill_cell(self) -> None:
        """The extra PnL belongs to the selected side, action and spread."""
        hazard = 0.2
        dt = 3.0
        tick = 0.02
        reference_price = 50.0
        alpha = np.zeros((2, 2, 2))
        alpha[0, 1, 1] = 20.0  # buy, improve, two-tick spread
        intensity = np.zeros((2, 2, 2))
        intensity[0, 1, 1] = hazard
        inputs = GPInputs(
            transition_probabilities=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            clock_intensity_per_second=np.zeros(6),
            execution_intensity_per_second={
                "through": intensity,
            },
            calibration_dates=("2026-07-02",),
            calibration_reference_price_hkd=reference_price,
            mean_lifecycle_seconds=dt,
            audit={},
            maker_fill_alpha_hkd_per_lot={"through": alpha},
        )
        config = GPConfig(
            spread_states=2, max_inventory_lots=1,
            time_steps=1, horizon_seconds=dt,
            inventory_penalty_gamma=0.0,
        )
        q_zero = config.max_inventory_lots
        probability = -np.expm1(-hazard * dt)
        fee = reference_price * LOT_SIZE * ALL_IN_FEE_RATE
        # At two ticks, improving by one tick earns zero quoted edge; a fill
        # then requires a market sale at half-spread of two HKD per lot.
        expected = probability * (20.0 - 2.0 - 2.0 * fee)
        neutral = solve_gp_policy(
            inputs, tick, "through", 0, config, market_orders_enabled=False
        )
        self.assertEqual(neutral.bid_action[1, 1, q_zero], ACTION_IMPROVE)
        self.assertEqual(neutral.ask_action[1, 1, q_zero], ACTION_NONE)
        self.assertEqual(neutral.bid_action[1, 0, q_zero], ACTION_NONE)
        self.assertAlmostEqual(neutral.values[1, 1, q_zero], expected)

        drift = 0.005
        directional = solve_gp_drift_policy(
            inputs, tick, "through", 0, config,
            make_gp_drift_regime([drift], np.ones((1, 1))),
            market_orders_enabled=False,
        )
        post_fill_seconds = dt - probability / hazard
        self.assertEqual(
            directional.bid_action[1, 0, 1, q_zero], ACTION_IMPROVE
        )
        self.assertEqual(directional.ask_action[1, 0, 1, q_zero], ACTION_NONE)
        self.assertAlmostEqual(
            directional.values[1, 0, 1, q_zero],
            expected + LOT_SIZE * drift * post_fill_seconds,
        )

    def test_maker_fill_alpha_rejects_bad_shape_and_nonfinite_values(self) -> None:
        config = GPConfig(time_steps=1, horizon_seconds=3.0)
        for bad_alpha in (np.zeros((2, 2, 5)), np.full((2, 2, 6), np.nan)):
            inputs = replace(
                self.inputs(),
                maker_fill_alpha_hkd_per_lot={"through": bad_alpha},
            )
            with self.assertRaisesRegex(ValueError, "maker_fill_alpha"):
                solve_gp_policy(inputs, 0.02, "through", 0, config)
            with self.assertRaisesRegex(ValueError, "maker_fill_alpha"):
                solve_gp_drift_policy(
                    inputs, 0.02, "through", 0, config,
                    make_gp_drift_regime([0.0], np.ones((1, 1))),
                )

    def test_maker_fill_alpha_is_mode_specific_and_can_be_negative(self) -> None:
        intensity = np.zeros((2, 2, 1))
        intensity[0, 0, 0] = 0.2
        positive = np.zeros_like(intensity)
        positive[0, 0, 0] = 20.0
        negative = -positive
        inputs = GPInputs(
            transition_probabilities=np.ones((1, 1)),
            clock_intensity_per_second=np.zeros(6),
            execution_intensity_per_second={
                "through": intensity,
                "touch": intensity,
            },
            calibration_dates=("2026-07-02",),
            calibration_reference_price_hkd=50.0,
            mean_lifecycle_seconds=3.0,
            audit={},
            maker_fill_alpha_hkd_per_lot={
                "through": positive,
                "touch": negative,
            },
        )
        config = GPConfig(
            spread_states=1, max_inventory_lots=1,
            time_steps=1, horizon_seconds=3.0,
            inventory_penalty_gamma=0.0,
        )
        q_zero = config.max_inventory_lots
        through = solve_gp_policy(
            inputs, 0.02, "through", 0, config,
            market_orders_enabled=False,
        )
        touch = solve_gp_policy(
            inputs, 0.02, "touch", 0, config,
            market_orders_enabled=False,
        )
        self.assertEqual(through.bid_action[1, 0, q_zero], ACTION_BEST)
        self.assertEqual(touch.bid_action[1, 0, q_zero], ACTION_NONE)
        self.assertGreater(through.values[1, 0, q_zero], 0.0)
        self.assertEqual(touch.values[1, 0, q_zero], 0.0)

    def test_positive_drift_makes_long_inventory_more_valuable(self) -> None:
        config = GPConfig(
            time_steps=4,
            horizon_seconds=12.0,
            inventory_penalty_gamma=0.01,
        )
        regime = make_gp_drift_regime(
            [-0.02, 0.02], np.eye(2), ["down", "up"]
        )
        policy = solve_gp_drift_policy(
            self.inputs(), 0.02, "through", 0, config, regime,
            market_orders_enabled=False,
        )
        cap = config.max_inventory_lots
        long_one = cap + 1
        self.assertGreater(
            policy.values[-1, 1, 1, long_one],
            policy.values[-1, 0, 1, long_one],
        )

    def test_last_step_flat_inventory_responds_to_post_fill_drift(self) -> None:
        """At q=0, only a fill can create drift exposure in the last step."""
        hazard = 0.2
        dt = 3.0
        tick = 0.02
        reference_price = 50.0
        intensity = np.full((2, 2, 1), hazard)
        inputs = GPInputs(
            transition_probabilities=np.ones((1, 1)),
            clock_intensity_per_second=np.ones(6),
            execution_intensity_per_second={"through": intensity},
            calibration_dates=("2026-07-02",),
            calibration_reference_price_hkd=reference_price,
            mean_lifecycle_seconds=dt,
            audit={},
        )
        config = GPConfig(
            spread_states=1, max_inventory_lots=1,
            time_steps=1, horizon_seconds=dt,
            inventory_penalty_gamma=0.0,
        )
        q_zero = config.max_inventory_lots
        neutral = solve_gp_policy(
            inputs, tick, "through", 0, config, market_orders_enabled=False
        )
        self.assertEqual(neutral.bid_action[1, 0, q_zero], ACTION_NONE)
        self.assertEqual(neutral.ask_action[1, 0, q_zero], ACTION_NONE)

        fill_probability = -np.expm1(-hazard * dt)
        post_fill_seconds = dt - fill_probability / hazard
        fee = reference_price * LOT_SIZE * ALL_IN_FEE_RATE
        net_fill_and_liquidation = -2.0 * fee
        for drift, expected_bid, expected_ask in (
            (0.1, ACTION_BEST, ACTION_NONE),
            (-0.1, ACTION_NONE, ACTION_BEST),
        ):
            with_drift = solve_gp_drift_policy(
                inputs, tick, "through", 0, config,
                make_gp_drift_regime([drift], np.ones((1, 1))),
                market_orders_enabled=False,
            )
            self.assertEqual(with_drift.bid_action[1, 0, 0, q_zero], expected_bid)
            self.assertEqual(with_drift.ask_action[1, 0, 0, q_zero], expected_ask)
            self.assertAlmostEqual(
                with_drift.values[1, 0, 0, q_zero],
                fill_probability * net_fill_and_liquidation
                + LOT_SIZE * abs(drift) * post_fill_seconds,
            )

    def test_fitted_drift_regime_keeps_session_boundaries_separate(self) -> None:
        regime = fit_gp_drift_regime(
            [np.array([-0.02, -0.01]), np.array([0.01, 0.02])],
            state_count=2,
            transition_smoothing=0.0,
        )
        states = assign_gp_drift_states(
            np.array([-0.02, 0.02]), regime
        )
        self.assertEqual(states.tolist(), [0, 1])
        self.assertEqual(regime.transition_probabilities[0, 1], 0.0)
        self.assertEqual(regime.transition_probabilities[1, 0], 0.0)

    def test_drift_transition_rescaling_matches_integer_steps(self) -> None:
        transition = np.asarray([[0.8, 0.2], [0.1, 0.9]])
        regime = make_gp_drift_regime([-0.01, 0.01], transition)
        rescaled = rescale_gp_drift_regime(regime, 1.0, 2.0)
        np.testing.assert_allclose(
            rescaled.transition_probabilities, transition @ transition
        )


if __name__ == "__main__":
    unittest.main()
