"""Gap flattening must split only at recorded feed gaps, not lunch."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_new_hybrid_month import simulate_as  # noqa: E402
from directional_market_maker import (  # noqa: E402
    ASDriftParameters, QuoteParameters, TradeData, as_drift_quote_prices,
    simulate_market_maker,
)


class NewHybridGapTests(unittest.TestCase):
    def test_as_quote_uses_day_tick_grid(self) -> None:
        params = ASDriftParameters(0.02, 1.0, 12.0, 1.0, 5)
        bid, ask, _ = as_drift_quote_prices(
            108.05, 108.10, 0.5, 0, params, tick_hkd=0.05
        )
        self.assertAlmostEqual(bid / 0.05, round(bid / 0.05))
        self.assertAlmostEqual(ask / 0.05, round(ask / 0.05))
        self.assertLess(bid, ask)

    def test_through_fill_uses_day_tick_size(self) -> None:
        book = np.zeros((21, 40), dtype=np.float32)
        book[:, :4] = (100.05, 100.0, 100.0, 100.0)
        day = SimpleNamespace(
            date="2026-07-03", book=book,
            send_times=np.asarray([20260703093000000 + i * 1000 for i in range(21)]),
            segments=np.ones(21, dtype=np.int16),
        )
        trades = TradeData(
            trade_time_ms=np.asarray((34_201_000,)),
            send_time_ms=np.asarray((34_201_000,)),
            price_hkd=np.asarray((99.98,)), quantity=np.asarray((100,)),
            metadata={},
        )
        params = QuoteParameters(0, 0.0, 0.0, 1)
        result, _ = simulate_market_maker(
            day, trades, np.asarray((0,)), np.asarray((0.0,)),
            params, "through", "test", tick_hkd=0.05,
        )
        self.assertEqual(result["maker_fills"], 0)

    def test_forced_flat_splits_at_gap_but_not_lunch(self) -> None:
        day = SimpleNamespace(
            segments=np.asarray((1, 1, 2, 2, 3, 3)),
            metadata={"gap_events": [{"new_segment": 3}]},
        )
        indices = np.arange(6, dtype=np.int64)
        forecast = np.arange(6, dtype=np.float64)
        calls = []

        def fake_simulator(day_arg, trades, selected, alpha, parameters,
                           mode, name, tick_hkd):
            calls.append((selected.tolist(), alpha.tolist()))
            count = len(selected)
            return ({"gross_pnl_hkd": float(count), "fees_hkd": 0.0,
                     "net_pnl_hkd": float(count), "maker_fills": count}, [])

        with patch("run_new_hybrid_month.simulate_market_maker", fake_simulator), \
                patch("run_new_hybrid_month.gp.infer_tick_hkd", return_value=0.05):
            result = simulate_as(day, None, indices, forecast, None,
                                 "through", "test", True)
        self.assertEqual(calls, [([0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0]),
                                 ([4, 5], [4.0, 5.0])])
        self.assertEqual(result["gap_flatten_count"], 1)
        self.assertEqual(result["net_pnl_hkd"], 6.0)


if __name__ == "__main__":
    unittest.main()
