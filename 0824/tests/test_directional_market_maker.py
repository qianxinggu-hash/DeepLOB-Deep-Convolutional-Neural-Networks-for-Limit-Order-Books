import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from directional_market_maker import (  # noqa: E402
    ASParameters,
    QuoteParameters,
    classic_as_quote_prices,
    compact_time_to_day_ms,
    fit_alpha_calibrator,
    quote_prices,
)


class DirectionalMarketMakerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = QuoteParameters(1, 2.0, 1.0, 5)

    def test_compact_timestamp_conversion(self) -> None:
        value = np.array([20260721093010891], dtype=np.int64)
        self.assertEqual(int(compact_time_to_day_ms(value)[0]), 34_210_891)

    def test_positive_alpha_moves_quote_center_up_without_crossing(self) -> None:
        neutral_bid, neutral_ask, _ = quote_prices(99.98, 100.00, 0.0, 0, self.parameters)
        up_bid, up_ask, _ = quote_prices(99.98, 100.00, 1.0, 0, self.parameters)
        self.assertGreaterEqual(up_bid, neutral_bid)
        self.assertGreater(up_ask, neutral_ask)
        self.assertLess(up_bid, up_ask)

    def test_long_inventory_moves_quote_center_down(self) -> None:
        _, _, flat_shift = quote_prices(99.98, 100.00, 0.0, 0, self.parameters)
        bid, ask, long_shift = quote_prices(99.98, 100.00, 0.0, 5, self.parameters)
        self.assertLess(long_shift, flat_shift)
        self.assertLess(bid, ask)

    def test_failed_alpha_calibration_never_inverts_signal(self) -> None:
        score = np.array([-1.0, 0.0, 1.0])
        future = np.array([1.0, 0.0, -1.0])
        calibration = fit_alpha_calibrator(score, future)
        self.assertEqual(calibration.slope_ticks, 0.0)

    def test_classic_as_long_inventory_lowers_reservation_price(self) -> None:
        parameters = ASParameters(0.02, 1.0, 10.0, 5)
        _, _, flat_shift = classic_as_quote_prices(99.98, 100.00, 0, parameters)
        bid, ask, long_shift = classic_as_quote_prices(99.98, 100.00, 3, parameters)
        self.assertLess(long_shift, flat_shift)
        self.assertLess(bid, ask)

    def test_classic_as_quotes_are_passive(self) -> None:
        parameters = ASParameters(0.05, 2.0, 5.0, 5)
        bid, ask, _ = classic_as_quote_prices(99.98, 100.00, -5, parameters)
        self.assertLessEqual(bid, 99.98)
        self.assertGreaterEqual(ask, 100.00)


if __name__ == "__main__":
    unittest.main()
