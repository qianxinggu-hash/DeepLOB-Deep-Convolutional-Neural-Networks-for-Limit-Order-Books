import unittest

import numpy as np

from train_7709_deeplob import DayData, eligible_targets, equation4_returns


class Equation4Tests(unittest.TestCase):
    def make_day(self, mids: list[float], sessions: list[int]) -> DayData:
        features = np.zeros((len(mids), 40), dtype=np.float32)
        features[:, 0] = np.asarray(mids) + 0.5
        features[:, 2] = np.asarray(mids) - 0.5
        return DayData(
            date="2026-01-01",
            path=None,
            features=features,
            sessions=np.asarray(sessions, dtype=np.uint8),
            metadata={},
        )

    def test_equation4_uses_symmetric_k_means(self) -> None:
        day = self.make_day([10, 20, 30, 40, 50], [1, 1, 1, 1, 1])
        returns = equation4_returns(day, k=2)
        self.assertTrue(np.isnan(returns[0]))
        self.assertAlmostEqual(returns[1], 35 / 15 - 1)
        self.assertAlmostEqual(returns[2], 45 / 25 - 1)
        self.assertTrue(np.isnan(returns[3]))

    def test_targets_and_returns_do_not_cross_sessions(self) -> None:
        day = self.make_day(
            [10, 11, 12, 13, 100, 101, 102, 103],
            [1, 1, 1, 1, 2, 2, 2, 2],
        )
        returns = equation4_returns(day, k=1)
        targets = eligible_targets(day, returns, sequence_length=2, stride=1)
        np.testing.assert_array_equal(targets, [1, 2, 5, 6])
        self.assertTrue(np.isnan(returns[3]))
        self.assertTrue(np.isnan(returns[7]))


if __name__ == "__main__":
    unittest.main()
