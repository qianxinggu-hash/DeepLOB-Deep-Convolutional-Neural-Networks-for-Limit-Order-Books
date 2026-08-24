from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lob_reconstruction import AbsoluteLevelBook, StrictMBOBook  # noqa: E402
from run_experiment import eligible_indices, prepare_data, symmetric_returns  # noqa: E402
from run_improved_experiment import causal_features, interval_indices  # noqa: E402


class AbsoluteLevelBookTests(unittest.TestCase):
    def test_absolute_update_overwrites_and_zero_deletes(self) -> None:
        book = AbsoluteLevelBook()
        book.set_level("bid", 1000, 100)
        book.set_level("bid", 1000, 60)
        self.assertEqual(book.bids[1000], 60)
        book.set_level("bid", 1000, 0)
        self.assertNotIn(1000, book.bids)

    def test_crossed_snapshot_is_skipped_without_mutation(self) -> None:
        book = AbsoluteLevelBook()
        for index in range(10):
            book.set_level("bid", 1000 - index, 100)
            book.set_level("ask", 1000 + index, 100)
        state, reason = book.snapshot()
        self.assertIsNone(state)
        self.assertEqual(reason, "crossed_or_locked")
        self.assertEqual(len(book.bids), 10)
        self.assertEqual(len(book.asks), 10)


class StrictMBOBookTests(unittest.TestCase):
    def test_modify_uses_quantity_delta_and_delete_removes_remainder(self) -> None:
        book = StrictMBOBook()
        self.assertTrue(book.add(1, 0, 1000, 100))
        self.assertTrue(book.add(2, 0, 1000, 50))
        self.assertEqual(book.bids[1000], 150)
        self.assertTrue(book.modify(1, 0, 40))
        self.assertEqual(book.bids[1000], 90)
        self.assertTrue(book.delete(2, 0))
        self.assertEqual(book.bids[1000], 40)

    def test_missing_order_taints_state_without_guessing(self) -> None:
        book = StrictMBOBook()
        self.assertFalse(book.delete(999, 0))
        self.assertTrue(book.tainted)
        state, reason = book.snapshot()
        self.assertIsNone(state)
        self.assertEqual(reason, "tainted_mbo_state")

    def test_duplicate_add_does_not_overwrite_original(self) -> None:
        book = StrictMBOBook()
        self.assertTrue(book.add(1, 0, 1000, 100))
        self.assertFalse(book.add(1, 1, 1100, 50))
        self.assertEqual(book.orders[1], (0, 1000, 100))
        self.assertEqual(book.bids[1000], 100)
        self.assertNotIn(1100, book.asks)


class SplitTests(unittest.TestCase):
    def test_train_and_test_windows_are_strictly_disjoint(self) -> None:
        n = 1000
        returns = np.linspace(-0.01, 0.01, n)
        segments = np.ones(n, dtype=np.int16)
        train, test = eligible_indices(returns, segments, 700, 100, 20)
        self.assertLess(int(train[-1]) + 20, 700)
        self.assertGreaterEqual(int(test[0]) - 100 + 1, 700)

    def test_returns_do_not_cross_segments(self) -> None:
        mids = np.r_[np.arange(50.0), np.arange(100.0, 150.0)]
        segments = np.r_[np.ones(50), np.full(50, 2)]
        returns = symmetric_returns(mids, segments, 5)
        self.assertTrue(np.isnan(returns[48]))
        self.assertTrue(np.isnan(returns[51]))

    def test_normalization_uses_train_prefix_only(self) -> None:
        features = np.zeros((500, 40), dtype=np.float32)
        for level in range(10):
            base = level * 4
            features[:, base] = 10.1 + level * 0.01
            features[:, base + 1] = 100 + level
            features[:, base + 2] = 9.9 - level * 0.01
            features[:, base + 3] = 100 + level
        features[350:, 1::4] = 1_000_000
        features[350:, 3::4] = 1_000_000
        segments = np.ones(500, dtype=np.int16)
        prepared = prepare_data(features, segments, 0.7, 20, 5, 1 / 3)
        self.assertLess(float(prepared.mean[1]), np.log1p(200))

    def test_interval_indices_keep_all_context_inside_partition(self) -> None:
        returns = np.linspace(-0.01, 0.01, 1000)
        segments = np.ones(1000, dtype=np.int16)
        indices = interval_indices(returns, segments, 400, 800, 100, 20)
        self.assertGreaterEqual(int(indices[0]) - 99, 400)
        self.assertLess(int(indices[-1]) + 20, 800)

    def test_causal_features_are_finite_and_25_wide(self) -> None:
        rows = 300
        features = np.empty((rows, 40), dtype=np.float32)
        for level in range(10):
            base = level * 4
            trend = np.arange(rows, dtype=np.float32) * 0.001
            features[:, base] = 10.1 + level * 0.01 + trend
            features[:, base + 1] = 100 + level
            features[:, base + 2] = 9.9 - level * 0.01 + trend
            features[:, base + 3] = 120 + level
        output = causal_features(features, np.arange(100, 300))
        self.assertEqual(output.shape, (200, 25))
        self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
