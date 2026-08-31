from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lob_reconstruction import (  # noqa: E402
    AbsoluteLevelBook,
    StrictMBOBook,
    TolerantMBOBook,
    compact_hk_time_to_day_ms,
    reconstruct_mbo_csv,
)
from run_experiment import (  # noqa: E402
    eligible_indices,
    forward_returns,
    prepare_data,
    symmetric_returns,
)
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


class TolerantMBOBookTests(unittest.TestCase):
    def test_missing_modify_delete_are_ignored_but_new_adds_continue(self) -> None:
        book = TolerantMBOBook()
        self.assertFalse(book.modify(999, 0, 100))
        self.assertFalse(book.delete(999, 0))
        self.assertFalse(book.tainted)
        self.assertTrue(book.add(1, 0, 1000, 50))
        self.assertEqual(book.bids[1000], 50)
        self.assertEqual(book.errors["modify_missing_order"], 1)
        self.assertEqual(book.errors["delete_missing_order"], 1)

    def test_time_gap_quarantines_snapshots_and_splits_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "synthetic.csv"
            fieldnames = [
                "SendTime",
                "MsgType",
                "SecurityId",
                "Price",
                "Quantity",
                "Side",
                "OrderID",
                "OrderBookPosition",
                "TradeID",
                "TradeTime",
                "TrdType",
            ]
            rows: list[dict[str, object]] = []
            for level in range(10):
                rows.append(
                    {
                        "SendTime": 20260703093000000,
                        "MsgType": 30,
                        "SecurityId": 7709,
                        "Price": 1000 - level,
                        "Quantity": 100,
                        "Side": 0,
                        "OrderID": 100 + level,
                    }
                )
                rows.append(
                    {
                        "SendTime": 20260703093000000,
                        "MsgType": 30,
                        "SecurityId": 7709,
                        "Price": 1100 + level,
                        "Quantity": 100,
                        "Side": 1,
                        "OrderID": 200 + level,
                    }
                )
            rows.extend(
                [
                    {
                        "SendTime": 20260703093000001,
                        "MsgType": 31,
                        "SecurityId": 7709,
                        "Price": "",
                        "Quantity": 90,
                        "Side": 0,
                        "OrderID": 100,
                    },
                    {
                        "SendTime": 20260703093301001,
                        "MsgType": 31,
                        "SecurityId": 7709,
                        "Price": "",
                        "Quantity": 50,
                        "Side": 0,
                        "OrderID": 999999,
                    },
                    {
                        "SendTime": 20260703093801002,
                        "MsgType": 31,
                        "SecurityId": 7709,
                        "Price": "",
                        "Quantity": 80,
                        "Side": 0,
                        "OrderID": 400,
                    },
                ]
            )
            rebuilt_rows: list[dict[str, object]] = []
            for level in range(10):
                rebuilt_rows.append(
                    {
                        "SendTime": 20260703093301001,
                        "MsgType": 30,
                        "SecurityId": 7709,
                        "Price": 1000 - level,
                        "Quantity": 100,
                        "Side": 0,
                        "OrderID": 400 + level,
                    }
                )
                rebuilt_rows.append(
                    {
                        "SendTime": 20260703093301001,
                        "MsgType": 30,
                        "SecurityId": 7709,
                        "Price": 1100 + level,
                        "Quantity": 100,
                        "Side": 1,
                        "OrderID": 500 + level,
                    }
                )
            rows[22:22] = rebuilt_rows
            recovery_rows: list[dict[str, object]] = []
            start_second = 9 * 3600 + 33 * 60 + 1
            for elapsed in range(20, 301, 20):
                total_second = start_second + elapsed
                hour, remainder = divmod(total_second, 3600)
                minute, second = divmod(remainder, 60)
                recovery_rows.append(
                    {
                        "SendTime": int(
                            f"20260703{hour:02d}{minute:02d}{second:02d}000"
                        ),
                        "MsgType": 31,
                        "SecurityId": 7709,
                        "Price": "",
                        "Quantity": 80 + elapsed % 40,
                        "Side": 0,
                        "OrderID": 400,
                    }
                )
            rows[-1:-1] = recovery_rows
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            result = reconstruct_mbo_csv(
                source,
                snapshot_every=1,
                policy="tolerant",
                gap_threshold_ms=30_000,
                recovery_ms=300_000,
            )

        self.assertEqual(result.metadata["reconstruction_policy"], "tolerant")
        self.assertEqual(len(result.metadata["gap_events"]), 1)
        self.assertEqual(result.metadata["order_errors"]["modify_missing_order"], 1)
        self.assertFalse(result.metadata["final_state_tainted"])
        self.assertEqual(set(result.segments.tolist()), {1, 2})
        self.assertGreaterEqual(
            result.metadata["quality_counts"]["skipped_recovery_changed_groups"],
            1,
        )

    def test_compact_time_to_day_ms(self) -> None:
        self.assertEqual(
            compact_hk_time_to_day_ms(20260703093647406),
            ((9 * 60 + 36) * 60 + 47) * 1000 + 406,
        )


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

    def test_forward_returns_use_current_mid_and_future_only(self) -> None:
        mids = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
        segments = np.ones(len(mids), dtype=np.int16)
        returns = forward_returns(mids, segments, 2)
        self.assertAlmostEqual(returns[0], 103.0 / 100.0 - 1.0)
        self.assertAlmostEqual(returns[3], 109.0 / 106.0 - 1.0)
        self.assertTrue(np.isnan(returns[4]))
        self.assertTrue(np.isnan(returns[5]))

    def test_forward_returns_do_not_cross_segments(self) -> None:
        mids = np.r_[np.arange(10.0, 20.0), np.arange(100.0, 110.0)]
        segments = np.r_[np.ones(10), np.full(10, 2)]
        returns = forward_returns(mids, segments, 3)
        self.assertTrue(np.isnan(returns[7]))
        self.assertTrue(np.isnan(returns[8]))

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
