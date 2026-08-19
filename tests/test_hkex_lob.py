import unittest

import numpy as np

from evaluate_7709_transfer import (
    forward_returns,
    labels_from_returns,
    valid_target_indices,
)
from prepare_hkex_lob import ASK, BID, OrderBook, required_price, session_id


class OrderBookTests(unittest.TestCase):
    def test_old_and_new_price_encodings_match(self) -> None:
        self.assertEqual(required_price({"Price": "14.530"}), 14_530)
        self.assertEqual(required_price({"Price": "14530"}), 14_530)

    def test_add_modify_delete_and_feature_order(self) -> None:
        book = OrderBook()
        order_id = 1
        for level in range(10):
            self.assertTrue(book.add(order_id, BID, 10_000 - level, 100 + level))
            order_id += 1
            self.assertTrue(book.add(order_id, ASK, 10_010 + level, 200 + level))
            order_id += 1

        snapshot = book.snapshot(10)
        self.assertIsNotNone(snapshot)
        np.testing.assert_allclose(
            snapshot[:4], np.array([0.10010, 0.00200, 0.10000, 0.00100])
        )

        self.assertTrue(book.modify(1, 500))
        self.assertAlmostEqual(float(book.snapshot(10)[3]), 0.005)
        self.assertTrue(book.delete(1))
        self.assertIsNone(book.snapshot(10))

    def test_repairs_only_passive_side_of_crossed_book(self) -> None:
        book = OrderBook()
        self.assertTrue(book.add(1, ASK, 10_000, 100))
        self.assertTrue(book.add(2, ASK, 10_010, 200))
        self.assertTrue(book.add(3, BID, 10_020, 300))
        removed = book.repair_crossed_book(BID)
        self.assertEqual(removed, 2)
        self.assertEqual(list(book.levels[ASK]), [])
        self.assertEqual(list(book.levels[BID]), [10_020])

    def test_sessions(self) -> None:
        self.assertEqual(session_id(20260709092959999), 0)
        self.assertEqual(session_id(20260709093000000), 1)
        self.assertEqual(session_id(20260709120000000), 0)
        self.assertEqual(session_id(20260709130000000), 2)
        self.assertEqual(session_id(20260709160000000), 0)


class LabelTests(unittest.TestCase):
    def test_future_returns_do_not_cross_sessions(self) -> None:
        mids = np.array([100.0, 101.0, 102.0, 200.0, 201.0, 202.0])
        sessions = np.array([1, 1, 1, 2, 2, 2])
        returns = forward_returns(mids, sessions, horizon=1)
        self.assertTrue(np.isnan(returns[2]))
        self.assertTrue(np.isnan(returns[5]))
        self.assertAlmostEqual(returns[0], 0.01)
        self.assertAlmostEqual(returns[3], 0.005)

    def test_labels_and_windows_do_not_cross_sessions(self) -> None:
        returns = np.array([-0.02, 0.0, 0.02, -0.02, 0.0, 0.02, np.nan])
        labels = labels_from_returns(returns, alpha=0.01)
        np.testing.assert_array_equal(labels, [0, 1, 2, 0, 1, 2, -1])
        sessions = np.array([1, 1, 1, 2, 2, 2, 2])
        indices = valid_target_indices(sessions, labels, sequence_length=2, cutoff=0)
        np.testing.assert_array_equal(indices, [1, 2, 4, 5])


if __name__ == "__main__":
    unittest.main()
