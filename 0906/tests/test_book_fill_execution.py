import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from directional_market_maker import (  # noqa: E402
    QuoteParameters,
    TradeData,
    book_fill_snapshot_indices,
    compact_time_to_day_ms,
    simulate_market_maker,
)


class BookFillExecutionTests(unittest.TestCase):
    @staticmethod
    def trades(times: list[int], prices: list[float]) -> TradeData:
        return TradeData(
            trade_time_ms=np.asarray(times, dtype=np.int64),
            send_time_ms=np.asarray(times, dtype=np.int64),
            price_hkd=np.asarray(prices, dtype=np.float64),
            quantity=np.ones(len(times), dtype=np.int64),
            metadata={},
        )

    def test_current_snapshot_never_fills_the_order(self) -> None:
        book = np.array(
            [
                [99.98, 1, 99.96, 1],
                [100.00, 1, 99.98, 1],
            ],
            dtype=np.float64,
        )
        hits = book_fill_snapshot_indices(
            book,
            np.array([0, 10]),
            quote_index=0,
            expiry_index=1,
            active_after_ms=-1,
            bid_quote_hkd=99.98,
            ask_quote_hkd=100.00,
            place_bid=True,
            place_ask=False,
            fill_mode="touch",
        )
        self.assertIsNone(hits["buy"])

    def test_touch_uses_a_later_opposite_best_quote(self) -> None:
        book = np.array(
            [
                [100.00, 1, 99.98, 1],
                [100.00, 1, 99.98, 1],
                [99.98, 1, 99.96, 1],
                [100.00, 1, 99.98, 1],
            ],
            dtype=np.float64,
        )
        hits = book_fill_snapshot_indices(
            book,
            np.array([0, 10, 20, 30]),
            quote_index=0,
            expiry_index=3,
            active_after_ms=5,
            bid_quote_hkd=99.98,
            ask_quote_hkd=100.00,
            place_bid=True,
            place_ask=True,
            fill_mode="touch",
        )
        self.assertEqual(hits, {"buy": 2, "sell": None})

    def test_through_requires_one_more_tick(self) -> None:
        book = np.array(
            [
                [100.00, 1, 99.98, 1],
                [99.98, 1, 99.96, 1],
                [99.96, 1, 99.94, 1],
            ],
            dtype=np.float64,
        )
        hits = book_fill_snapshot_indices(
            book,
            np.array([0, 10, 20]),
            0,
            2,
            5,
            99.98,
            100.00,
            True,
            False,
            "through",
        )
        self.assertEqual(hits["buy"], 2)

    def test_latency_filters_snapshots_before_order_is_active(self) -> None:
        book = np.array(
            [
                [100.00, 1, 99.98, 1],
                [99.98, 1, 99.96, 1],
                [100.00, 1, 99.98, 1],
            ],
            dtype=np.float64,
        )
        hits = book_fill_snapshot_indices(
            book,
            np.array([0, 3, 10]),
            0,
            2,
            5,
            99.98,
            100.00,
            True,
            False,
            "touch",
        )
        self.assertIsNone(hits["buy"])

    def test_expiry_snapshot_is_included_and_later_snapshot_is_not(self) -> None:
        book = np.tile(
            np.array([[100.00, 1, 99.98, 1]], dtype=np.float64), (22, 1)
        )
        book[20, 0] = 99.98
        book[20, 2] = 99.96
        hits = book_fill_snapshot_indices(
            book,
            np.arange(22) * 10,
            0,
            20,
            5,
            99.98,
            100.00,
            True,
            False,
            "touch",
        )
        self.assertEqual(hits["buy"], 20)
        book[20] = [100.00, 1, 99.98, 1]
        book[21] = [99.98, 1, 99.96, 1]
        hits = book_fill_snapshot_indices(
            book,
            np.arange(22) * 10,
            0,
            20,
            5,
            99.98,
            100.00,
            True,
            False,
            "touch",
        )
        self.assertIsNone(hits["buy"])

    def test_best_ask_alone_can_fill_buy_order(self) -> None:
        book = np.array(
            [
                [100.00, 1, 99.98, 1],
                [100.00, 1, 99.98, 1],
                [99.98, 1, 99.96, 1],
                [100.00, 1, 99.98, 1],
            ],
            dtype=np.float64,
        )
        send_times = np.array(
            [
                20260702093000000,
                20260702093000010,
                20260702093000020,
                20260702093000030,
            ],
            dtype=np.int64,
        )
        day = SimpleNamespace(
            date="2026-07-02",
            book=book,
            send_times=send_times,
            segments=np.zeros(4, dtype=np.int16),
        )
        result, events = simulate_market_maker(
            day,
            trades=self.trades([], []),
            quote_indices=np.array([0]),
            alpha_ticks=np.array([0.0]),
            parameters=QuoteParameters(0, 0.0, 0.0, 5, 3),
            fill_mode="touch",
            strategy_name="test",
            record_events=True,
        )
        self.assertEqual(result["buy_fills"], 1)
        self.assertEqual(result["bid_cancels"], 0)
        self.assertEqual(result["sell_fills"], 0)
        self.assertEqual(result["ask_cancels"], 1)
        self.assertEqual(result["quoted_bid"], 1)
        self.assertEqual(result["quoted_ask"], 1)
        self.assertEqual(result["best_quote_fills"], 1)
        self.assertEqual(result["best_ask_fills"], 1)
        self.assertEqual(result["best_bid_fills"], 0)
        self.assertEqual(result["trade_price_fills"], 0)
        self.assertEqual(result["best_quote_fill_share"], 1.0)
        self.assertEqual(result["trade_price_fill_share"], 0.0)
        self.assertEqual(events[0]["fill_snapshot_index"], 2)
        self.assertGreater(
            events[0]["fill_time_ms"],
            int(compact_time_to_day_ms(send_times)[0]) + 5,
        )
        self.assertEqual(events[0]["execution_evidence"], "best_ask")

    def test_trade_price_alone_can_fill_buy_order(self) -> None:
        book = np.tile(
            np.array([[100.00, 1, 99.98, 1]], dtype=np.float64), (4, 1)
        )
        send_times = np.array(
            [
                20260702093000000,
                20260702093000010,
                20260702093000020,
                20260702093000030,
            ],
            dtype=np.int64,
        )
        snapshot_ms = compact_time_to_day_ms(send_times)
        day = SimpleNamespace(
            date="2026-07-02",
            book=book,
            send_times=send_times,
            segments=np.zeros(4, dtype=np.int16),
        )
        result, events = simulate_market_maker(
            day,
            trades=self.trades([int(snapshot_ms[1])], [99.98]),
            quote_indices=np.array([0]),
            alpha_ticks=np.array([0.0]),
            parameters=QuoteParameters(0, 0.0, 0.0, 5, 3),
            fill_mode="touch",
            strategy_name="test",
            record_events=True,
        )
        self.assertEqual(result["buy_fills"], 1)
        self.assertEqual(result["bid_cancels"], 0)
        self.assertEqual(result["trade_price_fills"], 1)
        self.assertEqual(result["best_quote_fills"], 0)
        self.assertEqual(result["trade_price_fill_share"], 1.0)
        self.assertEqual(result["best_quote_fill_share"], 0.0)
        self.assertEqual(events[0]["execution_evidence"], "trade_price")
        self.assertEqual(events[0]["evidence_trade_price_hkd"], 99.98)

    def test_best_bid_alone_can_fill_sell_order(self) -> None:
        book = np.array(
            [
                [100.00, 1, 99.98, 1],
                [100.00, 1, 99.98, 1],
                [100.02, 1, 100.00, 1],
                [100.00, 1, 99.98, 1],
            ],
            dtype=np.float64,
        )
        send_times = np.array(
            [
                20260702093000000,
                20260702093000010,
                20260702093000020,
                20260702093000030,
            ],
            dtype=np.int64,
        )
        day = SimpleNamespace(
            date="2026-07-02",
            book=book,
            send_times=send_times,
            segments=np.zeros(4, dtype=np.int16),
        )
        result, events = simulate_market_maker(
            day,
            trades=self.trades([], []),
            quote_indices=np.array([0]),
            alpha_ticks=np.array([0.0]),
            parameters=QuoteParameters(0, 0.0, 0.0, 5, 3),
            fill_mode="touch",
            strategy_name="test",
            record_events=True,
        )
        self.assertEqual(result["sell_fills"], 1)
        sell_event = next(event for event in events if event["side"] == "sell")
        self.assertEqual(sell_event["execution_evidence"], "best_bid")
        self.assertEqual(result["best_bid_fills"], 1)
        self.assertEqual(result["best_quote_fills"], 1)

    def test_trade_price_alone_can_fill_sell_order(self) -> None:
        book = np.tile(
            np.array([[100.00, 1, 99.98, 1]], dtype=np.float64), (4, 1)
        )
        send_times = np.array(
            [
                20260702093000000,
                20260702093000010,
                20260702093000020,
                20260702093000030,
            ],
            dtype=np.int64,
        )
        snapshot_ms = compact_time_to_day_ms(send_times)
        day = SimpleNamespace(
            date="2026-07-02",
            book=book,
            send_times=send_times,
            segments=np.zeros(4, dtype=np.int16),
        )
        result, events = simulate_market_maker(
            day,
            trades=self.trades([int(snapshot_ms[1])], [100.00]),
            quote_indices=np.array([0]),
            alpha_ticks=np.array([0.0]),
            parameters=QuoteParameters(0, 0.0, 0.0, 5, 3),
            fill_mode="touch",
            strategy_name="test",
            record_events=True,
        )
        self.assertEqual(result["sell_fills"], 1)
        sell_event = next(event for event in events if event["side"] == "sell")
        self.assertEqual(sell_event["execution_evidence"], "trade_price")
        self.assertEqual(result["trade_price_fills"], 1)
        self.assertEqual(sell_event["evidence_trade_price_hkd"], 100.00)

    def test_order_cancels_when_neither_trade_nor_book_condition_occurs(self) -> None:
        book = np.tile(
            np.array([[100.00, 1, 99.98, 1]], dtype=np.float64), (4, 1)
        )
        send_times = np.array(
            [
                20260702093000000,
                20260702093000010,
                20260702093000020,
                20260702093000030,
            ],
            dtype=np.int64,
        )
        day = SimpleNamespace(
            date="2026-07-02",
            book=book,
            send_times=send_times,
            segments=np.zeros(4, dtype=np.int16),
        )
        result, events = simulate_market_maker(
            day,
            trades=self.trades([], []),
            quote_indices=np.array([0]),
            alpha_ticks=np.array([0.0]),
            parameters=QuoteParameters(0, 0.0, 0.0, 5, 3),
            fill_mode="touch",
            strategy_name="test",
            record_events=True,
        )
        self.assertEqual(result["maker_fills"], 0)
        self.assertEqual(result["bid_cancels"], 1)
        self.assertEqual(result["ask_cancels"], 1)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
