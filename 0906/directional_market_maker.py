#!/usr/bin/env python3
"""Market-making execution model using future trades or L1 quote snapshots.

The pricing and modelling components are inherited from the 0824 experiment.
Only the order execution lifecycle is replaced here: an order becomes active
after the configured latency, can fill from either a subsequent trade price or
the opposite L1 best quote, and expires after ``quote_horizon_snapshots``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
LEGACY_DIR = HERE.parent / "0824"
if str(LEGACY_DIR) not in sys.path:
    sys.path.append(str(LEGACY_DIR))

_legacy_spec = importlib.util.spec_from_file_location(
    "_deeplob_0824_directional_market_maker",
    LEGACY_DIR / "directional_market_maker.py",
)
if _legacy_spec is None or _legacy_spec.loader is None:
    raise RuntimeError("cannot load the 0824 market-making components")
_legacy = importlib.util.module_from_spec(_legacy_spec)
sys.modules[_legacy_spec.name] = _legacy
_legacy_spec.loader.exec_module(_legacy)

# Re-export the unchanged model, quote, data-loading, and calibration API so the
# existing walk-forward experiment can use this module as a drop-in replacement.
for _name, _value in vars(_legacy).items():
    if not _name.startswith("__"):
        globals().setdefault(_name, _value)


def book_fill_snapshot_indices(
    book: np.ndarray,
    snapshot_times_ms: np.ndarray,
    quote_index: int,
    expiry_index: int,
    active_after_ms: int,
    bid_quote_hkd: float,
    ask_quote_hkd: float,
    place_bid: bool,
    place_ask: bool,
    fill_mode: str,
) -> dict[str, int | None]:
    """Return the first future L1 snapshot that fills each resting order.

    The inspected window is ``quote_index + 1`` through ``expiry_index``, both
    in the same session.  Snapshots at or before ``active_after_ms`` are ignored
    because the order has not reached the exchange yet.

    ``touch`` regards the opposite best quote reaching our limit as a fill:
    best ask <= our bid, or best bid >= our ask.  ``through`` requires the
    opposite best quote to move one additional tick beyond our limit.
    """

    if fill_mode not in {"touch", "through"}:
        raise ValueError(fill_mode)
    if quote_index < 0 or expiry_index >= len(book) or quote_index >= expiry_index:
        raise ValueError("invalid quote/expiry snapshot range")
    if len(snapshot_times_ms) != len(book):
        raise ValueError("book/timestamp length mismatch")

    future_indices = np.arange(quote_index + 1, expiry_index + 1, dtype=np.int64)
    active = snapshot_times_ms[future_indices] > active_after_ms
    future_indices = future_indices[active]
    if not len(future_indices):
        return {"buy": None, "sell": None}

    one_tick = TICK_HKD if fill_mode == "through" else 0.0
    best_asks = book[future_indices, 0].astype(np.float64, copy=False)
    best_bids = book[future_indices, 2].astype(np.float64, copy=False)
    buy_matches = (
        np.flatnonzero(best_asks <= bid_quote_hkd - one_tick + 1e-9)
        if place_bid
        else np.empty(0, dtype=np.int64)
    )
    sell_matches = (
        np.flatnonzero(best_bids >= ask_quote_hkd + one_tick - 1e-9)
        if place_ask
        else np.empty(0, dtype=np.int64)
    )
    return {
        "buy": int(future_indices[buy_matches[0]]) if len(buy_matches) else None,
        "sell": int(future_indices[sell_matches[0]]) if len(sell_matches) else None,
    }


def simulate_market_maker(
    day: object,
    trades: TradeData | None,
    quote_indices: np.ndarray,
    alpha_ticks: np.ndarray,
    parameters: QuoteParameters | ASParameters | ASDriftParameters,
    fill_mode: str,
    strategy_name: str,
    record_events: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Simulate trade-or-book fills and 20-snapshot order expiry.

    A buy fills when either a future trade price or future best ask reaches the
    bid.  A sell fills when either a future trade price or future best bid
    reaches the ask.  The earliest qualifying evidence wins.
    """

    if fill_mode not in {"touch", "through"}:
        raise ValueError(fill_mode)
    if len(quote_indices) != len(alpha_ticks):
        raise ValueError("quote/signal length mismatch")
    if not len(quote_indices):
        raise ValueError("at least one quote index is required")

    quote_times = compact_time_to_day_ms(day.send_times)
    inventory = 0
    cash = 0.0
    fees = 0.0
    maker_fills = 0
    buy_fills = 0
    sell_fills = 0
    trade_price_fills = 0
    best_ask_fills = 0
    best_bid_fills = 0
    quoted_bid = 0
    quoted_ask = 0
    bid_cancels = 0
    ask_cancels = 0
    processed_quotes = 0
    max_abs_inventory = 0
    maker_turnover = 0.0
    flatten_turnover = 0.0
    equity_path: list[float] = [0.0]
    events: list[dict[str, object]] = []

    for quote_number, (index, alpha_value) in enumerate(zip(quote_indices, alpha_ticks)):
        index = int(index)
        expiry_index = index + parameters.quote_horizon_snapshots
        if expiry_index >= len(day.book) or day.segments[index] != day.segments[expiry_index]:
            continue
        processed_quotes += 1
        best_ask = float(day.book[index, 0])
        best_bid = float(day.book[index, 2])
        if isinstance(parameters, ASDriftParameters):
            bid_quote, ask_quote, center_shift = as_drift_quote_prices(
                best_bid, best_ask, float(alpha_value), inventory, parameters
            )
        elif isinstance(parameters, ASParameters):
            bid_quote, ask_quote, center_shift = classic_as_quote_prices(
                best_bid, best_ask, inventory, parameters
            )
        else:
            bid_quote, ask_quote, center_shift = quote_prices(
                best_bid, best_ask, float(alpha_value), inventory, parameters
            )

        place_bid = inventory < parameters.max_inventory_lots
        place_ask = inventory > -parameters.max_inventory_lots
        quoted_bid += int(place_bid)
        quoted_ask += int(place_ask)
        active_after_ms = int(quote_times[index]) + LATENCY_MS
        hits = book_fill_snapshot_indices(
            day.book,
            quote_times,
            index,
            expiry_index,
            active_after_ms,
            bid_quote,
            ask_quote,
            place_bid,
            place_ask,
            fill_mode,
        )

        expiry_ms = int(quote_times[expiry_index]) + LATENCY_MS
        trade_left = int(
            np.searchsorted(trades.trade_time_ms, active_after_ms, side="right")
        )
        trade_right = int(
            np.searchsorted(trades.trade_time_ms, expiry_ms, side="right")
        )
        trade_prices = trades.price_hkd[trade_left:trade_right]
        one_tick = TICK_HKD if fill_mode == "through" else 0.0
        buy_trade_hits = (
            np.flatnonzero(trade_prices <= bid_quote - one_tick + 1e-9)
            if place_bid
            else np.empty(0, dtype=np.int64)
        )
        sell_trade_hits = (
            np.flatnonzero(trade_prices >= ask_quote + one_tick - 1e-9)
            if place_ask
            else np.empty(0, dtype=np.int64)
        )

        # (evidence time, side, execution price, source, trade index, book index)
        pending: list[tuple[int, str, float, str, int | None, int | None]] = []
        buy_candidates: list[tuple[int, str, int | None, int | None]] = []
        if len(buy_trade_hits):
            trade_index = trade_left + int(buy_trade_hits[0])
            buy_candidates.append(
                (int(trades.trade_time_ms[trade_index]), "trade_price", trade_index, None)
            )
        if hits["buy"] is not None:
            book_index = int(hits["buy"])
            buy_candidates.append(
                (int(quote_times[book_index]), "best_ask", None, book_index)
            )
        if buy_candidates:
            fill_time, source, trade_index, book_index = min(buy_candidates)
            pending.append(
                (fill_time, "buy", bid_quote, source, trade_index, book_index)
            )
        elif place_bid:
            bid_cancels += 1

        sell_candidates: list[tuple[int, str, int | None, int | None]] = []
        if len(sell_trade_hits):
            trade_index = trade_left + int(sell_trade_hits[0])
            sell_candidates.append(
                (int(trades.trade_time_ms[trade_index]), "trade_price", trade_index, None)
            )
        if hits["sell"] is not None:
            book_index = int(hits["sell"])
            sell_candidates.append(
                (int(quote_times[book_index]), "best_bid", None, book_index)
            )
        if sell_candidates:
            fill_time, source, trade_index, book_index = min(sell_candidates)
            pending.append(
                (fill_time, "sell", ask_quote, source, trade_index, book_index)
            )
        elif place_ask:
            ask_cancels += 1
        pending.sort(key=lambda item: (item[0], item[1]))

        for (
            fill_time_ms,
            side,
            execution_price,
            evidence_source,
            evidence_trade_index,
            evidence_book_index,
        ) in pending:
            if side == "buy":
                cash -= execution_price * LOT_SIZE
                inventory += 1
                buy_fills += 1
            else:
                cash += execution_price * LOT_SIZE
                inventory -= 1
                sell_fills += 1
            notional = execution_price * LOT_SIZE
            maker_turnover += notional
            fees += notional * ALL_IN_FEE_RATE
            maker_fills += 1
            if evidence_source == "trade_price":
                trade_price_fills += 1
            elif evidence_source == "best_ask":
                best_ask_fills += 1
            elif evidence_source == "best_bid":
                best_bid_fills += 1
            else:
                raise AssertionError(f"unknown execution evidence: {evidence_source}")
            max_abs_inventory = max(max_abs_inventory, abs(inventory))
            if record_events:
                fill_snapshot_index = int(
                    min(
                        np.searchsorted(quote_times, fill_time_ms, side="left"),
                        len(quote_times) - 1,
                    )
                )
                events.append(
                    {
                        "event_type": "fill",
                        "execution_evidence": evidence_source,
                        "strategy": strategy_name,
                        "fill_mode": fill_mode,
                        "date": day.date,
                        "quote_number": quote_number,
                        "snapshot_index": index,
                        "quote_snapshot_index": index,
                        "fill_snapshot_index": fill_snapshot_index,
                        "expiry_snapshot_index": expiry_index,
                        "signal_time": int(day.send_times[index]),
                        "fill_time_ms": fill_time_ms,
                        "trade_time_ms": (
                            int(trades.trade_time_ms[evidence_trade_index])
                            if evidence_trade_index is not None
                            else fill_time_ms
                        ),
                        "side": side,
                        "price_hkd": execution_price,
                        "evidence_trade_price_hkd": (
                            float(trades.price_hkd[evidence_trade_index])
                            if evidence_trade_index is not None
                            else None
                        ),
                        "evidence_book_snapshot_index": evidence_book_index,
                        "evidence_best_ask_hkd": (
                            float(day.book[evidence_book_index, 0])
                            if evidence_book_index is not None
                            else None
                        ),
                        "evidence_best_bid_hkd": (
                            float(day.book[evidence_book_index, 2])
                            if evidence_book_index is not None
                            else None
                        ),
                        "inventory_after_lots": inventory,
                        "alpha_ticks": float(alpha_value),
                        "center_shift_ticks": center_shift,
                        "fee_hkd": notional * ALL_IN_FEE_RATE,
                    }
                )

        expiry_ask = float(day.book[expiry_index, 0])
        expiry_bid = float(day.book[expiry_index, 2])
        expiry_mid = (expiry_bid + expiry_ask) / 2.0
        equity_path.append(cash + inventory * LOT_SIZE * expiry_mid - fees)

    final_index = int(quote_indices[-1] + parameters.quote_horizon_snapshots)
    final_bid = float(day.book[final_index, 2])
    final_ask = float(day.book[final_index, 0])
    flatten_side = "none"
    flatten_lots = abs(inventory)
    if inventory > 0:
        flatten_side = "sell"
        notional = inventory * LOT_SIZE * final_bid
        flatten_turnover = notional
        cash += notional
        fees += notional * ALL_IN_FEE_RATE
    elif inventory < 0:
        flatten_side = "buy"
        notional = -inventory * LOT_SIZE * final_ask
        flatten_turnover = notional
        cash -= notional
        fees += notional * ALL_IN_FEE_RATE
    inventory = 0
    gross_pnl = cash
    net_pnl = gross_pnl - fees
    equity_path.append(net_pnl)
    equity = np.asarray(equity_path, dtype=np.float64)
    running_max = np.maximum.accumulate(equity)
    max_drawdown_hkd = float(np.max(running_max - equity))
    feeable_turnover = maker_turnover + flatten_turnover
    official_only_fees = feeable_turnover * OFFICIAL_FIXED_FEE_RATE
    all_in_fees = feeable_turnover * ALL_IN_FEE_RATE
    if not np.isclose(fees, all_in_fees, atol=1e-7):
        raise AssertionError("fee accounting mismatch")
    if quoted_bid != buy_fills + bid_cancels:
        raise AssertionError("bid order lifecycle mismatch")
    if quoted_ask != sell_fills + ask_cancels:
        raise AssertionError("ask order lifecycle mismatch")
    best_quote_fills = best_ask_fills + best_bid_fills
    if maker_fills != trade_price_fills + best_quote_fills:
        raise AssertionError("fill evidence accounting mismatch")

    result: dict[str, Any] = {
        "strategy": strategy_name,
        "fill_mode": fill_mode,
        "date": day.date,
        "quotes": processed_quotes,
        "quoted_bid": quoted_bid,
        "quoted_ask": quoted_ask,
        "maker_fills": maker_fills,
        "buy_fills": buy_fills,
        "sell_fills": sell_fills,
        "trade_price_fills": trade_price_fills,
        "best_quote_fills": best_quote_fills,
        "best_ask_fills": best_ask_fills,
        "best_bid_fills": best_bid_fills,
        "trade_price_fill_share": float(trade_price_fills / max(maker_fills, 1)),
        "best_quote_fill_share": float(best_quote_fills / max(maker_fills, 1)),
        "bid_cancels": bid_cancels,
        "ask_cancels": ask_cancels,
        "expired_orders": bid_cancels + ask_cancels,
        "fill_rate_per_side_quote": float(maker_fills / max(quoted_bid + quoted_ask, 1)),
        "max_abs_inventory_lots": max_abs_inventory,
        "gross_pnl_hkd": gross_pnl,
        "official_only_fees_hkd": official_only_fees,
        "fees_hkd": all_in_fees,
        "net_pnl_official_only_hkd": gross_pnl - official_only_fees,
        "net_pnl_hkd": net_pnl,
        "maker_turnover_hkd": maker_turnover,
        "feeable_turnover_hkd": feeable_turnover,
        "gross_bps_of_feeable_turnover": (
            float(gross_pnl / feeable_turnover * 10_000) if feeable_turnover else None
        ),
        "net_bps_of_maker_turnover": (
            float(net_pnl / maker_turnover * 10_000) if maker_turnover else None
        ),
        "break_even_fee_bps_per_execution_side": (
            float(gross_pnl / feeable_turnover * 10_000) if feeable_turnover else None
        ),
        "max_drawdown_hkd": max_drawdown_hkd,
        "end_flatten_side": flatten_side,
        "end_flatten_lots": flatten_lots,
        "parameters": asdict(parameters),
        "execution_model": {
            "evidence": "earliest qualifying future trade price OR opposite L1 best quote",
            "latency_ms": LATENCY_MS,
            "lifetime_snapshots": parameters.quote_horizon_snapshots,
            "expiry_inclusive": True,
            "unfilled_action": "cancel",
            "trade_prints_used": True,
            "buy_condition": "bid quote >= future best ask OR bid quote >= future trade price",
            "sell_condition": "ask quote <= future best bid OR ask quote <= future trade price",
        },
        "execution_cost": {
            "official_fixed_fee_bps_per_side": OFFICIAL_FIXED_FEE_RATE * 10_000,
            "brokerage_assumption_bps_per_side": BROKERAGE_ASSUMPTION_RATE * 10_000,
            "all_in_bps_per_side": ALL_IN_FEE_RATE * 10_000,
        },
    }
    return result, events
