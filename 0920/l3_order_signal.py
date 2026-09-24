#!/usr/bin/env python3
"""Causal order-identity features aligned to the existing ten-level quote states.

Only dates with no recorded reconstruction gaps or order errors are accepted.
Messages with one SendTime are applied atomically before any feature is read.
The sampled MBO reconstruction is checked against the tracked order book at
every quote used by the drift experiment.
"""

from __future__ import annotations

import csv
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "0824"))
from lob_reconstruction import compact_hk_time_to_day_ms  # noqa: E402


FEATURE_NAMES = (
    "best_order_count_imbalance",
    "best_mean_order_age_log_imbalance",
    "best_fresh_quantity_share_imbalance_5s",
    "best_order_size_concentration_imbalance",
    "best_old_removed_quantity_pressure_5s",
    "best_removed_order_count_pressure_5s",
)
WINDOW_MS = 5_000
FRESH_MS = 5_000
OLD_MS = 10_000
AGE_CAP_MS = 300_000


def _price_mills(value: str) -> int:
    return int(value) if "." not in value else round(float(value) * 1000)


def _best_features(
    orders: dict[int, tuple[int, int, int, int]],
    level_ids: list[dict[int, set[int]]],
    events: deque[tuple[int, int, int, int, int]],
    now_ms: int,
    bid_price: int,
    ask_price: int,
) -> np.ndarray:
    side_stats = []
    for side, price in ((0, bid_price), (1, ask_price)):
        ids = level_ids[side].get(price, ())
        if not ids:
            raise ValueError(f"empty tracked best level at {now_ms}: {side=} {price=}")
        quantities = [orders[order_id][2] for order_id in ids]
        ages = [min(max(now_ms - orders[order_id][3], 0), AGE_CAP_MS) for order_id in ids]
        total = sum(quantities)
        count = len(quantities)
        mean_age_s = sum(q * a for q, a in zip(quantities, ages)) / total / 1000
        fresh_share = sum(q for q, a in zip(quantities, ages) if a <= FRESH_MS) / total
        concentration = sum(q * q for q in quantities) / (total * total)
        side_stats.append((count, total, mean_age_s, fresh_share, concentration))
    (bid_n, bid_q, bid_age, bid_fresh, bid_hhi), (ask_n, ask_q, ask_age, ask_fresh, ask_hhi) = side_stats
    old_removed = [0, 0]
    removed_count = [0, 0]
    for event_ms, side, price, quantity, age_ms in events:
        if price != (bid_price if side == 0 else ask_price):
            continue
        removed_count[side] += 1
        if age_ms >= OLD_MS:
            old_removed[side] += quantity
    return np.asarray(
        [
            (bid_n - ask_n) / (bid_n + ask_n),
            np.log1p(bid_age) - np.log1p(ask_age),
            bid_fresh - ask_fresh,
            ask_hhi - bid_hhi,
            np.clip(old_removed[1] / ask_q - old_removed[0] / bid_q, -5, 5),
            np.clip(removed_count[1] / ask_n - removed_count[0] / bid_n, -5, 5),
        ],
        dtype=np.float32,
    )


def extract_l3_features(
    raw_csv: Path,
    quote_times: np.ndarray,
    quote_book: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Stream raw MBO once and return six order-identity features per quote.

    The input book has the standard 40-column alternating ask/bid L2 layout.
    Every quote's best prices and quantities must match this independent replay.
    """
    if quote_book.shape != (len(quote_times), 40):
        raise ValueError("expected one 40-column book row per quote")
    if len(quote_times) and np.any(np.diff(quote_times) <= 0):
        raise ValueError("quote times must be strictly increasing")
    out = np.empty((len(quote_times), len(FEATURE_NAMES)), dtype=np.float32)
    orders: dict[int, tuple[int, int, int, int]] = {}
    level_ids: list[dict[int, set[int]]] = [{}, {}]
    level_qty: list[dict[int, int]] = [{}, {}]
    cancels: deque[tuple[int, int, int, int, int]] = deque()
    quote_at = 0
    group_time: int | None = None
    group_ms = 0
    diagnostics = {"rows": 0, "adds": 0, "modifies": 0, "deletes": 0, "quotes_checked": 0}

    def flush() -> None:
        nonlocal quote_at
        if group_time is None:
            return
        while cancels and cancels[0][0] < group_ms - WINDOW_MS:
            cancels.popleft()
        while quote_at < len(quote_times) and int(quote_times[quote_at]) == group_time:
            row = quote_book[quote_at]
            ask_price = round(float(row[0]) * 1000)
            bid_price = round(float(row[2]) * 1000)
            if (
                (max(level_qty[0]) if level_qty[0] else None) != bid_price
                or (min(level_qty[1]) if level_qty[1] else None) != ask_price
                or level_qty[0].get(bid_price) != round(float(row[3]))
                or level_qty[1].get(ask_price) != round(float(row[1]))
            ):
                raise ValueError(f"MBO/L2 best-book mismatch at {group_time} quote {quote_at}")
            out[quote_at] = _best_features(
                orders, level_ids, cancels, group_ms, bid_price, ask_price
            )
            diagnostics["quotes_checked"] += 1
            quote_at += 1

    with raw_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"SendTime", "MsgType", "SecurityId", "Price", "Quantity", "Side", "OrderID"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{raw_csv}: missing columns {sorted(missing)}")
        for row in reader:
            if int(row["SecurityId"]) != 7709:
                continue
            diagnostics["rows"] += 1
            send_time = int(row["SendTime"])
            if group_time is None:
                group_time = send_time
                group_ms = compact_hk_time_to_day_ms(send_time)
            elif send_time != group_time:
                if send_time < group_time:
                    raise ValueError(f"out-of-order SendTime at {send_time}")
                flush()
                if quote_at == len(quote_times):
                    break
                if int(quote_times[quote_at]) < send_time:
                    raise ValueError(f"quote time {quote_times[quote_at]} absent from MBO")
                group_time = send_time
                group_ms = compact_hk_time_to_day_ms(send_time)
            kind = int(row["MsgType"])
            if kind == 50:
                continue  # trade prints do not mutate the order book
            if kind not in (30, 31, 32):
                continue
            order_id = int(row["OrderID"])
            side = int(row["Side"])
            if kind == 30:
                if order_id in orders:
                    raise ValueError(f"duplicate OrderID {order_id} at {send_time}")
                price = _price_mills(row["Price"])
                quantity = int(row["Quantity"])
                if quantity <= 0 or side not in (0, 1):
                    raise ValueError(f"invalid add {order_id} at {send_time}")
                orders[order_id] = (side, price, quantity, group_ms)
                level_ids[side].setdefault(price, set()).add(order_id)
                level_qty[side][price] = level_qty[side].get(price, 0) + quantity
                diagnostics["adds"] += 1
            else:
                if order_id not in orders:
                    raise ValueError(f"unknown OrderID {order_id} at {send_time}")
                old_side, price, old_qty, birth_ms = orders[order_id]
                if old_side != side:
                    raise ValueError(f"OrderID side changed at {send_time}")
                if kind == 31:
                    quantity = int(row["Quantity"])
                    if quantity <= 0:
                        raise ValueError(f"invalid modify {order_id} at {send_time}")
                    orders[order_id] = (side, price, quantity, birth_ms)
                    level_qty[side][price] += quantity - old_qty
                    diagnostics["modifies"] += 1
                else:
                    cancels.append((group_ms, side, price, old_qty, group_ms - birth_ms))
                    del orders[order_id]
                    level_ids[side][price].remove(order_id)
                    if not level_ids[side][price]:
                        del level_ids[side][price]
                    remaining = level_qty[side][price] - old_qty
                    if remaining:
                        level_qty[side][price] = remaining
                    else:
                        del level_qty[side][price]
                    diagnostics["deletes"] += 1
        flush()
    if quote_at != len(quote_times):
        raise ValueError(f"matched {quote_at}/{len(quote_times)} quote times")
    if not np.isfinite(out).all():
        raise ValueError("non-finite L3 features")
    return out, diagnostics
