#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "sortedcontainers>=2.4",
# ]
# ///
"""Reconstruct DeepLOB snapshots from HKEX order-level CSV messages.

The supported feed uses message type 30 for a new order, 31 for a quantity
change, 32 for an order deletion, and 50 for a trade print.  Trade prints are
not applied to the book because the corresponding order change/delete message
updates the book separately.

Snapshots use the same first-40-feature ordering as the FI-2010 files:

    ask_price_1, ask_size_1, bid_price_1, bid_size_1, ..., level 10

Prices and quantities are divided by 100,000 to match the decimal-precision
normalised FI-2010 checkpoint included in this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from itertools import islice
from pathlib import Path

import numpy as np
from sortedcontainers import SortedDict


ADD = 30
MODIFY = 31
DELETE = 32
TRADE = 50
BID = 0
ASK = 1
FEATURE_SCALE = 100_000.0
RECONSTRUCTION_VERSION = 3
EXPECTED_COLUMNS = (
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
    "PriceLevel",
    "UpdateAction",
    "NumOrders",
)
REQUIRED_BOOK_COLUMNS = {
    "SendTime",
    "MsgType",
    "Price",
    "Quantity",
    "Side",
    "OrderID",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct 10-level DeepLOB snapshots from HKEX events"
    )
    parser.add_argument("csv", type=Path, help="HKEX order-event CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/hk07709_2026-07-09_lob.npz"),
    )
    parser.add_argument("--security-id", type=int, default=7709)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=10,
        help="emit one snapshot after this many accepted book updates",
    )
    parser.add_argument("--levels", type=int, default=10)
    return parser.parse_args()


def session_id(send_time: int) -> int:
    """Return 1/2 for HK continuous sessions and 0 otherwise."""
    time_of_day = send_time % 1_000_000_000
    if 93_000_000 <= time_of_day < 120_000_000:
        return 1
    if 130_000_000 <= time_of_day < 160_000_000:
        return 2
    return 0


class OrderBook:
    """Order-level book with aggregated, sorted price levels."""

    def __init__(self) -> None:
        self.orders: dict[int, tuple[int, int, int]] = {}
        self.levels = {BID: SortedDict(), ASK: SortedDict()}
        self.errors: Counter[str] = Counter()

    def _adjust_level(self, side: int, price: int, delta: int) -> None:
        levels = self.levels[side]
        quantity = levels.get(price, 0) + delta
        if quantity < 0:
            self.errors["negative_level_quantity"] += 1
            quantity = 0
        if quantity:
            levels[price] = quantity
        elif price in levels:
            del levels[price]

    def add(self, order_id: int, side: int, price: int, quantity: int) -> bool:
        if order_id in self.orders:
            self.errors["duplicate_add"] += 1
            old_side, old_price, old_quantity = self.orders[order_id]
            self._adjust_level(old_side, old_price, -old_quantity)
        if side not in (BID, ASK) or price <= 0 or quantity <= 0:
            self.errors["invalid_add"] += 1
            return False
        self.orders[order_id] = (side, price, quantity)
        self._adjust_level(side, price, quantity)
        return True

    def modify(self, order_id: int, quantity: int) -> bool:
        old = self.orders.get(order_id)
        if old is None:
            self.errors["modify_missing_order"] += 1
            return False
        if quantity <= 0:
            self.errors["invalid_modify_quantity"] += 1
            return False
        side, price, old_quantity = old
        self.orders[order_id] = (side, price, quantity)
        self._adjust_level(side, price, quantity - old_quantity)
        return True

    def delete(self, order_id: int) -> bool:
        old = self.orders.pop(order_id, None)
        if old is None:
            self.errors["delete_missing_order"] += 1
            return False
        side, price, quantity = old
        self._adjust_level(side, price, -quantity)
        return True

    def repair_crossed_book(self, aggressor_side: int | None) -> int:
        """Remove stale passive orders when a feed gap leaves a crossed book.

        An order-level export can occasionally omit the final delete for a
        resting order.  If a later update on the opposite side trades through
        that price, the stale order otherwise poisons every later snapshot.
        The most recently updated side is treated as the aggressor and only
        crossed price levels on the passive side are removed.
        """
        if aggressor_side not in (BID, ASK):
            return 0
        if not self.levels[BID] or not self.levels[ASK]:
            return 0
        best_bid = self.levels[BID].peekitem(-1)[0]
        best_ask = self.levels[ASK].peekitem(0)[0]
        if best_bid < best_ask:
            return 0

        passive_side = ASK if aggressor_side == BID else BID
        if passive_side == ASK:
            crossed_prices = {
                price for price in self.levels[ASK].keys() if price <= best_bid
            }
        else:
            crossed_prices = {
                price for price in self.levels[BID].keys() if price >= best_ask
            }
        stale_order_ids = [
            order_id
            for order_id, (side, price, _) in self.orders.items()
            if side == passive_side and price in crossed_prices
        ]
        for order_id in stale_order_ids:
            self.delete(order_id)
        if stale_order_ids:
            self.errors["crossed_book_repair_events"] += 1
            self.errors["crossed_book_repaired_orders"] += len(stale_order_ids)
            self.errors["crossed_book_repaired_levels"] += len(crossed_prices)
        return len(stale_order_ids)

    def snapshot(self, levels: int) -> np.ndarray | None:
        if len(self.levels[BID]) < levels or len(self.levels[ASK]) < levels:
            return None
        bids = list(islice(reversed(self.levels[BID].items()), levels))
        asks = list(islice(self.levels[ASK].items(), levels))
        if bids[0][0] >= asks[0][0]:
            self.errors["crossed_or_locked_snapshot"] += 1
            return None

        features = np.empty(levels * 4, dtype=np.float32)
        for level, ((ask_price, ask_size), (bid_price, bid_size)) in enumerate(
            zip(asks, bids)
        ):
            start = level * 4
            features[start : start + 4] = (
                ask_price / FEATURE_SCALE,
                ask_size / FEATURE_SCALE,
                bid_price / FEATURE_SCALE,
                bid_size / FEATURE_SCALE,
            )
        return features


class SnapshotBuffer:
    def __init__(self, feature_count: int, initial_capacity: int = 600_000) -> None:
        self.features = np.empty((initial_capacity, feature_count), dtype=np.float32)
        self.send_times = np.empty(initial_capacity, dtype=np.int64)
        self.sessions = np.empty(initial_capacity, dtype=np.uint8)
        self.size = 0

    def append(self, features: np.ndarray, send_time: int, session: int) -> None:
        if self.size == len(self.features):
            capacity = max(1, len(self.features) * 2)
            self.features = np.resize(self.features, (capacity, self.features.shape[1]))
            self.send_times = np.resize(self.send_times, capacity)
            self.sessions = np.resize(self.sessions, capacity)
        self.features[self.size] = features
        self.send_times[self.size] = send_time
        self.sessions[self.size] = session
        self.size += 1

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.features[: self.size].copy(),
            self.send_times[: self.size].copy(),
            self.sessions[: self.size].copy(),
        )


def required_int(row: dict[str, str], column: str) -> int:
    value = row[column]
    if not value:
        raise ValueError(f"Missing required {column} value")
    return int(value)


def required_price(row: dict[str, str], column: str = "Price") -> int:
    """Return the exchange fixed-point price used by the newer CSV schema.

    Older HKEX exports encode prices as decimal currency values (for example,
    ``14.530``), while newer exports encode the same value as ``14530``.
    """
    value = row[column]
    if not value:
        raise ValueError(f"Missing required {column} value")
    if "." not in value:
        return int(value)
    try:
        fixed = Decimal(value) * 1_000
    except InvalidOperation as exc:
        raise ValueError(f"Invalid {column} value: {value!r}") from exc
    if fixed != fixed.to_integral_value():
        raise ValueError(f"{column} has more than three decimal places: {value!r}")
    return int(fixed)


def reconstruct(
    csv_path: Path,
    security_id: int,
    snapshot_every: int,
    levels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    book = OrderBook()
    buffer = SnapshotBuffer(levels * 4)
    message_counts: Counter[int] = Counter()
    accepted_updates = Counter()
    skipped_security = 0
    invalid_rows = 0
    previous_send_time = -1
    out_of_order = 0
    update_counter = {1: 0, 2: 0}
    book_update_groups = Counter()
    group_send_time: int | None = None
    group_changed = False
    last_changed_side: int | None = None

    def flush_group() -> None:
        nonlocal group_changed, last_changed_side
        if group_send_time is None or not group_changed:
            return
        current_session = session_id(group_send_time)
        if not current_session:
            group_changed = False
            last_changed_side = None
            return
        book_update_groups[current_session] += 1
        update_counter[current_session] += 1
        if update_counter[current_session] % snapshot_every == 0:
            book.repair_crossed_book(last_changed_side)
            features = book.snapshot(levels)
            if features is not None:
                buffer.append(features, group_send_time, current_session)
        group_changed = False
        last_changed_side = None

    security_column = "SecurityId"
    csv_columns: tuple[str, ...] = ()
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        csv_columns = tuple(reader.fieldnames or ())
        if "SecurityId" in csv_columns:
            security_column = "SecurityId"
        elif "SecurityCode" in csv_columns:
            security_column = "SecurityCode"
        else:
            raise ValueError(
                "Unexpected CSV columns: neither SecurityId nor SecurityCode is present"
            )
        missing = REQUIRED_BOOK_COLUMNS - set(csv_columns)
        if missing:
            raise ValueError(
                f"Unexpected CSV columns. Missing {sorted(missing)}, found {csv_columns}"
            )

        for row in reader:
            try:
                if required_int(row, security_column) != security_id:
                    skipped_security += 1
                    continue
                send_time = required_int(row, "SendTime")
                message_type = required_int(row, "MsgType")
                message_counts[message_type] += 1
                if send_time < previous_send_time:
                    out_of_order += 1
                previous_send_time = send_time
                if group_send_time is None:
                    group_send_time = send_time
                elif send_time != group_send_time:
                    flush_group()
                    group_send_time = send_time

                changed = False
                changed_side: int | None = None
                if message_type == ADD:
                    changed_side = required_int(row, "Side")
                    changed = book.add(
                        required_int(row, "OrderID"),
                        changed_side,
                        required_price(row),
                        required_int(row, "Quantity"),
                    )
                elif message_type == MODIFY:
                    existing = book.orders.get(required_int(row, "OrderID"))
                    changed_side = existing[0] if existing is not None else None
                    changed = book.modify(
                        required_int(row, "OrderID"),
                        required_int(row, "Quantity"),
                    )
                elif message_type == DELETE:
                    existing = book.orders.get(required_int(row, "OrderID"))
                    changed_side = existing[0] if existing is not None else None
                    changed = book.delete(required_int(row, "OrderID"))
                elif message_type != TRADE:
                    book.errors["unsupported_message_type"] += 1

                if not changed:
                    continue
                accepted_updates[message_type] += 1
                group_changed = True
                last_changed_side = changed_side
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1

        flush_group()

    features, send_times, sessions = buffer.arrays()
    metadata: dict[str, object] = {
        "source": str(csv_path.resolve()),
        "reconstruction_version": RECONSTRUCTION_VERSION,
        "security_id": security_id,
        "source_columns": list(csv_columns),
        "security_column": security_column,
        "price_input": "decimal currency converted x1000 when needed; fixed-point otherwise",
        "crossed_book_policy": "remove crossed passive orders using the last updated side as aggressor",
        "feature_order": "ask_price,ask_size,bid_price,bid_size repeated for 10 levels",
        "feature_scale": FEATURE_SCALE,
        "snapshot_every_book_update_groups": snapshot_every,
        "levels": levels,
        "message_counts": dict(sorted(message_counts.items())),
        "accepted_update_counts": dict(sorted(accepted_updates.items())),
        "continuous_session_book_update_groups": dict(
            sorted(book_update_groups.items())
        ),
        "book_errors": dict(sorted(book.errors.items())),
        "invalid_rows": invalid_rows,
        "skipped_other_security_rows": skipped_security,
        "out_of_order_send_times": out_of_order,
        "snapshot_count": len(features),
        "session_counts": {
            str(value): int((sessions == value).sum()) for value in np.unique(sessions)
        },
        "min_send_time": int(send_times.min()) if len(send_times) else None,
        "max_send_time": int(send_times.max()) if len(send_times) else None,
    }
    return features, send_times, sessions, metadata


def main() -> None:
    args = parse_args()
    if args.snapshot_every < 1:
        raise ValueError("--snapshot-every must be at least 1")
    if args.levels != 10:
        raise ValueError("DeepLOB requires exactly 10 levels")
    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)

    features, send_times, sessions, metadata = reconstruct(
        args.csv, args.security_id, args.snapshot_every, args.levels
    )
    if not len(features):
        raise RuntimeError("No valid continuous-session snapshots were reconstructed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        send_times=send_times,
        sessions=sessions,
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
