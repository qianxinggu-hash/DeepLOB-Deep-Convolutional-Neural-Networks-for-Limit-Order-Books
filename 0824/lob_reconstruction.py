#!/usr/bin/env python3
"""Reliable HK 07709 L2 reconstruction helpers for the 0824 experiment.

The CSV files are market-by-order (MBO) messages.  The companion NPZ files are
market-by-price (MBP/L2) messages whose quote events contain the *absolute*
aggregate quantity at a price.  The 0824 experiment deliberately uses MBP for
the model input because an absolute level update can recover from an earlier
missing order event without guessing which individual order is stale.
"""

from __future__ import annotations

import json
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


CLEAR = 0xC0000003
ASK_LEVEL = 0xD0000001
ASK_EXECUTION = 0xD0000002
BID_LEVEL = 0xE0000001
BID_EXECUTION = 0xE0000002
EXPECTED_EVENT_CODES = {
    CLEAR,
    ASK_LEVEL,
    ASK_EXECUTION,
    BID_LEVEL,
    BID_EXECUTION,
}
LEVELS = 10
UTC_TO_HK_MS = 8 * 60 * 60 * 1000
DAY_MS = 24 * 60 * 60 * 1000
MBO_ADD = 30
MBO_MODIFY = 31
MBO_DELETE = 32
MBO_TRADE = 50
MBO_BID = 0
MBO_ASK = 1


def hk_session_from_utc_ms(exchange_ms: int) -> int:
    """Return 1/2 for the HK continuous morning/afternoon sessions."""

    local_ms = (exchange_ms + UTC_TO_HK_MS) % DAY_MS
    if (9 * 60 + 30) * 60_000 <= local_ms < 12 * 60 * 60_000:
        return 1
    if 13 * 60 * 60_000 <= local_ms < 16 * 60 * 60_000:
        return 2
    return 0


def compact_hk_time(exchange_ms: int) -> int:
    """Convert epoch milliseconds to YYYYMMDDhhmmssSSS in Asia/Shanghai."""

    hk = timezone(timedelta(hours=8))
    value = datetime.fromtimestamp(exchange_ms / 1000, tz=timezone.utc).astimezone(hk)
    return int(value.strftime("%Y%m%d%H%M%S") + f"{value.microsecond // 1000:03d}")


def compact_hk_time_to_utc_ms(send_time: int) -> int:
    """Convert YYYYMMDDhhmmssSSS in Hong Kong time to UTC epoch milliseconds."""

    value = str(send_time)
    if len(value) != 17:
        raise ValueError(f"invalid compact send time: {send_time}")
    local = datetime.strptime(value[:14], "%Y%m%d%H%M%S").replace(
        microsecond=int(value[14:]) * 1000,
        tzinfo=timezone(timedelta(hours=8)),
    )
    return int(local.timestamp() * 1000)


def hk_session_from_compact(send_time: int) -> int:
    """Return 1/2 for the HK continuous morning/afternoon sessions."""

    time_of_day = send_time % 1_000_000_000
    if 93_000_000 <= time_of_day < 120_000_000:
        return 1
    if 130_000_000 <= time_of_day < 160_000_000:
        return 2
    return 0


def _parse_price_hkd(value: str) -> float:
    if not value:
        return 0.0
    return float(value) if "." in value else int(value) / 1000.0


def audit_csv_trade_volume(csv_path: Path, security_id: int = 7709) -> dict[str, object]:
    """Stream one numeric HKEX CSV and total MsgType=50 printed quantity."""

    rows = 0
    trade_prints = 0
    trade_quantity = 0
    trade_notional_hkd = 0.0
    book_events = 0
    invalid_rows = 0
    first_send_time: int | None = None
    last_send_time: int | None = None
    message_counts: Counter[int] = Counter()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = stream.readline().rstrip("\r\n").split(",")
        index = {name: position for position, name in enumerate(header)}
        security_column = "SecurityId" if "SecurityId" in index else "SecurityCode"
        required = {"SendTime", "MsgType", security_column, "Price", "Quantity"}
        missing = required - set(index)
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")

        for raw_line in stream:
            values = raw_line.rstrip("\r\n").split(",")
            rows += 1
            try:
                if int(values[index[security_column]]) != security_id:
                    continue
                send_time = int(values[index["SendTime"]])
                message_type = int(values[index["MsgType"]])
                first_send_time = send_time if first_send_time is None else first_send_time
                last_send_time = send_time
                message_counts[message_type] += 1
                if message_type in (30, 31, 32):
                    book_events += 1
                elif message_type == 50:
                    quantity = int(values[index["Quantity"]])
                    price_hkd = _parse_price_hkd(values[index["Price"]])
                    trade_prints += 1
                    trade_quantity += quantity
                    trade_notional_hkd += price_hkd * quantity
            except (IndexError, TypeError, ValueError):
                invalid_rows += 1

    date = csv_path.stem.removeprefix("hk07709_")
    return {
        "date": date,
        "source": str(csv_path.resolve()),
        "rows": rows,
        "message_counts": dict(sorted(message_counts.items())),
        "book_events": book_events,
        "trade_prints": trade_prints,
        "trade_quantity": trade_quantity,
        "trade_notional_hkd": trade_notional_hkd,
        "invalid_rows": invalid_rows,
        "first_send_time": first_send_time,
        "last_send_time": last_send_time,
    }


def audit_and_select_highest_volume(
    raw_dir: Path, security_id: int = 7709
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records = [
        audit_csv_trade_volume(path, security_id)
        for path in sorted(raw_dir.glob("hk07709_*.csv"))
    ]
    if not records:
        raise FileNotFoundError(f"No hk07709_*.csv under {raw_dir}")
    selected = max(records, key=lambda row: int(row["trade_quantity"]))
    return records, selected


@dataclass
class ReconstructedLOB:
    features: np.ndarray
    exchange_ms: np.ndarray
    send_times: np.ndarray
    segments: np.ndarray
    metadata: dict[str, object]


class AbsoluteLevelBook:
    """Aggregated order book updated by absolute price-level quantities."""

    def __init__(self) -> None:
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()

    def set_level(self, side: str, price_mills: int, quantity: int) -> None:
        if price_mills <= 0 or quantity < 0:
            raise ValueError("price must be positive and quantity non-negative")
        levels = self.bids if side == "bid" else self.asks
        if quantity == 0:
            levels.pop(price_mills, None)
        else:
            levels[price_mills] = quantity

    def snapshot(self, levels: int = LEVELS) -> tuple[np.ndarray | None, str | None]:
        if len(self.bids) < levels or len(self.asks) < levels:
            return None, "insufficient_depth"
        bids = sorted(self.bids.items(), reverse=True)[:levels]
        asks = sorted(self.asks.items())[:levels]
        if bids[0][0] >= asks[0][0]:
            return None, "crossed_or_locked"
        output = np.empty(levels * 4, dtype=np.float32)
        for level, ((ask_price, ask_qty), (bid_price, bid_qty)) in enumerate(
            zip(asks, bids)
        ):
            start = level * 4
            output[start : start + 4] = (
                ask_price / 1000.0,
                ask_qty,
                bid_price / 1000.0,
                bid_qty,
            )
        return output, None


class StrictMBOBook:
    """Order-level state with no speculative repair of missing feed events."""

    def __init__(self) -> None:
        self.orders: dict[int, tuple[int, int, int]] = {}
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}
        self.errors: Counter[str] = Counter()
        self.tainted = False

    def _adjust(self, side: int, price: int, delta: int) -> bool:
        levels = self.bids if side == MBO_BID else self.asks
        quantity = levels.get(price, 0) + delta
        if quantity < 0:
            self.errors["negative_level_quantity"] += 1
            self.tainted = True
            return False
        if quantity:
            levels[price] = quantity
        else:
            levels.pop(price, None)
        return True

    def add(self, order_id: int, side: int, price: int, quantity: int) -> bool:
        if side not in (MBO_BID, MBO_ASK) or price <= 0 or quantity <= 0:
            self.errors["invalid_add"] += 1
            self.tainted = True
            return False
        if order_id in self.orders:
            self.errors["duplicate_add"] += 1
            self.tainted = True
            return False
        self.orders[order_id] = (side, price, quantity)
        return self._adjust(side, price, quantity)

    def modify(self, order_id: int, side: int, quantity: int) -> bool:
        old = self.orders.get(order_id)
        if old is None:
            self.errors["modify_missing_order"] += 1
            self.tainted = True
            return False
        old_side, price, old_quantity = old
        if side != old_side:
            self.errors["modify_side_mismatch"] += 1
            self.tainted = True
            return False
        if quantity <= 0:
            self.errors["invalid_modify_quantity"] += 1
            self.tainted = True
            return False
        self.orders[order_id] = (old_side, price, quantity)
        return self._adjust(old_side, price, quantity - old_quantity)

    def delete(self, order_id: int, side: int) -> bool:
        old = self.orders.get(order_id)
        if old is None:
            self.errors["delete_missing_order"] += 1
            self.tainted = True
            return False
        old_side, price, quantity = old
        if side != old_side:
            self.errors["delete_side_mismatch"] += 1
            self.tainted = True
            return False
        del self.orders[order_id]
        return self._adjust(old_side, price, -quantity)

    def snapshot(self, levels: int = LEVELS) -> tuple[np.ndarray | None, str | None]:
        if self.tainted:
            return None, "tainted_mbo_state"
        if len(self.bids) < levels or len(self.asks) < levels:
            return None, "insufficient_depth"
        bids = sorted(self.bids.items(), reverse=True)[:levels]
        asks = sorted(self.asks.items())[:levels]
        if bids[0][0] >= asks[0][0]:
            return None, "crossed_or_locked"
        output = np.empty(levels * 4, dtype=np.float32)
        for level, ((ask_price, ask_qty), (bid_price, bid_qty)) in enumerate(
            zip(asks, bids)
        ):
            start = level * 4
            output[start : start + 4] = (
                ask_price / 1000.0,
                ask_qty,
                bid_price / 1000.0,
                bid_qty,
            )
        return output, None


def reconstruct_mbo_csv(
    source: Path,
    snapshot_every: int = 10,
    levels: int = LEVELS,
    security_id: int = 7709,
) -> ReconstructedLOB:
    """Strictly replay an HKEX order-event CSV into sampled ten-level states.

    Add/modify/delete messages update the order map and aggregate price levels.
    Trade prints are informational and are not applied again.  Rows sharing one
    SendTime are applied atomically.  Any impossible order transition taints the
    state, after which snapshots are withheld instead of guessing a repair.
    """

    if snapshot_every < 1:
        raise ValueError("snapshot_every must be positive")
    if levels != LEVELS:
        raise ValueError("DeepLOB requires ten levels")

    book = StrictMBOBook()
    feature_rows: list[np.ndarray] = []
    send_time_rows: list[int] = []
    exchange_rows: list[int] = []
    segment_rows: list[int] = []
    counts: Counter[str] = Counter()
    message_counts: Counter[str] = Counter()
    sample_counter: Counter[int] = Counter()
    previous_send_time: int | None = None
    group_send_time: int | None = None
    group_changed = False
    csv_columns: tuple[str, ...] = ()

    def flush_group() -> None:
        nonlocal group_changed
        if group_send_time is None or not group_changed:
            return
        session = hk_session_from_compact(group_send_time)
        if not session:
            counts["changed_groups_outside_continuous_session"] += 1
            group_changed = False
            return
        counts[f"changed_groups_session_{session}"] += 1
        sample_counter[session] += 1
        if sample_counter[session] % snapshot_every == 0:
            counts["snapshot_attempts"] += 1
            state, reason = book.snapshot(levels)
            if state is None:
                counts[f"skipped_{reason}"] += 1
            else:
                feature_rows.append(state)
                send_time_rows.append(group_send_time)
                exchange_rows.append(compact_hk_time_to_utc_ms(group_send_time))
                segment_rows.append(session)
        group_changed = False

    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        csv_columns = tuple(reader.fieldnames or ())
        security_column = "SecurityId" if "SecurityId" in csv_columns else "SecurityCode"
        required = {
            "SendTime",
            "MsgType",
            security_column,
            "Price",
            "Quantity",
            "Side",
            "OrderID",
        }
        missing = required - set(csv_columns)
        if missing:
            raise ValueError(f"{source}: missing columns {sorted(missing)}")

        for row in reader:
            try:
                if int(row[security_column]) != security_id:
                    counts["skipped_other_security"] += 1
                    continue
                send_time = int(row["SendTime"])
                if previous_send_time is not None and send_time < previous_send_time:
                    counts["out_of_order_send_times"] += 1
                    book.tainted = True
                previous_send_time = send_time
                if group_send_time is None:
                    group_send_time = send_time
                elif send_time != group_send_time:
                    flush_group()
                    group_send_time = send_time

                message_type = int(row["MsgType"])
                message_counts[str(message_type)] += 1
                changed = False
                if message_type == MBO_ADD:
                    changed = book.add(
                        int(row["OrderID"]),
                        int(row["Side"]),
                        int(row["Price"]) if "." not in row["Price"] else int(round(float(row["Price"]) * 1000)),
                        int(row["Quantity"]),
                    )
                elif message_type == MBO_MODIFY:
                    changed = book.modify(
                        int(row["OrderID"]), int(row["Side"]), int(row["Quantity"])
                    )
                elif message_type == MBO_DELETE:
                    changed = book.delete(int(row["OrderID"]), int(row["Side"]))
                elif message_type != MBO_TRADE:
                    counts["unsupported_message_type"] += 1
                if changed:
                    group_changed = True
                    counts[f"accepted_message_{message_type}"] += 1
            except (KeyError, TypeError, ValueError):
                counts["invalid_rows"] += 1
                book.tainted = True

        flush_group()

    if not feature_rows:
        detail = dict(sorted(book.errors.items()))
        raise RuntimeError(f"no valid snapshots reconstructed; order errors={detail}")
    features = np.stack(feature_rows).astype(np.float32, copy=False)
    send_time_array = np.asarray(send_time_rows, dtype=np.int64)
    exchange_array = np.asarray(exchange_rows, dtype=np.int64)
    segment_array = np.asarray(segment_rows, dtype=np.int16)
    mids = (features[:, 0].astype(np.float64) + features[:, 2]) / 2
    spreads = features[:, 0].astype(np.float64) - features[:, 2]
    metadata: dict[str, object] = {
        "source": str(source.resolve()),
        "source_kind": "market-by-order (MBO) order event stream",
        "reconstruction_version": "0824-v2-mbo-strict",
        "source_columns": list(csv_columns),
        "message_counts": dict(sorted(message_counts.items())),
        "snapshot_every_changed_send_time_groups": snapshot_every,
        "atomic_group": "all rows with the same SendTime",
        "levels": levels,
        "feature_order": "ask_price_hkd,ask_size,bid_price_hkd,bid_size repeated for levels 1..10",
        "state_policy": {
            "add": "insert a new OrderID and add its quantity to its side/price level",
            "modify": "replace quantity of an existing OrderID and apply only the delta",
            "delete": "remove an existing OrderID and subtract its remaining quantity",
            "trade": "ignored for state because the order update/delete is delivered separately",
            "same_send_time": "apply atomically before sampling",
            "impossible_transition": "taint state and withhold all later snapshots; never overwrite or guess",
            "crossed_or_locked": "skip observation without mutating the book",
        },
        "quality_counts": dict(sorted(counts.items())),
        "order_errors": dict(sorted(book.errors.items())),
        "final_state_tainted": book.tainted,
        "snapshot_count": int(len(features)),
        "segments": {
            str(segment): int(np.sum(segment_array == segment))
            for segment in np.unique(segment_array)
        },
        "first_send_time": int(send_time_array[0]),
        "last_send_time": int(send_time_array[-1]),
        "mid_price_hkd": {
            "min": float(mids.min()),
            "median": float(np.median(mids)),
            "max": float(mids.max()),
        },
        "spread_hkd": {
            "min": float(spreads.min()),
            "median": float(np.median(spreads)),
            "max": float(spreads.max()),
        },
    }
    return ReconstructedLOB(
        features=features,
        exchange_ms=exchange_array,
        send_times=send_time_array,
        segments=segment_array,
        metadata=metadata,
    )


def reconstruct_l2_npz(
    source: Path,
    snapshot_every: int = 10,
    levels: int = LEVELS,
) -> ReconstructedLOB:
    """Replay an absolute MBP/L2 event stream into sampled ten-level states.

    All events sharing the same exchange millisecond are applied atomically.
    Sampling happens only after a complete group and only during continuous
    trading.  A crossed/locked observation is skipped; the book is never
    mutated by a speculative repair.
    """

    if snapshot_every < 1:
        raise ValueError("snapshot_every must be positive")
    if levels != LEVELS:
        raise ValueError("DeepLOB requires ten levels")

    with np.load(source, allow_pickle=False) as archive:
        if archive.files != ["data"]:
            raise ValueError(f"{source}: expected only a 'data' array, found {archive.files}")
        events = archive["data"]
    required_fields = {"ev", "exch_ts", "px", "qty"}
    fields = set(events.dtype.names or ())
    if not required_fields <= fields:
        raise ValueError(f"{source}: missing event fields {sorted(required_fields - fields)}")
    if not len(events):
        raise ValueError(f"{source}: empty event stream")

    exchange_ns = events["exch_ts"].astype(np.int64, copy=False)
    if np.any(np.diff(exchange_ns) < 0):
        raise ValueError("event timestamps are not monotonic; replay order is ambiguous")
    event_codes = events["ev"].astype(np.uint64, copy=False)
    unknown_codes = set(map(int, np.unique(event_codes))) - EXPECTED_EVENT_CODES
    if unknown_codes:
        raise ValueError(f"unknown event codes: {[hex(code) for code in sorted(unknown_codes)]}")

    book = AbsoluteLevelBook()
    feature_rows: list[np.ndarray] = []
    exchange_rows: list[int] = []
    send_time_rows: list[int] = []
    segment_rows: list[int] = []
    counts: Counter[str] = Counter()
    event_count_by_code: Counter[str] = Counter()
    segment_ids: dict[tuple[int, int], int] = {}
    sample_counter: Counter[tuple[int, int]] = Counter()
    clear_generation = 0

    group_ms: int | None = None
    group_changed = False

    def segment_id(session: int) -> int:
        key = (session, clear_generation)
        if key not in segment_ids:
            segment_ids[key] = len(segment_ids) + 1
        return segment_ids[key]

    def flush_group() -> None:
        nonlocal group_changed
        if group_ms is None or not group_changed:
            return
        session = hk_session_from_utc_ms(group_ms)
        if not session:
            counts["changed_groups_outside_continuous_session"] += 1
            group_changed = False
            return
        key = (session, clear_generation)
        sample_counter[key] += 1
        counts[f"changed_groups_session_{session}"] += 1
        if sample_counter[key] % snapshot_every == 0:
            counts["snapshot_attempts"] += 1
            state, reason = book.snapshot(levels)
            if state is None:
                counts[f"skipped_{reason}"] += 1
            else:
                feature_rows.append(state)
                exchange_rows.append(group_ms)
                send_time_rows.append(compact_hk_time(group_ms))
                segment_rows.append(segment_id(session))
        group_changed = False

    for row in events:
        exchange_ms = int(row["exch_ts"]) // 1_000_000
        if group_ms is None:
            group_ms = exchange_ms
        elif exchange_ms != group_ms:
            flush_group()
            group_ms = exchange_ms

        code = int(row["ev"])
        event_count_by_code[hex(code)] += 1
        if code == CLEAR:
            book.clear()
            clear_generation += 1
            counts["clear_events"] += 1
            group_changed = False
        elif code == ASK_LEVEL:
            book.set_level(
                "ask", int(round(float(row["px"]) * 1000)), int(round(float(row["qty"])))
            )
            group_changed = True
        elif code == BID_LEVEL:
            book.set_level(
                "bid", int(round(float(row["px"]) * 1000)), int(round(float(row["qty"])))
            )
            group_changed = True
        elif code in (ASK_EXECUTION, BID_EXECUTION):
            counts["execution_events_ignored_for_state"] += 1

    flush_group()

    if not feature_rows:
        raise RuntimeError("no valid continuous-session LOB snapshots were reconstructed")
    features = np.stack(feature_rows).astype(np.float32, copy=False)
    exchange_array = np.asarray(exchange_rows, dtype=np.int64)
    send_time_array = np.asarray(send_time_rows, dtype=np.int64)
    segment_array = np.asarray(segment_rows, dtype=np.int16)
    mids = (features[:, 0].astype(np.float64) + features[:, 2]) / 2
    spreads = features[:, 0].astype(np.float64) - features[:, 2]

    metadata: dict[str, object] = {
        "source": str(source.resolve()),
        "source_kind": "market-by-price (MBP/L2) absolute level event stream",
        "reconstruction_version": "0824-v1",
        "event_rows": int(len(events)),
        "event_counts": dict(sorted(event_count_by_code.items())),
        "snapshot_every_changed_millisecond_groups": snapshot_every,
        "atomic_group": "all events with the same exch_ts millisecond",
        "levels": levels,
        "feature_order": "ask_price_hkd,ask_size,bid_price_hkd,bid_size repeated for levels 1..10",
        "state_policy": {
            "level_event": "overwrite aggregate quantity at side and price; zero deletes level",
            "execution_event": "ignored for resting state because level events already contain post-event absolute depth",
            "clear_event": "clear both sides and start a new segment",
            "crossed_or_locked": "skip observation without mutating the book",
            "insufficient_depth": "skip until both sides contain at least ten levels",
        },
        "quality_counts": dict(sorted(counts.items())),
        "snapshot_count": int(len(features)),
        "segments": {
            str(segment): int(np.sum(segment_array == segment))
            for segment in np.unique(segment_array)
        },
        "first_send_time": int(send_time_array[0]),
        "last_send_time": int(send_time_array[-1]),
        "mid_price_hkd": {
            "min": float(mids.min()),
            "median": float(np.median(mids)),
            "max": float(mids.max()),
        },
        "spread_hkd": {
            "min": float(spreads.min()),
            "median": float(np.median(spreads)),
            "max": float(spreads.max()),
        },
    }
    return ReconstructedLOB(
        features=features,
        exchange_ms=exchange_array,
        send_times=send_time_array,
        segments=segment_array,
        metadata=metadata,
    )


def save_reconstruction(output: Path, result: ReconstructedLOB) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=result.features,
        exchange_ms=result.exchange_ms,
        send_times=result.send_times,
        segments=result.segments,
        metadata=np.array(json.dumps(result.metadata, ensure_ascii=False)),
    )
