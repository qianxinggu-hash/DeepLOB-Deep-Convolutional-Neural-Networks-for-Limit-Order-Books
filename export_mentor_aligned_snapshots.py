#!/usr/bin/env python3
"""Replay mentor L2 quotes into 10-level snapshots aligned to our s10 times.

Event codes ending in ``1`` are absolute price-level quote updates. Codes
ending in ``2`` are execution/trade events and must not persist in the book.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sortedcontainers import SortedDict

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data/processed/7709"
OURS_NPZ = PROCESSED / "hk07709_2026-08-07_s10_lob.npz"
MENT_NPZ = PROCESSED / "hk07709_2026-08-07.npz"
OUT_CSV = PROCESSED / "hk07709_2026-08-07_mentor_s10_lob.csv"
DESKTOP_CSV = Path.home() / "Desktop/7709_csv/hk07709_2026-08-07_mentor_s10_lob.csv"

CLEAR = 0xC0000003
ASK_LEVEL, ASK_EXECUTION = 0xD0000001, 0xD0000002
BID_LEVEL, BID_EXECUTION = 0xE0000001, 0xE0000002
UTC_MIDNIGHT_MS = 1_786_060_800_000
LEVELS = 10

SNAPSHOT_COLUMNS = ["send_time", "session"]
for level in range(1, LEVELS + 1):
    SNAPSHOT_COLUMNS.extend(
        [
            f"ask_price_{level}_hkd",
            f"ask_size_{level}",
            f"bid_price_{level}_hkd",
            f"bid_size_{level}",
        ]
    )
SNAPSHOT_COLUMNS.extend(["crossed", "n_bid", "n_ask"])


def hk_send_to_utc_ms(send: np.ndarray) -> np.ndarray:
    tod = send.astype(np.int64) % 1_000_000_000
    hour = tod // 10_000_000
    minute = (tod % 10_000_000) // 100_000
    second = (tod % 100_000) // 1_000
    millis = tod % 1_000
    utc_hour = hour - 8
    day_ms = ((utc_hour * 60 + minute) * 60 + second) * 1000 + millis
    return UTC_MIDNIGHT_MS + day_ms


def fmt(value: float | str) -> str:
    if value == "":
        return ""
    return f"{float(value):.6g}"


class L2Book:
    def __init__(self) -> None:
        self.bids: SortedDict[float, float] = SortedDict()
        self.asks: SortedDict[float, float] = SortedDict()

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()

    def set_level(self, side: str, price: float, qty: float) -> None:
        book = self.bids if side == "bid" else self.asks
        if qty <= 0:
            book.pop(price, None)
        else:
            book[price] = qty

    def apply(self, ev: int, price: float, qty: float) -> None:
        if ev == CLEAR:
            self.clear()
        elif ev == ASK_LEVEL:
            self.set_level("ask", price, qty)
        elif ev == BID_LEVEL:
            self.set_level("bid", price, qty)
        elif ev in (ASK_EXECUTION, BID_EXECUTION):
            # Execution messages describe traded quantity, not resting depth.
            return

    def snapshot(self) -> tuple[list[object], int, int, int]:
        bids_all = list(reversed(self.bids.items()))
        asks_all = list(self.asks.items())
        raw_crossed = int(
            bool(bids_all) and bool(asks_all) and bids_all[0][0] >= asks_all[0][0]
        )
        bids = bids_all[:LEVELS]
        asks = asks_all[:LEVELS]
        row: list[object] = []
        for index in range(LEVELS):
            ask_px, ask_qty = asks[index] if index < len(asks) else ("", "")
            bid_px, bid_qty = bids[index] if index < len(bids) else ("", "")
            row.extend([ask_px, ask_qty, bid_px, bid_qty])
        return row, raw_crossed, len(self.bids), len(self.asks)


def main() -> None:
    ours = np.load(OURS_NPZ, allow_pickle=False)
    send_times = ours["send_times"].astype(np.int64)
    sessions = ours["sessions"].astype(np.int64)
    our_ms = hk_send_to_utc_ms(send_times)

    mentor = np.load(MENT_NPZ, allow_pickle=False)["data"]
    ment_ms = mentor["exch_ts"].astype(np.int64) // 1_000_000
    ev = mentor["ev"].astype(np.uint64)
    px = mentor["px"].astype(np.float64)
    qty = mentor["qty"].astype(np.float64)

    book = L2Book()
    cursor = 0
    n_events = len(mentor)
    n_snap = len(send_times)
    crossed_count = 0
    print(f"Replaying {n_events:,} mentor events onto {n_snap:,} s10 timestamps", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    DESKTOP_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SNAPSHOT_COLUMNS)
        for i in range(n_snap):
            target = int(our_ms[i])
            while cursor < n_events and int(ment_ms[cursor]) <= target:
                book.apply(int(ev[cursor]), float(px[cursor]), float(qty[cursor]))
                cursor += 1
            levels, crossed, n_bid, n_ask = book.snapshot()
            crossed_count += crossed
            writer.writerow(
                [int(send_times[i]), int(sessions[i])]
                + [fmt(value) for value in levels]
                + [crossed, n_bid, n_ask]
            )
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1:,}/{n_snap:,}", flush=True)

    DESKTOP_CSV.write_bytes(OUT_CSV.read_bytes())
    print(
        f"Saved {OUT_CSV} ({OUT_CSV.stat().st_size / 1e6:.1f} MB); "
        f"crossed snapshots={crossed_count:,}/{n_snap:,}",
        flush=True,
    )
    print(f"Copied {DESKTOP_CSV}", flush=True)


if __name__ == "__main__":
    main()
