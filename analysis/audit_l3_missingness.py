#!/usr/bin/env python3
"""Audit local 07709 MBO CSV fields and archived reconstruction diagnostics.

Run from the repository root with ``.venv/bin/python analysis/audit_l3_missingness.py``.
The CSV scan treats blanks by message type; a blank OrderID on a trade print
is not considered a missing order event. Time gaps are observations, not a
count of missing messages.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "7709_tickdata"
OUTPUT = ROOT / "analysis/l3_missingness_audit.json"
JULY_DIR = ROOT / "0824/data/july_2026"
AUGUST_DIR = ROOT / "0920/output/august_holdout"
GAP_THRESHOLD_MS = 30_000
REQUIRED_BY_TYPE = {
    "30": ("SendTime", "MsgType", "SecurityId", "Price", "Quantity", "Side", "OrderID"),
    "31": ("SendTime", "MsgType", "SecurityId", "Quantity", "Side", "OrderID"),
    "32": ("SendTime", "MsgType", "SecurityId", "Side", "OrderID"),
    "50": ("SendTime", "MsgType", "SecurityId", "Price", "Quantity"),
}


def day_ms(value: str) -> int:
    value = value.zfill(17)
    return ((int(value[8:10]) * 60 + int(value[10:12])) * 60 + int(value[12:14])) * 1000 + int(value[14:17])


def session(value: str) -> int:
    t = day_ms(value)
    if 34_200_000 <= t < 43_200_000:
        return 1
    if 46_800_000 <= t < 57_600_000:
        return 2
    return 0


def scan_csv(path: Path) -> dict:
    counts = Counter()
    blanks = defaultdict(Counter)
    required_blanks = defaultdict(Counter)
    malformed = Counter()
    gaps = []
    previous_time = None
    previous_session = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        security_column = "SecurityId" if "SecurityId" in columns else "SecurityCode"
        for row in reader:
            counts["rows"] += 1
            kind = row.get("MsgType", "") or "<blank>"
            counts[f"type_{kind}"] += 1
            for column in columns:
                if not row.get(column):
                    blanks[kind][column] += 1
            for column in REQUIRED_BY_TYPE.get(kind, ()):
                actual_column = security_column if column == "SecurityId" else column
                if not row.get(actual_column):
                    required_blanks[kind][actual_column] += 1
            if None in row:
                malformed["extra_fields"] += 1
            stamp = row.get("SendTime", "")
            if not stamp or len(stamp) != 17 or not stamp.isdigit():
                malformed["invalid_send_time"] += 1
                continue
            if stamp[:8] != path.stem[-10:].replace("-", ""):
                malformed["wrong_date"] += 1
            current_session = session(stamp)
            if previous_time is not None and stamp < previous_time:
                malformed["out_of_order_send_time"] += 1
            if previous_time is not None and current_session and current_session == previous_session:
                delta = day_ms(stamp) - day_ms(previous_time)
                if delta > GAP_THRESHOLD_MS:
                    gaps.append({"before": previous_time, "after": stamp, "gap_ms": delta})
            previous_time = stamp
            previous_session = current_session
    return {
        "file": str(path.relative_to(ROOT)),
        "columns": columns,
        "counts": dict(sorted(counts.items())),
        "blank_cells_by_type": {k: dict(sorted(v.items())) for k, v in sorted(blanks.items())},
        "required_blank_cells_by_type": {k: dict(sorted(v.items())) for k, v in sorted(required_blanks.items())},
        "malformed": dict(sorted(malformed.items())),
        "same_session_gaps_over_30s": gaps,
    }


def cache_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
    kinds = meta.get("message_counts", {})
    errors = meta.get("order_errors", {})
    gaps = meta.get("gap_events", [])
    return {
        "cache": str(path.relative_to(ROOT)),
        "raw_source_available": (RAW_DIR / (path.name[:18] + ".csv")).exists(),
        "message_counts": kinds,
        "order_errors": errors,
        "modify_delete_attempts": int(kinds.get("31", 0)) + int(kinds.get("32", 0)),
        "unknown_order_events": int(errors.get("modify_missing_order", 0)) + int(errors.get("delete_missing_order", 0)),
        "gap_events": gaps,
        "snapshot_count": meta.get("snapshot_count"),
        "final_state_tainted": meta.get("final_state_tainted"),
        "first_snapshot": meta.get("first_send_time"),
        "last_snapshot": meta.get("last_send_time"),
        "quality_counts": meta.get("quality_counts", {}),
    }


def main() -> None:
    raw = {p.stem[-10:]: scan_csv(p) for p in sorted(RAW_DIR.glob("hk07709_*.csv"))}
    cache_files = sorted(JULY_DIR.glob("*_mbo_tolerant_5m_s10.npz")) + sorted(AUGUST_DIR.glob("*_mbo_tolerant_5m_s10.npz"))
    caches = {p.name[8:18]: cache_metadata(p) for p in cache_files}
    result = {
        "definitions": {
            "raw_field_missing": "Empty CSV cell, reported by MsgType; required fields follow replay needs, so optional trade-only/order-only columns are excluded from the required-field rate.",
            "unknown_order_event": "MsgType 31 or 32 refers to an OrderID absent from the current tolerant MBO reconstruction state; counts events, not provably lost messages.",
            "time_gap": "Two consecutive SendTime values in the same continuous HK trading session differ by more than 30,000 ms; this detects suspicious quiet intervals, not exact missing-message counts.",
            "cache_scope": "The July/August reconstruction metadata may come from raw files not present in this checkout; local raw availability is reported per date.",
        },
        "raw": raw,
        "caches": caches,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
