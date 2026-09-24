#!/usr/bin/env python3
"""Build the 20-day July cache manifest after two-minute gap recovery.

Run only after the gap-date raw MBO files have been reconstructed with
``recovery_ms=120000`` in ``0920/output/gap_2m_books``.  For those dates this
script deletes the closed interval from two minutes before the last pre-gap
message through two minutes after the first resumed message.  Gap-free dates
continue to use the original cache.  Paths in the manifest are repo-relative.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from gap_window_filter import (
    enforce_safe_day_eligibility,
    save_gap_filtered_reconstruction,
)


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "0824"))
from run_multiday_strategy import load_day  # noqa: E402

OUTPUT_DIR = ROOT / "0920/output/gap_2m_books"
GAMMA_RESULT = ROOT / "0920/output/gamma_results.json"


def _metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"]))


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def build_manifest() -> dict:
    provenance = json.loads(GAMMA_RESULT.read_text(encoding="utf-8"))["provenance"]
    dates = provenance["clean_trading_dates"]
    source_by_date = {row["date"]: row for row in provenance["reconstruction"]}
    book_by_date: dict[str, str] = {}
    trade_by_date: dict[str, str] = {}
    audit_by_date: dict[str, dict] = {}
    for date in dates:
        original = ROOT / f"0824/data/july_2026/hk07709_{date}_mbo_tolerant_5m_s10.npz"
        trade = ROOT / f"0824/data/july_2026/hk07709_{date}_trades.npz"
        if not original.is_file() or not trade.is_file():
            raise FileNotFoundError(f"{date}: original book or trade cache missing")
        expected_gap_count = len(source_by_date[date].get("gap_events") or [])
        if expected_gap_count:
            rebuilt = OUTPUT_DIR / f"hk07709_{date}_mbo_tolerant_recovery_2m_s10.npz"
            filtered = OUTPUT_DIR / f"hk07709_{date}_mbo_tolerant_gap_blackout_2m_s10.npz"
            if not rebuilt.is_file():
                raise FileNotFoundError(f"{date}: 2-minute recovery cache missing: {rebuilt}")
            audit = save_gap_filtered_reconstruction(rebuilt, filtered)
            if audit["gap_count"] != expected_gap_count:
                raise ValueError(f"{date}: gap count changed on rebuild")
            book = filtered
            audit["cache_choice"] = "rebuilt_and_blackout_filtered"
        else:
            metadata = _metadata(original)
            if metadata.get("gap_events"):
                raise ValueError(f"{date}: source audit and original cache disagree on gaps")
            book = original
            audit = {
                "date": date,
                "gap_count": 0,
                "original_snapshot_count": int(metadata["snapshot_count"]),
                "removed_snapshot_count": 0,
                "retained_snapshot_count": int(metadata["snapshot_count"]),
                "cache_choice": "original_gap_free",
            }
        book_by_date[date] = _relative(book)
        trade_by_date[date] = _relative(trade)
        loaded = load_day(date, "tolerant_mbo", book)
        safe, eligible_audit = enforce_safe_day_eligibility(loaded)
        if not len(safe.eligible):
            raise ValueError(f"{date}: no safe eligible quotes after blackout")
        eligible = safe.eligible
        if not np.all(safe.segments[eligible - 100] == safe.segments[eligible]):
            raise AssertionError(f"{date}: 100-snapshot feature history crosses segment")
        if not np.all(safe.segments[eligible + 20] == safe.segments[eligible]):
            raise AssertionError(f"{date}: 20-snapshot label crosses segment")
        audit["eligible_quotes_before_boundary_check"] = eligible_audit["quote_count"]
        audit["eligible_quotes_after_boundary_check"] = eligible_audit[
            "retained_quote_count"
        ]
        audit["removed_eligible_boundary_quotes"] = eligible_audit[
            "removed_quote_count"
        ]
        audit_by_date[date] = audit

    if len(dates) != 20 or len(book_by_date) != 20:
        raise ValueError("expected 20 unique source-audit dates")
    return {
        "rule": "Exclude the closed interval from 2 minutes before previous_send_time through 2 minutes after resume_send_time for each recorded same-session gap; use recovery_ms=120000 reconstruction on gap dates.",
        "date_order": dates,
        "training_protocol": "Rolling prior completed dates only; target and 100-snapshot feature histories must stay within retained segment.",
        "book_cache_by_date": book_by_date,
        "trade_cache_by_date": trade_by_date,
        "audit_by_date": audit_by_date,
        "summary": {
            "dates": len(dates),
            "gap_dates": sum(row["gap_count"] > 0 for row in audit_by_date.values()),
            "gap_events": sum(row["gap_count"] for row in audit_by_date.values()),
            "removed_snapshots": sum(
                row["removed_snapshot_count"] for row in audit_by_date.values()
            ),
            "retained_snapshots": sum(
                row["retained_snapshot_count"] for row in audit_by_date.values()
            ),
            "safe_eligible_quotes": sum(
                row["eligible_quotes_after_boundary_check"]
                for row in audit_by_date.values()
            ),
            "removed_eligible_boundary_quotes": sum(
                row["removed_eligible_boundary_quotes"]
                for row in audit_by_date.values()
            ),
        },
    }


def main() -> None:
    manifest = build_manifest()
    destination = OUTPUT_DIR / "filtered_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(destination)
    print(json.dumps(manifest["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
