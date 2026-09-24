#!/usr/bin/env python3
"""Reconstruct July gap days with a two-minute post-gap quarantine.

The original tolerant caches quarantine five minutes after each feed gap.
This creates separate caches so the requested two-minute sensitivity does
not alter the earlier backtests.  Pre-gap masking is applied downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0824"))
from lob_reconstruction import reconstruct_mbo_csv, save_reconstruction  # noqa: E402

SOURCE_AUDIT = HERE / "output/gamma_results.json"
DEFAULT_OUTPUT = HERE / "output/gap_2m_books"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path,
                        default=Path.home() / "Downloads/7709_202607")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source = json.loads(SOURCE_AUDIT.read_text())
    eligible = set(source["provenance"]["clean_trading_dates"])
    gap_dates = {
        row["date"] for row in source["provenance"]["reconstruction"]
        if row["date"] in eligible and row.get("gap_events")
    }
    selected = sorted(gap_dates if args.dates is None else args.dates)
    if not selected or not set(selected) <= gap_dates:
        parser.error("dates must be among the eligible July gap dates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for date in selected:
        raw = args.raw_dir / f"hk07709_{date}.csv"
        if not raw.is_file():
            raise FileNotFoundError(raw)
        destination = args.output_dir / f"hk07709_{date}_mbo_tolerant_recovery_2m_s10.npz"
        if args.force or not destination.exists():
            book = reconstruct_mbo_csv(raw, snapshot_every=10,
                                       policy="tolerant", recovery_ms=120_000)
            save_reconstruction(destination, book)
            metadata = book.metadata
            status = "reconstructed"
        else:
            with np.load(destination, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
            status = "cached"
        if metadata.get("recovery_ms") != 120_000:
            raise AssertionError(f"{date}: unexpected recovery_ms")
        record = {
            "date": date,
            "status": status,
            "cache": str(destination),
            "snapshots": int(metadata["snapshot_count"]),
            "gap_count": len(metadata.get("gap_events", [])),
            "order_errors": metadata.get("order_errors", {}),
            "final_state_tainted": bool(metadata.get("final_state_tainted", False)),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    manifest = {
        "source_audit": str(SOURCE_AUDIT),
        "raw_dir": str(args.raw_dir),
        "policy": "tolerant",
        "gap_threshold_ms": 30_000,
        "post_gap_recovery_ms": 120_000,
        "pre_gap_exclusion_ms": 120_000,
        "pre_gap_exclusion_applied_in": "0920/gap_window_filter.py",
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
