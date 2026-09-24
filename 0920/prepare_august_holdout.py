#!/usr/bin/env python3
"""Prepare the later August MBO and trade caches used in drift diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0824"))
from directional_market_maker import load_or_extract_trades  # noqa: E402
from lob_reconstruction import reconstruct_mbo_csv, save_reconstruction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", default=["2026-08-04", "2026-08-07"],
                        choices=["2026-08-04", "2026-08-07"])
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "7709_tickdata")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "output/august_holdout")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for date in args.dates:
        raw = args.raw_dir / f"hk07709_{date}.csv"
        if not raw.is_file():
            raise FileNotFoundError(raw)
        book_path = args.output_dir / f"hk07709_{date}_mbo_tolerant_5m_s10.npz"
        if args.force or not book_path.exists():
            book = reconstruct_mbo_csv(raw, snapshot_every=10, policy="tolerant")
            save_reconstruction(book_path, book)
            metadata = book.metadata
        else:
            with np.load(book_path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
        trade_path = args.output_dir / f"hk07709_{date}_trades.npz"
        load_or_extract_trades(raw, trade_path)
        print(json.dumps({
            "date": date,
            "book_cache": str(book_path),
            "trade_cache": str(trade_path),
            "snapshots": metadata.get("snapshot_count"),
            "feed_gap_count": len(metadata.get("gap_events", [])),
            "order_errors": metadata.get("order_errors", {}),
            "final_state_tainted": metadata.get("final_state_tainted", False),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
