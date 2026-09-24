#!/usr/bin/env python3
"""Measure realized clock time for fixed snapshot horizons at test quotes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = HERE / "output/gap_2m_books/filtered_manifest.json"
DATA = HERE / "output/direction_dataset"


def day_milliseconds(times: np.ndarray) -> np.ndarray:
    clock = times % 1_000_000_000
    return (((clock // 10_000_000 * 60 + (clock // 100_000) % 100) * 60 +
             (clock // 1_000) % 100) * 1_000 + clock % 1_000)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    dates = manifest["date_order"][1:]
    result = {}
    for horizon in (5, 10, 20):
        seconds = []
        for date in dates:
            with np.load(ROOT / manifest["book_cache_by_date"][date],
                         allow_pickle=False) as book, np.load(
                             DATA / f"day_{date}.npz", allow_pickle=False
                         ) as data:
                indices = data["quote_indices"]
                time_ms = day_milliseconds(book["send_times"])
                elapsed = (time_ms[indices + horizon] - time_ms[indices]) / 1000
                if np.any(elapsed < 0):
                    raise AssertionError(f"negative elapsed time on {date}")
                seconds.append(elapsed)
        values = np.concatenate(seconds)
        result[str(horizon)] = {
            "quotes": len(values), "median_seconds": float(np.median(values)),
            "p10_seconds": float(np.quantile(values, 0.1)),
            "p90_seconds": float(np.quantile(values, 0.9)),
        }
    (DATA / "horizon_duration.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
