#!/usr/bin/env python3
"""Cache causal July features and aligned 5/10/20-snapshot direction labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0824"))
sys.path.insert(0, str(ROOT / "0906"))

from run_direct_drift_signal import prepare_day  # noqa: E402
from run_multiday_strategy import load_day  # noqa: E402


MANIFEST = HERE / "output/gap_2m_books/filtered_manifest.json"
SOURCE = HERE / "output/new_hybrid_signal"
OUTPUT = HERE / "output/direction_dataset"
HORIZONS = (5, 10, 20)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for date in manifest["date_order"]:
        path = ROOT / manifest["book_cache_by_date"][date]
        prepared = prepare_day(date, path, safe_eligibility=True)
        valid = prepared.has_100_snapshot_history
        indices = prepared.indices[valid]
        with np.load(SOURCE / "embeddings" / f"{date}.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["quote_indices"], indices):
                raise AssertionError(f"embedding indices differ on {date}")
            embedding = archive["embedding"].copy()
        l2 = np.column_stack((embedding, prepared.features)).astype(np.float32)
        l3 = np.empty((len(indices), 0), dtype=np.float32)
        if prepared.zero_recorded_errors:
            with np.load(SOURCE / "l3_safe" / f"{date}.npz", allow_pickle=False) as archive:
                if not np.array_equal(archive["quote_indices"], prepared.indices):
                    raise AssertionError(f"L3 indices differ on {date}")
                l3 = archive["l3_features"][valid].copy()
        day = load_day(date, "tolerant_mbo_gap2m", path)
        midpoint = (day.book[:, 0].astype(np.float64) + day.book[:, 2]) / 2.0
        targets = {}
        for horizon in HORIZONS:
            if np.any(day.segments[indices] != day.segments[indices + horizon]):
                raise AssertionError(f"horizon {horizon} crosses segment on {date}")
            raw = (midpoint[indices + horizon] - midpoint[indices]) / prepared.tick_hkd
            target = np.rint(raw * 2) / 2
            if np.max(np.abs(raw - target)) > 0.01:
                raise AssertionError(f"target not on half-tick grid on {date}")
            targets[f"move_{horizon}_ticks"] = target.astype(np.float32)
        if not np.allclose(targets["move_20_ticks"],
                           prepared.actual_move_ticks[valid], atol=0.01):
            raise AssertionError(f"20-snapshot target differs on {date}")
        if not np.isfinite(l2).all() or not np.isfinite(l3).all():
            raise AssertionError(f"nonfinite features on {date}")
        output = OUTPUT / f"day_{date}.npz"
        np.savez_compressed(
            output, quote_indices=indices, l2=l2, l3=l3,
            microprice_move_ticks=prepared.microprice_move_ticks[valid].astype(np.float32),
            tick_hkd=np.asarray(prepared.tick_hkd),
            l3_available=np.asarray(prepared.zero_recorded_errors),
            **targets,
        )
        rows.append({"date": date, "quote_count": len(indices),
                     "l3_available": prepared.zero_recorded_errors,
                     "tick_hkd": prepared.tick_hkd,
                     "target_20_down": int(np.count_nonzero(targets["move_20_ticks"] < 0)),
                     "target_20_flat": int(np.count_nonzero(targets["move_20_ticks"] == 0)),
                     "target_20_up": int(np.count_nonzero(targets["move_20_ticks"] > 0))})
        print("CACHED", date, len(indices), "L3", prepared.zero_recorded_errors,
              flush=True)
    result = {
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "source_signal": str((SOURCE / "results.json").relative_to(ROOT)),
        "horizons": HORIZONS,
        "label": "sign of rounded half-tick endpoint midpoint move; 0 is flat",
        "features": "64 frozen DeepLOB embedding + 28 causal L2; optional six OrderID lifecycle features",
        "dates": rows,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
