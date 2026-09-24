#!/usr/bin/env python3
"""Compare July new-hybrid markout RMSE with a zero-change forecast."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIGNAL = HERE / "output/new_hybrid_signal/results.json"
MANIFEST = HERE / "output/gap_2m_books/filtered_manifest.json"
OUTPUT = HERE / "output/new_hybrid_signal/zero_baseline_diagnostic.json"


def metrics(y: np.ndarray, book: np.ndarray, fused: np.ndarray,
            spread: np.ndarray, ticks: np.ndarray,
            causal_seconds: np.ndarray) -> dict:
    zero_rmse = float(np.sqrt(np.mean(np.square(y))))
    fused_rmse = float(np.sqrt(np.mean(np.square(y - fused))))
    return {
        "quote_count": int(len(y)),
        "zero_change_rmse_ticks": zero_rmse,
        "book_rmse_ticks": float(np.sqrt(np.mean(np.square(y - book)))),
        "new_hybrid_rmse_ticks": fused_rmse,
        "new_hybrid_mse_increase_vs_zero_pct": float(
            100 * (fused_rmse**2 / zero_rmse**2 - 1)
        ),
        "zero_change_mae_ticks": float(np.mean(np.abs(y))),
        "new_hybrid_mae_ticks": float(np.mean(np.abs(y - fused))),
        "actual_move_std_ticks": float(np.std(y)),
        "new_hybrid_prediction_std_ticks": float(np.std(fused)),
        "actual_abs_move_at_least_3_ticks_share": float(np.mean(np.abs(y) >= 2.999)),
        "new_hybrid_prediction_at_clip_share": float(np.mean(np.abs(fused) >= 2.999999)),
        "median_spread_ticks": float(np.median(spread)),
        "median_causal_horizon_seconds": float(np.median(causal_seconds)),
        "tick_hkd_values": {f"{t:g}": int(np.count_nonzero(ticks == t))
                            for t in np.unique(ticks)},
    }


def main() -> None:
    signal = json.loads(SIGNAL.read_text())
    manifest = json.loads(MANIFEST.read_text())
    pooled = {"all": [], "l3_available": [], "l3_unavailable": []}
    daily = []
    for row in signal["daily"]:
        date = row["date"]
        with np.load(HERE / "output/new_hybrid_signal/predictions" /
                     f"day_{date}.npz", allow_pickle=False) as prediction, np.load(
                         ROOT / manifest["book_cache_by_date"][date], allow_pickle=False
                     ) as book_cache:
            valid = prediction["has_100_snapshot_history"]
            indices = prediction["quote_indices"][valid]
            tick = float(prediction["tick_hkd"])
            book = book_cache["features"][indices]
            values = (
                prediction["actual_move_ticks_diagnostic"][valid].astype(np.float64),
                prediction["book_only_expected_move_ticks"][valid].astype(np.float64),
                prediction["expected_move_ticks"][valid].astype(np.float64),
                (book[:, 0] - book[:, 2]).astype(np.float64) / tick,
                np.full(np.count_nonzero(valid), tick, dtype=np.float64),
                prediction["causal_horizon_seconds"][valid].astype(np.float64),
            )
        calculated = metrics(*values)
        if not np.isclose(calculated["book_rmse_ticks"], row["book_only"]["rmse_ticks"],
                          rtol=0, atol=1e-9):
            raise AssertionError(f"book RMSE differs on {date}")
        if not np.isclose(calculated["new_hybrid_rmse_ticks"],
                          row["new_hybrid"]["rmse_ticks"], rtol=0, atol=1e-9):
            raise AssertionError(f"fused RMSE differs on {date}")
        daily.append({"date": date, "l3_available": row["l3_available"], **calculated})
        pooled["all"].append(values)
        pooled["l3_available" if row["l3_available"] else "l3_unavailable"].append(values)
    summary = {name: metrics(*(np.concatenate([r[i] for r in rows])
                               for i in range(6)))
               for name, rows in pooled.items()}
    for name, source_name in (("all", "all"), ("l3_available", "l3_available"),
                              ("l3_unavailable", "l3_unavailable")):
        for model_name, key in (("book", "book_rmse_ticks"),
                                ("fused", "new_hybrid_rmse_ticks")):
            expected = signal["pooled"][source_name][model_name]["rmse_ticks"]
            if not np.isclose(summary[name][key], expected, rtol=0, atol=1e-9):
                raise AssertionError(f"pooled {name}/{model_name} RMSE differs")
    result = {
        "target": signal["target"],
        "zero_change_prediction_ticks": 0.0,
        "summary": summary,
        "daily": daily,
        "signal_source": str(SIGNAL.relative_to(ROOT)),
        "price_source": str(MANIFEST.relative_to(ROOT)),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
