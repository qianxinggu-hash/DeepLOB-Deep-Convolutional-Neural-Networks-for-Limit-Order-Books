#!/usr/bin/env python3
"""Reconcile monthly new-hybrid replay and day-block prediction uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFERENCE = HERE / "output/gap_2m_direct_gp_quality/results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month-dir", type=Path,
                        default=HERE / "output/new_hybrid_month")
    args = parser.parse_args()
    month = json.loads((args.month_dir / "results.json").read_text())
    signal = json.loads((HERE / "output/new_hybrid_signal/results.json").read_text())
    reference = json.loads(REFERENCE.read_text())
    treatment = month.get("gap_inventory", "carry")
    corrected_path = HERE / f"output/new_hybrid_as_corrected_{treatment}/results.json"
    corrected_as_rows = None
    if corrected_path.exists():
        corrected_as = json.loads(corrected_path.read_text())
        corrected_as_rows = {(r["date"], r["fill_mode"]): r
                             for r in corrected_as["daily"]}
    reference_rows = {
        (r["date"], r["gamma"], r["fill_mode"]): r
        for r in reference["daily"] if r["gap_inventory"] == treatment
    }
    gp_checks, fallback_checks, day_checks, as_checks = [], [], [], []
    for row in month["daily"]:
        day_checks.append(row["quote_decisions"] > 0)
        if not row["l3_available"]:
            fallback_checks.append(np.isclose(
                row["book_drift_net_hkd"], row["new_hybrid_net_hkd"],
                rtol=0, atol=1e-7,
            ))
        if row["family"] == "GP":
            old = reference_rows[(row["date"], row["gamma"], row["fill_mode"])]
            gp_checks.append(np.isclose(
                row["baseline_net_hkd"], old["baseline_net_hkd"],
                rtol=0, atol=1e-7,
            ))
        else:
            if corrected_as_rows is not None:
                corrected = corrected_as_rows[(row["date"], row["fill_mode"])]
                as_checks.append(all(np.isclose(
                    row[key], corrected[key], rtol=0, atol=1e-7,
                ) for key in ("baseline_net_hkd", "book_drift_net_hkd",
                              "new_hybrid_net_hkd", "baseline_all_in_fees_hkd",
                              "new_hybrid_all_in_fees_hkd")))

    clean_dates = [row["date"] for row in signal["daily"] if row["l3_available"]]
    reference_dates = [row["date"] for row in signal["daily"] if not row["l3_available"]]
    if len(clean_dates) != 10 or len(reference_dates) != 9:
        raise AssertionError("expected 10 L3 test dates and 9 fallback dates")
    book_sse, fused_sse = [], []
    for date in clean_dates:
        with np.load(HERE / "output/new_hybrid_signal/predictions" /
                     f"day_{date}.npz", allow_pickle=False) as cache:
            valid = cache["has_100_snapshot_history"]
            y = cache["actual_move_ticks_diagnostic"][valid]
            book = cache["book_only_expected_move_ticks"][valid]
            fused = cache["expected_move_ticks"][valid]
            book_sse.append(float(np.square(y - book).sum()))
            fused_sse.append(float(np.square(y - fused).sum()))
    book_sse = np.asarray(book_sse)
    fused_sse = np.asarray(fused_sse)
    rng = np.random.default_rng(20260923)
    draws = rng.integers(0, len(clean_dates), size=(20_000, len(clean_dates)))
    lifts = 100 * (book_sse[draws].sum(axis=1) - fused_sse[draws].sum(axis=1)) / book_sse[draws].sum(axis=1)
    result = {
        "month_dir": str(args.month_dir.resolve()),
        "gap_inventory": treatment,
        "gp_baseline_matches_prior_quality_replay": bool(all(gp_checks)),
        "gp_baseline_checks": len(gp_checks),
        "as_daily_tick_corrected_and_reconciled": bool(all(as_checks)) and bool(month.get("as_tick_corrected")),
        "as_daily_tick_checks": len(as_checks),
        "gap_day_book_fallback_exact": bool(all(fallback_checks)),
        "gap_day_fallback_checks": len(fallback_checks),
        "all_test_days_have_quotes": bool(all(day_checks)),
        "l3_available_test_dates": clean_dates,
        "l3_unavailable_test_dates": reference_dates,
        "clean_day_mse_reduction_pct": float(100 * (book_sse.sum() - fused_sse.sum()) / book_sse.sum()),
        "clean_day_mse_reduction_day_bootstrap_ci95_pct": np.quantile(lifts, (0.025, 0.975)).tolist(),
        "bootstrap_fraction_positive": float(np.mean(lifts > 0)),
        "limitations": [
            "Day bootstrap covers only these ten previously inspected clean July dates; it is not an untouched-date validation.",
            "Forced-flat uses historical knowledge of the feed-gap boundary and is a diagnostic, not a real-time gap detector.",
        ],
    }
    if not (result["gp_baseline_matches_prior_quality_replay"]
            and result["as_daily_tick_corrected_and_reconciled"]
            and result["gap_day_book_fallback_exact"]
            and result["all_test_days_have_quotes"]):
        raise AssertionError(result)
    (args.month_dir / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
