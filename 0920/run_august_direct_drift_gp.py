#!/usr/bin/env python3
"""Check a frozen July direct-drift model in the August 7 MBO replay.

The Ridge signal and GP inputs use the same five July dates with no recorded
reconstruction errors. August labels are used only for diagnostics. August 7
was examined in earlier repo experiments, so this is a later-date stress
check rather than a pristine untouched holdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import run_hybrid_month_experiment as month
import sweep_direct_drift_gamma as sweep
import gp_qvi_model as gp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0824"))
from directional_market_maker import load_or_extract_trades  # noqa: E402
from run_multiday_strategy import load_day  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=("2026-08-04", "2026-08-07"),
                        default="2026-08-07")
    parser.add_argument("--gammas", nargs="+", type=float,
                        default=(5.0, 3.125, 1.5625))
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "output/august_holdout")
    args = parser.parse_args()
    if not args.gammas or any(not math.isfinite(g) or g < 0 for g in args.gammas):
        parser.error("gammas must be finite and nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, _, july_days, july_trades = month._load_inputs()
    cache_path = HERE / f"output/direct_drift_signal/august_day_{args.date}.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        prior = [str(d) for d in cache["prior_dates"]]
        if prior != ["2026-07-23", "2026-07-27", "2026-07-28",
                     "2026-07-29", "2026-07-30"]:
            raise AssertionError("unexpected frozen model and GP prior dates")
        book_path = args.output_dir / f"hk07709_{args.date}_mbo_tolerant_5m_s10.npz"
        day = load_day(args.date, "tolerant_mbo", book_path)
        trades = load_or_extract_trades(
            ROOT / f"7709_tickdata/hk07709_{args.date}.csv",
            args.output_dir / f"hk07709_{args.date}_trades.npz",
        )
        config = gp.GPConfig(max_inventory_lots=10, inventory_penalty_gamma=5.0)
        indices = cache["quote_indices"]
        if not np.array_equal(indices, gp.select_quote_indices(
            day, day.eligible, config.quote_horizon_snapshots
        )):
            raise AssertionError("August signal and GP quote decisions differ")
        prediction = cache["expected_move_ticks"].copy()
        seconds = cache["causal_horizon_seconds"].copy()
        if not np.allclose(cache["predicted_drift_hkd_per_second"],
                           prediction * gp.infer_tick_hkd(day) / seconds,
                           rtol=0, atol=1e-12):
            raise AssertionError("drift unit conversion does not reconcile")
        signal = month.DaySignal(
            date=args.date,
            indices=indices.copy(),
            probabilities=np.zeros((len(indices), 3), dtype=np.float64),
            true_move_ticks=cache["actual_move_ticks_diagnostic"].copy(),
            causal_durations_seconds=seconds,
            realized_durations_seconds=cache["realized_horizon_seconds_diagnostic"].copy(),
            has_model_history=cache["has_100_snapshot_history"].copy(),
            segment_ids=cache["segment_ids"].copy(),
            quote_intervals_seconds=sweep.quote_intervals(day, indices),
        )
        regime, source_interval = sweep.prior_regime(
            cache, prior, july_days, config, 5
        )
    inputs = gp.aggregate_calibration_stats([
        gp.collect_daily_calibration_stats(july_days[d], july_trades[d], config)
        for d in prior
    ], config)
    results = {}
    for gamma in sorted(set(args.gammas), reverse=True):
        print("REPLAY", args.date, gamma, flush=True)
        one = month.run_one_day(
            day, trades, inputs, replace(config, inventory_penalty_gamma=gamma),
            signal, prediction, regime,
        )
        results[f"gamma_{gamma:g}"] = {
            "gamma": gamma,
            "through": one["results"]["through"],
            "touch": one["results"]["touch"],
            "action_audit": one["counterfactual_action_audit_at_hybrid_visited_states"],
        }
        for mode in gp.FILL_MODES:
            for result in (one["results"][mode]["martingale_gp"],
                           one["results"][mode]["hybrid_drift_gp"]):
                if not math.isclose(result["net_pnl_hkd"],
                                    result["gross_pnl_hkd"]-result["all_in_fees_hkd"],
                                    rel_tol=0, abs_tol=1e-6):
                    raise AssertionError("cashflow does not reconcile")
                if result["quote_decisions"] != len(indices):
                    raise AssertionError("quote count does not reconcile")
    payload = {
        "experiment": "Frozen July direct endpoint drift on later August MBO day",
        "date": args.date,
        "training_and_gp_calibration_dates": prior,
        "test_reconstruction_metadata": {
            "order_errors": day.metadata.get("order_errors", {}),
            "feed_gap_count": len(day.metadata.get("gap_events", [])),
            "final_state_tainted": bool(day.metadata.get("final_state_tainted", False)),
        },
        "signal_cache": str(cache_path),
        "quote_decisions": len(indices),
        "drift_transition_source_interval_seconds": source_interval,
        "drift_regime": month.asdict(regime),
        "results": results,
        "limitations": [
            "August 7 had been examined in earlier repository experiments; this is a later-date stress check, not a pristine untouched holdout.",
            "Touch and through are execution proxies; results do not include queue position or borrow feasibility.",
        ],
    }
    destination = args.output_dir / f"direct_drift_gp_{args.date}.json"
    month.write_json(destination, payload)
    print("SAVED", destination.resolve(), flush=True)


if __name__ == "__main__":
    main()
