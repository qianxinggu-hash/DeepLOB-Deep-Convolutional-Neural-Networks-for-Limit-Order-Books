#!/usr/bin/env python3
"""July AS and corrected GP/QVI replay with frozen-DeepLOB L2/L3 fusion.

All models share the same two-minute-gap-filtered book, prior-only calibration,
20-snapshot quote lifetime, 5ms latency, trade fill proxies and fee schedule.
The L3 branch is available only on dates with complete, error-free MBO state;
other dates use the book-only forecast and policy exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0906"))
sys.path.insert(0, str(ROOT / "0824"))

import gp_qvi_model as gp  # noqa: E402
import sweep_direct_drift_gamma as sweep  # noqa: E402
from directional_market_maker import (  # noqa: E402
    ASDriftParameters, ASParameters, load_or_extract_trades, simulate_market_maker,
)
from gap_window_filter import enforce_safe_day_eligibility  # noqa: E402
from run_direct_drift_signal import prepare_day  # noqa: E402
from run_multiday_strategy import load_day  # noqa: E402


MANIFEST = HERE / "output/gap_2m_books/filtered_manifest.json"
DEFAULT_SIGNAL = HERE / "output/new_hybrid_signal"
DEFAULT_OUTPUT = HERE / "output/new_hybrid_month"
AS_GAMMA = 0.02
AS_DECAY = 1.0
AS_CAP = 5
GP_CAP = 10


def summarize(rows: list[dict], family: str, mode: str, gamma: float | None,
              subset: str) -> dict:
    selected = [r for r in rows if r["family"] == family and r["fill_mode"] == mode
                and r["gamma"] == gamma
                and (subset == "all" or
                     (subset == "l3_available" and r["l3_available"]) or
                     (subset == "l3_unavailable" and not r["l3_available"]))]
    return {
        "family": family, "fill_mode": mode, "gamma": gamma, "subset": subset,
        "days": len(selected),
        "quote_decisions": sum(r["quote_decisions"] for r in selected),
        "baseline_net_hkd": sum(r["baseline_net_hkd"] for r in selected),
        "book_drift_net_hkd": sum(r["book_drift_net_hkd"] for r in selected),
        "new_hybrid_net_hkd": sum(r["new_hybrid_net_hkd"] for r in selected),
        "new_minus_baseline_hkd": sum(r["new_hybrid_net_hkd"] - r["baseline_net_hkd"]
                                      for r in selected),
        "new_minus_book_hkd": sum(r["new_hybrid_net_hkd"] - r["book_drift_net_hkd"]
                                  for r in selected),
        "baseline_maker_fills": sum(r["baseline_maker_fills"] for r in selected),
        "new_hybrid_maker_fills": sum(r["new_hybrid_maker_fills"] for r in selected),
        "days_new_better_than_book": sum(r["new_hybrid_net_hkd"] > r["book_drift_net_hkd"] + 1e-8
                                         for r in selected),
        "days_new_worse_than_book": sum(r["new_hybrid_net_hkd"] < r["book_drift_net_hkd"] - 1e-8
                                        for r in selected),
    }


def gp_policy_set(inputs: object, tick: float, mode: str, config: gp.GPConfig,
                  regime: object | None) -> dict:
    return {
        bucket: (
            gp.solve_gp_policy(inputs, tick, mode, bucket, config)
            if regime is None else
            gp.solve_gp_drift_policy(inputs, tick, mode, bucket, config, regime)
        ) for bucket in range(len(gp.SESSION_BUCKETS))
    }


def checked(result: dict) -> None:
    fees = result.get("all_in_fees_hkd", result.get("fees_hkd"))
    if (not np.isclose(result["net_pnl_hkd"],
                       result["gross_pnl_hkd"] - fees,
                       rtol=0, atol=1e-6)
            or result.get("end_inventory_lots", 0) != 0):
        raise AssertionError("cash, fees or end inventory fail to reconcile")


def simulate_as(
    day: object, trades: object, indices: np.ndarray, forecast: np.ndarray,
    parameters: object, mode: str, name: str, forced_flat: bool,
) -> dict:
    """In sensitivity mode, flatten at the last retained quote before a feed gap."""
    gap_segments = {int(event["new_segment"])
                    for event in (day.metadata.get("gap_events") or [])}
    cuts = [i for i in range(1, len(indices))
            if forced_flat and int(day.segments[int(indices[i])]) in gap_segments
            and day.segments[int(indices[i])] != day.segments[int(indices[i - 1])]]
    starts, ends = [0, *cuts], [*cuts, len(indices)]
    parts = []
    for start, end in zip(starts, ends):
        if start == end:
            continue
        result, _ = simulate_market_maker(
            day, trades, indices[start:end], forecast[start:end],
            parameters, mode, name, tick_hkd=gp.infer_tick_hkd(day),
        )
        checked(result)
        parts.append(result)
    return {
        "gross_pnl_hkd": sum(p["gross_pnl_hkd"] for p in parts),
        "fees_hkd": sum(p["fees_hkd"] for p in parts),
        "net_pnl_hkd": sum(p["net_pnl_hkd"] for p in parts),
        "maker_fills": sum(p["maker_fills"] for p in parts),
        "gap_flatten_count": max(len(parts) - 1, 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gp-gammas", type=float, nargs="+", default=(5.0, 3.125))
    parser.add_argument("--limit-test-days", type=int, default=None)
    parser.add_argument("--forced-flat", action="store_true",
                        help="diagnostic forced inventory flatten before recorded feed gaps")
    parser.add_argument("--as-only", action="store_true",
                        help="replay only AS (useful for a corrected daily tick audit)")
    args = parser.parse_args()
    if not args.gp_gammas or any(g <= 0 or not np.isfinite(g) for g in args.gp_gammas):
        parser.error("GP gammas must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    dates = manifest["date_order"]
    if args.limit_test_days is not None:
        dates = dates[:1 + args.limit_test_days]
    base_gp = gp.GPConfig(max_inventory_lots=GP_CAP)
    days, prepared, trades, stats = {}, {}, {}, {}
    for date in dates:
        book_path = ROOT / manifest["book_cache_by_date"][date]
        trade_path = ROOT / manifest["trade_cache_by_date"][date]
        days[date], _ = enforce_safe_day_eligibility(
            load_day(date, "tolerant_mbo_gap2m", book_path)
        )
        days[date].features = None
        prepared[date] = prepare_day(date, book_path, safe_eligibility=True)
        source = Path.home() / f"Downloads/7709_202607/hk07709_{date}.csv"
        trades[date] = load_or_extract_trades(source, trade_path)
        stats[date] = gp.collect_daily_calibration_stats(days[date], trades[date], base_gp)
        print("CALIBRATION", date, flush=True)

    rows = []
    for ordinal, date in enumerate(dates[1:], start=1):
        signal_cache = args.signal_dir / "predictions" / f"day_{date}.npz"
        with np.load(signal_cache, allow_pickle=False) as cache:
            prior_dates = [str(x) for x in cache["prior_dates"]]
            expected_prior = [d for d in dates[:ordinal] if prepared[d].zero_recorded_errors][-5:]
            if prior_dates != expected_prior:
                raise AssertionError(f"prior signal dates differ on {date}")
            indices = gp.select_quote_indices(days[date], days[date].eligible,
                                               base_gp.quote_horizon_snapshots)
            if not np.array_equal(cache["quote_indices"], indices):
                raise AssertionError(f"signal/GP quote starts differ on {date}")
            l3_available = bool(cache["l3_available"])
            if l3_available != prepared[date].zero_recorded_errors:
                raise AssertionError(f"L3 availability audit differs on {date}")
            book_prediction = cache["book_only_expected_move_ticks"].copy()
            fused_prediction = cache["expected_move_ticks"].copy()
            durations = cache["causal_horizon_seconds"].copy()
            prior_book_cache = {key: cache[key] for key in cache.files}
            prior_book_cache["prior_expected_move_ticks"] = cache[
                "prior_book_only_expected_move_ticks"
            ].copy()
            book_regime, book_interval = sweep.prior_regime(
                prior_book_cache, prior_dates, days, base_gp, 5
            )
            if l3_available:
                fused_regime, fused_interval = sweep.prior_regime(
                    cache, prior_dates, days, base_gp, 5
                )
            else:
                fused_regime, fused_interval = book_regime, book_interval
                if not np.array_equal(book_prediction, fused_prediction):
                    raise AssertionError("gap-day fused signal did not fall back exactly")
        inputs = gp.aggregate_calibration_stats([stats[d] for d in prior_dates], base_gp)
        tick = gp.infer_tick_hkd(days[date])
        prior_y = np.concatenate([
            prepared[d].actual_move_ticks[prepared[d].has_100_snapshot_history]
            for d in prior_dates
        ])
        variance = float(np.var(prior_y, ddof=1))
        as_base = ASParameters(AS_GAMMA, AS_DECAY, variance, AS_CAP)
        as_drift = ASDriftParameters(AS_GAMMA, AS_DECAY, variance, 1.0, AS_CAP)
        for mode in gp.FILL_MODES:
            as_results = {}
            for name, prediction_values, params in (
                ("baseline", np.zeros(len(indices)), as_base),
                ("book", book_prediction, as_drift),
                ("new", fused_prediction, as_drift),
            ):
                if name == "new" and not l3_available:
                    as_results[name] = as_results["book"]
                else:
                    as_results[name] = simulate_as(
                        days[date], trades[date], indices, prediction_values,
                        params, mode, f"new_hybrid_{name}_as", args.forced_flat,
                    )
                checked(as_results[name])
            rows.append({
                "date": date, "family": "AS", "fill_mode": mode, "gamma": None,
                "l3_available": l3_available, "prior_dates": "|".join(prior_dates),
                "quote_decisions": len(indices),
                "baseline_net_hkd": as_results["baseline"]["net_pnl_hkd"],
                "book_drift_net_hkd": as_results["book"]["net_pnl_hkd"],
                "new_hybrid_net_hkd": as_results["new"]["net_pnl_hkd"],
                "baseline_maker_fills": as_results["baseline"]["maker_fills"],
                "new_hybrid_maker_fills": as_results["new"]["maker_fills"],
                "baseline_all_in_fees_hkd": as_results["baseline"]["fees_hkd"],
                "new_hybrid_all_in_fees_hkd": as_results["new"]["fees_hkd"],
                "book_regime_interval_seconds": None,
                "new_regime_interval_seconds": None,
            })
        if args.as_only:
            print("AS_REPLAY", date, flush=True)
            continue
        book_states = gp.assign_gp_drift_states(
            book_prediction * tick / durations, book_regime
        )
        fused_states = gp.assign_gp_drift_states(
            fused_prediction * tick / durations, fused_regime
        )
        for gamma in sorted(set(args.gp_gammas), reverse=True):
            config = replace(base_gp, inventory_penalty_gamma=gamma)
            for mode in gp.FILL_MODES:
                baseline_policy = gp_policy_set(inputs, tick, mode, config, None)
                book_policy = gp_policy_set(inputs, tick, mode, config, book_regime)
                new_policy = (gp_policy_set(inputs, tick, mode, config, fused_regime)
                              if l3_available else book_policy)
                baseline = gp.simulate_gp_day(
                    days[date], trades[date], baseline_policy, config, mode,
                    "martingale_gp", flatten_on_segment_change=args.forced_flat,
                )
                book_result = gp.simulate_gp_day(
                    days[date], trades[date], book_policy, config, mode,
                    "new_hybrid_book_gp", book_states,
                    flatten_on_segment_change=args.forced_flat,
                )
                new_result = (gp.simulate_gp_day(
                    days[date], trades[date], new_policy, config, mode,
                    "new_hybrid_gp", fused_states,
                    flatten_on_segment_change=args.forced_flat,
                ) if l3_available else book_result)
                for value in (baseline, book_result, new_result):
                    checked(value)
                    if value["quote_decisions"] != len(indices):
                        raise AssertionError(f"GP decisions differ on {date}")
                rows.append({
                    "date": date, "family": "GP", "fill_mode": mode,
                    "gamma": gamma, "l3_available": l3_available,
                    "prior_dates": "|".join(prior_dates),
                    "quote_decisions": len(indices),
                    "baseline_net_hkd": baseline["net_pnl_hkd"],
                    "book_drift_net_hkd": book_result["net_pnl_hkd"],
                    "new_hybrid_net_hkd": new_result["net_pnl_hkd"],
                    "baseline_maker_fills": baseline["maker_fills"],
                    "new_hybrid_maker_fills": new_result["maker_fills"],
                    "baseline_all_in_fees_hkd": baseline["all_in_fees_hkd"],
                    "new_hybrid_all_in_fees_hkd": new_result["all_in_fees_hkd"],
                    "book_regime_interval_seconds": book_interval,
                    "new_regime_interval_seconds": fused_interval,
                })
            print("REPLAY", date, "gamma", gamma, flush=True)
        with (args.output_dir / "daily.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = [
        summarize(rows, family, mode, gamma, subset)
        for family, gamma in (("AS", None), *(("GP", g) for g in sorted(set(args.gp_gammas), reverse=True)))
        if not args.as_only or family == "AS"
        for mode in gp.FILL_MODES
        for subset in ("all", "l3_available", "l3_unavailable")
    ]
    if args.as_only:
        with (args.output_dir / "daily.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "name": "new hybrid",
        "signal_result": str((args.signal_dir / "results.json").resolve()),
        "dates": dates,
        "test_dates": dates[1:],
        "gap_policy": "whole L3 state unavailable on recorded-gap dates; use exact book-only fallback on those days",
        "gap_inventory": "forced_flat" if args.forced_flat else "carry",
        "fee_and_execution": "existing AS and corrected GP simulators; 5ms latency, 20-snapshot expiry, all-in fees, through/touch proxies",
        "as_parameters": {"risk_aversion_per_tick": AS_GAMMA,
                          "order_decay_per_tick": AS_DECAY,
                          "max_inventory_lots": AS_CAP,
                          "variance": "prior clean-day 20-snapshot endpoint variance"},
        "gp_gammas": sorted(set(args.gp_gammas), reverse=True),
        "as_only": args.as_only,
        "as_tick_corrected": True,
        "as_tick_hkd": "per-day exchange tick from gp.infer_tick_hkd",
        "summary": summary,
        "daily": rows,
        "cash_and_inventory_reconciled": True,
        "test_labels_not_used_in_current_day_fit": True,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps([s for s in summary if s["subset"] == "all"],
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
