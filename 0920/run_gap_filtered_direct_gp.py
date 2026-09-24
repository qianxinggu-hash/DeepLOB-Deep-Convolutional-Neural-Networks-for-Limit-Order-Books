#!/usr/bin/env python3
"""Replay rolling direct drift on July books cleaned around feed gaps.

The test-day reconstruction and every rolling calibration date use the same
two-minute gap-cleaned books.  Carry and forced-flat gap inventory treatments
are both reported because retrospectively known gap windows are not a live
trading signal.
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
from gap_window_filter import enforce_safe_day_eligibility  # noqa: E402
from directional_market_maker import load_or_extract_trades  # noqa: E402
from run_multiday_strategy import load_day  # noqa: E402


DEFAULT_MANIFEST = HERE / "output/gap_2m_books/filtered_manifest.json"
DEFAULT_SIGNAL = HERE / "output/direct_drift_signal_gap_2m"
DEFAULT_OUTPUT = HERE / "output/gap_2m_direct_gp"
SOURCE_AUDIT = HERE / "output/gamma_results.json"


def aggregate(rows: list[dict], gamma: float, mode: str,
              inventory_treatment: str) -> dict:
    chosen = [r for r in rows if r["gamma"] == gamma
              and r["fill_mode"] == mode
              and r["gap_inventory"] == inventory_treatment]
    return {
        "gamma": gamma,
        "fill_mode": mode,
        "gap_inventory": inventory_treatment,
        "test_days": len(chosen),
        "quote_decisions": sum(r["quote_decisions"] for r in chosen),
        "baseline_net_hkd": sum(r["baseline_net_hkd"] for r in chosen),
        "drift_net_hkd": sum(r["drift_net_hkd"] for r in chosen),
        "delta_hkd": sum(r["delta_hkd"] for r in chosen),
        "no_trade_net_hkd": 0.0,
        "baseline_maker_fills": sum(r["baseline_maker_fills"] for r in chosen),
        "drift_maker_fills": sum(r["drift_maker_fills"] for r in chosen),
        "baseline_segment_flatten_lots": sum(r["baseline_segment_flatten_lots"] for r in chosen),
        "drift_segment_flatten_lots": sum(r["drift_segment_flatten_lots"] for r in chosen),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--signal-cache", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--training-quality", choices=("source-audit", "zero-recorded-errors"),
        default="source-audit",
        help="Select the same completed-date training pool as the direct signal",
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=[5.0, 3.125])
    parser.add_argument("--limit-test-days", type=int, default=None)
    args = parser.parse_args()
    if not args.gammas or any(not np.isfinite(g) or g < 0 for g in args.gammas):
        parser.error("gammas must be nonnegative finite numbers")
    if (args.training_quality != "source-audit"
            and args.output_dir.resolve() == DEFAULT_OUTPUT.resolve()):
        parser.error("quality-filtered calibration requires a separate --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = json.loads(SOURCE_AUDIT.read_text())
    dates = list(source["provenance"]["clean_trading_dates"])
    if args.limit_test_days is not None:
        if args.limit_test_days < 1:
            parser.error("limit-test-days must be positive")
        dates = dates[:1 + args.limit_test_days]
    manifest = json.loads(args.book_manifest.read_text())
    mapping = manifest["book_cache_by_date"]
    if not set(dates) <= set(mapping):
        raise ValueError("manifest is missing a source-audit date")
    source_dir = Path.home() / "Downloads/7709_202607"
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    days = {}
    trades = {}
    for date in dates:
        path = Path(mapping[date]).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        days[date], _ = enforce_safe_day_eligibility(
            load_day(date, "gap_2m_tolerant_mbo", path)
        )
        days[date].features = None
        trade_path = ROOT / f"0824/data/july_2026/hk07709_{date}_trades.npz"
        trades[date] = load_or_extract_trades(
            source_dir / f"hk07709_{date}.csv", trade_path
        )
        if days[date].metadata.get("gap_events"):
            if days[date].metadata.get("recovery_ms") != 120_000:
                raise AssertionError(f"{date}: gap day did not use 2m reconstruction")

    base_config = gp.GPConfig(max_inventory_lots=10)
    stats = {
        date: gp.collect_daily_calibration_stats(days[date], trades[date], base_config)
        for date in dates
    }
    rows = []
    for ordinal, date in enumerate(dates[1:], start=1):
        available = dates[:ordinal]
        if args.training_quality == "zero-recorded-errors":
            available = [d for d in available if not days[d].metadata.get("gap_events")
                         and not any(int(v) for v in
                                     (days[d].metadata.get("order_errors") or {}).values())]
        prior_dates = available[-5:]
        if not prior_dates or any(d >= date for d in prior_dates):
            raise AssertionError(f"{date}: no strictly earlier training date")
        inputs = gp.aggregate_calibration_stats([stats[d] for d in prior_dates], base_config)
        with np.load(args.signal_cache / f"day_{date}.npz", allow_pickle=False) as cache:
            cache_prior = [str(x) for x in cache["prior_dates"]]
            if cache_prior != prior_dates:
                raise AssertionError(f"{date}: signal and GP calibration dates differ")
            indices = gp.select_quote_indices(
                days[date], days[date].eligible, base_config.quote_horizon_snapshots
            )
            if not np.array_equal(cache["quote_indices"], indices):
                raise AssertionError(f"{date}: signal decisions differ from GP replay")
            predictions = cache["expected_move_ticks"].copy()
            durations = cache["causal_horizon_seconds"].copy()
            regime, source_interval = sweep.prior_regime(
                cache, prior_dates, days, base_config, 5
            )
        tick = gp.infer_tick_hkd(days[date])
        state = gp.assign_gp_drift_states(predictions * tick / durations, regime)
        for gamma in sorted(set(args.gammas), reverse=True):
            config = replace(base_config, inventory_penalty_gamma=gamma)
            for mode in gp.FILL_MODES:
                baseline_policies = {
                    bucket: gp.solve_gp_policy(inputs, tick, mode, bucket, config)
                    for bucket in range(len(gp.SESSION_BUCKETS))
                }
                drift_policies = {
                    bucket: gp.solve_gp_drift_policy(
                        inputs, tick, mode, bucket, config, regime
                    ) for bucket in range(len(gp.SESSION_BUCKETS))
                }
                for treatment, flatten in (("carry", False), ("forced_flat", True)):
                    baseline = gp.simulate_gp_day(
                        days[date], trades[date], baseline_policies,
                        config, mode, "martingale_gp",
                        flatten_on_segment_change=flatten,
                    )
                    drift = gp.simulate_gp_day(
                        days[date], trades[date], drift_policies,
                        config, mode, "direct_drift_gp", state,
                        flatten_on_segment_change=flatten,
                    )
                    for result in (baseline, drift):
                        if not np.isclose(
                            result["net_pnl_hkd"],
                            result["gross_pnl_hkd"] - result["all_in_fees_hkd"],
                            rtol=0, atol=1e-6,
                        ) or result["end_inventory_lots"] != 0:
                            raise AssertionError("GP replay cash or inventory mismatch")
                    if baseline["quote_decisions"] != len(indices):
                        raise AssertionError("GP replay quote decisions differ")
                    row = {
                        "date": date, "gamma": gamma, "fill_mode": mode,
                        "gap_inventory": treatment,
                        "prior_dates": "|".join(prior_dates),
                        "quote_decisions": len(indices),
                        "baseline_net_hkd": baseline["net_pnl_hkd"],
                        "drift_net_hkd": drift["net_pnl_hkd"],
                        "delta_hkd": drift["net_pnl_hkd"] - baseline["net_pnl_hkd"],
                        "baseline_maker_fills": baseline["maker_fills"],
                        "drift_maker_fills": drift["maker_fills"],
                        "baseline_segment_flatten_lots": baseline["segment_flatten_lots"],
                        "drift_segment_flatten_lots": drift["segment_flatten_lots"],
                        "drift_regime_source_interval_seconds": source_interval,
                    }
                    rows.append(row)
            print("REPLAY", date, gamma, flush=True)
    summary = [
        aggregate(rows, gamma, mode, treatment)
        for gamma in sorted(set(args.gammas), reverse=True)
        for mode in gp.FILL_MODES
        for treatment in ("carry", "forced_flat")
    ]
    payload = {
        "experiment": "Rolling direct terminal-mid drift on gap +/-2m cleaned MBO",
        "book_manifest": str(args.book_manifest),
        "signal_cache": str(args.signal_cache),
        "training_quality": args.training_quality,
        "dates": dates,
        "prior_days": 5,
        "gamma_values": sorted(set(args.gammas), reverse=True),
        "summary": summary,
        "daily": rows,
        "limitations": [
            "Gap windows were identified after observing later feed messages; excluding them from historical test-day quotes is a diagnostic, not a deployable real-time gap detector.",
            "Forced-flat gap inventory uses the last reliable quote before the excluded interval; carry and forced-flat results bound different unobservable outage exposures.",
            "A two-minute post-gap reconstruction has not been proven complete without an authoritative order-book snapshot.",
        ],
        "validation": {
            "prior_only": all(all(p < r["date"] for p in r["prior_dates"].split("|")) for r in rows),
            "baseline_and_drift_same_cleaned_books": True,
            "cash_and_end_inventory_reconciled": True,
        },
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    with (args.output_dir / "daily.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
