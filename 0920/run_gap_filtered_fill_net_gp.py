#!/usr/bin/env python3
"""Walk-forward GP/QVI replay with fill-conditioned reward on gap-cleaned books."""

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
from build_fill_net_prior import load_prior_alpha  # noqa: E402
from directional_market_maker import load_or_extract_trades  # noqa: E402
from gap_window_filter import enforce_safe_day_eligibility  # noqa: E402
from run_multiday_strategy import load_day  # noqa: E402


DEFAULT_MANIFEST = HERE / "output/gap_2m_books/filtered_manifest.json"
DEFAULT_PRIOR = HERE / "output/fill_net_prior_gap2m"
DEFAULT_OUTPUT = HERE / "output/gap_2m_fill_net_gp"


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def aggregate(rows: list[dict], gamma: float, mode: str, estimate: str,
              treatment: str) -> dict:
    selected = [row for row in rows if row["gamma"] == gamma
                and row["fill_mode"] == mode and row["estimate"] == estimate
                and row["gap_inventory"] == treatment]
    return {
        "gamma": gamma,
        "fill_mode": mode,
        "estimate": estimate,
        "gap_inventory": treatment,
        "test_days": len(selected),
        "quote_decisions": sum(row["quote_decisions"] for row in selected),
        "baseline_net_hkd": sum(row["baseline_net_hkd"] for row in selected),
        "fill_net_gp_hkd": sum(row["fill_net_gp_hkd"] for row in selected),
        "delta_hkd": sum(row["delta_hkd"] for row in selected),
        "no_trade_net_hkd": 0.0,
        "baseline_maker_fills": sum(row["baseline_maker_fills"] for row in selected),
        "fill_net_maker_fills": sum(row["fill_net_maker_fills"] for row in selected),
        "fill_net_segment_flatten_lots": sum(
            row["fill_net_segment_flatten_lots"] for row in selected
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prior-cache", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gammas", type=float, nargs="+", default=[5.0, 3.125])
    parser.add_argument("--estimates", choices=["mean", "lcb"], nargs="+",
                        default=["mean", "lcb"])
    parser.add_argument("--limit-test-days", type=int, default=None)
    args = parser.parse_args()
    if not args.gammas or any(not np.isfinite(g) or g < 0 for g in args.gammas):
        parser.error("gammas must be nonnegative finite numbers")
    if args.limit_test_days is not None and args.limit_test_days < 1:
        parser.error("limit-test-days must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.book_manifest.read_text())
    dates = list(manifest["date_order"])
    if len(dates) != 20 or not all(a < b for a, b in zip(dates, dates[1:])):
        raise ValueError("expected 20 ordered July source-audit dates")
    if args.limit_test_days is not None:
        dates = dates[:1 + args.limit_test_days]
    days, trades = {}, {}
    for date in dates:
        path = resolve_path(manifest["book_cache_by_date"][date])
        days[date], _ = enforce_safe_day_eligibility(
            load_day(date, "gap_2m_tolerant_mbo", path)
        )
        days[date].features = None
        source = ROOT / "7709_tickdata" / f"hk07709_{date}.csv"
        if not source.is_file():
            source = Path.home() / "Downloads/7709_202607" / f"hk07709_{date}.csv"
        trades[date] = load_or_extract_trades(
            source, resolve_path(manifest["trade_cache_by_date"][date])
        )
    base_config = gp.GPConfig(max_inventory_lots=10)
    stats = {
        date: gp.collect_daily_calibration_stats(days[date], trades[date], base_config)
        for date in dates
    }
    rows = []
    quality_names = set()
    for date in dates[1:]:
        with np.load(args.prior_cache / f"day_{date}.npz", allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta_json"]))
        prior_dates = list(meta["prior_dates"])
        if not prior_dates or any(p >= date for p in prior_dates):
            raise AssertionError(f"{date}: noncausal prior")
        if not set(prior_dates) <= set(dates):
            raise AssertionError(f"{date}: prior date unavailable in replay")
        quality_names.add(meta["prior_quality"])
        inputs = gp.aggregate_calibration_stats([stats[p] for p in prior_dates], base_config)
        tick = gp.infer_tick_hkd(days[date])
        quote_indices = gp.select_quote_indices(
            days[date], days[date].eligible, base_config.quote_horizon_snapshots
        )
        if not np.isclose(tick, meta["tick_hkd"], rtol=0, atol=1e-12):
            raise AssertionError(f"{date}: test tick differs from fill prior")
        for gamma in sorted(set(args.gammas), reverse=True):
            config = replace(base_config, inventory_penalty_gamma=gamma)
            for mode in gp.FILL_MODES:
                baseline_policies = {
                    bucket: gp.solve_gp_policy(inputs, tick, mode, bucket, config)
                    for bucket in range(len(gp.SESSION_BUCKETS))
                }
                policies_by_estimate = {}
                for estimate in args.estimates:
                    policies_by_estimate[estimate] = {}
                    for bucket in range(len(gp.SESSION_BUCKETS)):
                        alpha = load_prior_alpha(
                            date, mode, bucket, inputs, tick, config,
                            output_dir=args.prior_cache, estimate=estimate,
                        )
                        adapted = replace(
                            inputs, maker_fill_alpha_hkd_per_lot={mode: alpha}
                        )
                        policies_by_estimate[estimate][bucket] = gp.solve_gp_policy(
                            adapted, tick, mode, bucket, config
                        )
                for treatment, flatten in (("carry", False), ("forced_flat", True)):
                    baseline = gp.simulate_gp_day(
                        days[date], trades[date], baseline_policies, config,
                        mode, "martingale_gp", flatten_on_segment_change=flatten,
                    )
                    for estimate, policies in policies_by_estimate.items():
                        result = gp.simulate_gp_day(
                            days[date], trades[date], policies, config,
                            mode, f"fill_net_{estimate}_gp",
                            flatten_on_segment_change=flatten,
                        )
                        for candidate in (baseline, result):
                            if (
                                candidate["quote_decisions"] != len(quote_indices)
                                or candidate["end_inventory_lots"] != 0
                                or not np.isclose(
                                    candidate["net_pnl_hkd"],
                                    candidate["gross_pnl_hkd"]
                                    - candidate["all_in_fees_hkd"],
                                    rtol=0, atol=1e-6,
                                )
                            ):
                                raise AssertionError("fill-net replay did not reconcile")
                        rows.append({
                            "date": date, "gamma": gamma, "fill_mode": mode,
                            "estimate": estimate,
                            "gap_inventory": treatment,
                            "prior_quality": meta["prior_quality"],
                            "prior_dates": "|".join(prior_dates),
                            "quote_decisions": len(quote_indices),
                            "baseline_net_hkd": baseline["net_pnl_hkd"],
                            "fill_net_gp_hkd": result["net_pnl_hkd"],
                            "delta_hkd": result["net_pnl_hkd"] - baseline["net_pnl_hkd"],
                            "baseline_maker_fills": baseline["maker_fills"],
                            "fill_net_maker_fills": result["maker_fills"],
                            "baseline_segment_flatten_lots": baseline["segment_flatten_lots"],
                            "fill_net_segment_flatten_lots": result["segment_flatten_lots"],
                        })
            print("REPLAY", date, gamma, flush=True)
    if len(quality_names) != 1:
        raise AssertionError("mixed prior-quality protocols")
    summary = [
        aggregate(rows, gamma, mode, estimate, treatment)
        for gamma in sorted(set(args.gammas), reverse=True)
        for mode in gp.FILL_MODES
        for estimate in args.estimates
        for treatment in ("carry", "forced_flat")
    ]
    output = {
        "experiment": "Fill-conditioned GP/QVI reward on July gap +/-2m books",
        "book_manifest": str(args.book_manifest),
        "fill_prior": str(args.prior_cache),
        "dates": dates,
        "rolling_prior_quality": next(iter(quality_names)),
        "gamma_values": sorted(set(args.gammas), reverse=True),
        "estimates": args.estimates,
        "summary": summary,
        "daily": rows,
        "validation": {
            "prior_only": all(all(p < r["date"] for p in r["prior_dates"].split("|"))
                              for r in rows),
            "same_cleaned_books_and_gp_inputs_within_comparison": True,
            "cash_and_end_inventory_reconciled": True,
        },
        "limitations": [
            "The gap windows are known only after a later feed message; removing test-day quotes before a gap is retrospective and cannot be a live trading rule.",
            "Carry and forced-flat gap inventory are two exposure sensitivities; forced flatten at the last reliable quote before a gap requires foreknowledge and is a diagnostic only.",
            "The two-minute post-gap book may be incomplete without an authoritative order-book snapshot.",
            "Touch and through are proxy fills without actual queue position.",
        ],
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    with (args.output_dir / "daily.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
