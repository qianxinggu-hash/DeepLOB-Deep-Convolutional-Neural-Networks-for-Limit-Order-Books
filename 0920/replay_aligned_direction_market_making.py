#!/usr/bin/env python3
"""Matched AS and GP replay of aligned 20-snapshot direction classifiers."""

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
    ASDriftParameters, ASParameters, load_or_extract_trades,
)
from gap_window_filter import enforce_safe_day_eligibility  # noqa: E402
from run_direct_drift_signal import prepare_day  # noqa: E402
from run_multiday_strategy import load_day  # noqa: E402
from run_new_hybrid_month import gp_policy_set, simulate_as, checked  # noqa: E402

MANIFEST = HERE / "output/gap_2m_books/filtered_manifest.json"
SIGNALS = HERE / "output/aligned_direction_comparison/signals"
OUT = HERE / "output/aligned_direction_market_making"
MODELS = ("deeplob", "old_hybrid", "new_l3_hybrid")


def compact(result: dict) -> dict:
    fees = result.get("all_in_fees_hkd", result.get("fees_hkd"))
    return {
        "net_hkd": float(result["net_pnl_hkd"]),
        "gross_hkd": float(result["gross_pnl_hkd"]),
        "fees_hkd": float(fees),
        "maker_fills": int(result["maker_fills"]),
        "quote_decisions": int(result.get("quote_decisions", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--signals-dir", type=Path, default=SIGNALS)
    parser.add_argument("--limit-test-days", type=int)
    parser.add_argument("--gammas", type=float, nargs="+", default=(5.0, 3.125))
    parser.add_argument("--tick-map", choices=("calibrated", "hard_one_tick"),
                        default="calibrated")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_day").mkdir(exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    dates = manifest["date_order"]
    if args.limit_test_days:
        dates = dates[:1 + args.limit_test_days]
    config = gp.GPConfig(max_inventory_lots=10)
    days, prepared, trades, stats = {}, {}, {}, {}
    for date in dates:
        path = ROOT / manifest["book_cache_by_date"][date]
        trade_path = ROOT / manifest["trade_cache_by_date"][date]
        days[date], _ = enforce_safe_day_eligibility(
            load_day(date, "tolerant_mbo_gap2m", path)
        )
        days[date].features = None
        prepared[date] = prepare_day(date, path, safe_eligibility=True)
        raw = Path.home() / f"Downloads/7709_202607/hk07709_{date}.csv"
        trades[date] = load_or_extract_trades(raw, trade_path)
        stats[date] = gp.collect_daily_calibration_stats(days[date], trades[date], config)
        print("CALIBRATED", date, flush=True)
    daily = []
    for ordinal, date in enumerate(dates[1:], 1):
        dest = out / "per_day" / f"{date}.json"
        if dest.exists():
            cached = json.loads(dest.read_text())
            if any(row.get("tick_map", "calibrated") != args.tick_map
                   for row in cached):
                raise AssertionError(f"{date}: cached replay uses another tick map")
            if any(Path(row.get("signals_dir", SIGNALS)).resolve() !=
                   args.signals_dir.resolve() for row in cached):
                raise AssertionError(f"{date}: cached replay uses another signals directory")
            gp_quote_count = next(r["quote_decisions"] for r in cached
                                  if r["family"] == "GP")
            for row in cached:
                if row["family"] == "AS":
                    row["quote_decisions"] = gp_quote_count
            dest.write_text(json.dumps(cached, indent=2, allow_nan=False) + "\n")
            daily.extend(cached)
            print("REUSE", date, flush=True)
            continue
        expected_prior = [d for d in dates[:ordinal]
                          if prepared[d].zero_recorded_errors][-5:]
        with np.load(args.signals_dir / f"day_{date}.npz", allow_pickle=False) as a:
            cache = {k: a[k].copy() for k in a.files}
        if list(cache["prior_dates"].astype(str)) != expected_prior:
            raise AssertionError(f"{date}: prior dates differ")
        indices = gp.select_quote_indices(days[date], days[date].eligible,
                                           config.quote_horizon_snapshots)
        if not np.array_equal(cache["quote_indices"], indices):
            raise AssertionError(f"{date}: quote decisions differ")
        clean = bool(cache["l3_available"])
        if clean != prepared[date].zero_recorded_errors:
            raise AssertionError(f"{date}: L3 eligibility differs")
        tick = gp.infer_tick_hkd(days[date])
        duration = cache["causal_horizon_seconds"]
        inputs = gp.aggregate_calibration_stats([stats[d] for d in expected_prior],
                                                 config)
        suffix = "_hard" if args.tick_map == "hard_one_tick" else ""
        predictions = {name: cache[f"expected_{name}{suffix}_ticks"]
                       for name in MODELS}
        if args.tick_map == "hard_one_tick":
            for values in predictions.values():
                if not np.isin(values, (-1.0, 0.0, 1.0)).all():
                    raise AssertionError("hard tick map is not -1/0/+1")
        if not clean and not np.array_equal(predictions["old_hybrid"],
                                             predictions["new_l3_hybrid"]):
            raise AssertionError("L3 fallback differs")
        regimes, intervals, states = {}, {}, {}
        for name in MODELS:
            branch = dict(cache)
            branch["prior_expected_move_ticks"] = cache[f"prior_{name}{suffix}_ticks"]
            regimes[name], intervals[name] = sweep.prior_regime(
                branch, expected_prior, days, config, 5
            )
            states[name] = gp.assign_gp_drift_states(
                predictions[name] * tick / duration, regimes[name]
            )
        prior_move = np.concatenate([
            prepared[d].actual_move_ticks[prepared[d].has_100_snapshot_history]
            for d in expected_prior
        ])
        variance = float(np.var(prior_move, ddof=1))
        as_base = ASParameters(0.02, 1.0, variance, 5)
        as_drift = ASDriftParameters(0.02, 1.0, variance, 1.0, 5)
        rows = []
        for fill_mode in gp.FILL_MODES:
            as_results = {}
            zero_as = simulate_as(days[date], trades[date], indices,
                                  np.zeros(len(indices)), as_base, fill_mode,
                                  "aligned_no_drift_as", False)
            as_results["no_drift"] = compact(zero_as)
            for name in MODELS:
                if name == "new_l3_hybrid" and not clean:
                    as_results[name] = as_results["old_hybrid"].copy()
                else:
                    as_results[name] = compact(simulate_as(
                        days[date], trades[date], indices, predictions[name],
                        as_drift, fill_mode, f"aligned_{name}_as", False,
                    ))
            for name, result in as_results.items():
                rows.append({"date": date, "family": "AS", "fill_mode": fill_mode,
                             "gamma": None, "model": name, "l3_available": clean,
                             "tick_map": args.tick_map,
                             "signals_dir": str(args.signals_dir.resolve()),
                             "prior_dates": expected_prior,
                             **{**result, "quote_decisions": len(indices)}})
            for gamma in args.gammas:
                gp_config = replace(config, inventory_penalty_gamma=gamma)
                base_policy = gp_policy_set(inputs, tick, fill_mode, gp_config, None)
                base = gp.simulate_gp_day(days[date], trades[date], base_policy,
                                          gp_config, fill_mode, "aligned_no_drift_gp")
                checked(base)
                gp_results = {"no_drift": compact(base)}
                for name in MODELS:
                    if name == "new_l3_hybrid" and not clean:
                        gp_results[name] = gp_results["old_hybrid"].copy()
                        continue
                    policy = gp_policy_set(inputs, tick, fill_mode,
                                           gp_config, regimes[name])
                    result = gp.simulate_gp_day(
                        days[date], trades[date], policy, gp_config, fill_mode,
                        f"aligned_{name}_gp", states[name]
                    )
                    checked(result)
                    gp_results[name] = compact(result)
                for name, result in gp_results.items():
                    if result["quote_decisions"] != len(indices):
                        raise AssertionError("GP quote count differs")
                    rows.append({"date": date, "family": "GP", "fill_mode": fill_mode,
                                 "gamma": gamma, "model": name, "l3_available": clean,
                                 "tick_map": args.tick_map,
                                 "signals_dir": str(args.signals_dir.resolve()),
                                 "prior_dates": expected_prior,
                                 "regime_interval_seconds": intervals.get(name),
                                 **result})
                print("REPLAY", date, fill_mode, gamma, flush=True)
        dest.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        daily.extend(rows)
    summary = []
    for family in ("AS", "GP"):
        for fill_mode in gp.FILL_MODES:
            for gamma in ((None,) if family == "AS" else tuple(args.gammas)):
                for subset in ("all", "l3_available", "l3_unavailable"):
                    selected = [r for r in daily if r["family"] == family
                                and r["fill_mode"] == fill_mode and r["gamma"] == gamma
                                and (subset == "all" or r["l3_available"] ==
                                     (subset == "l3_available"))]
                    by_name = {name: [r for r in selected if r["model"] == name]
                               for name in ("no_drift", *MODELS)}
                    for name, rows in by_name.items():
                        summary.append({
                            "family": family, "fill_mode": fill_mode,
                            "gamma": gamma, "subset": subset, "model": name,
                            "days": len(rows),
                            **{field: sum(r[field] for r in rows)
                               for field in ("net_hkd", "gross_hkd", "fees_hkd",
                                             "maker_fills", "quote_decisions")},
                        })
    result = {
        "protocol": "same 20-snapshot quote starts, prior-only QVI calibration, 5ms latency, all-in fees, through/touch proxies",
        "signal": ("argmax(down, flat, up) mapped directly to -1/0/+1 tick"
                   if args.tick_map == "hard_one_tick" else
                   "aligned 3-class probabilities, prior-only in-sample probability-to-tick calibration"),
        "tick_map": args.tick_map,
        "signals_dir": str(args.signals_dir),
        "gap_inventory": "carry",
        "gammas": args.gammas,
        "test_dates": dates[1:], "daily": daily, "summary": summary,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (out / "daily.csv").open("w", newline="") as f:
        flat = [{**row, "prior_dates": "|".join(row["prior_dates"])}
                for row in daily]
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(
            k for row in flat for k in row
        )))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps([s for s in summary if s["subset"] == "all"], indent=2),
          flush=True)


if __name__ == "__main__":
    main()
