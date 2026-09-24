#!/usr/bin/env python3
"""Measure L3's incremental GP PnL over the same-gamma L2-only forecast."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import run_new_hybrid_month as month


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "output/new_hybrid_gamma_sweep/l3_marginal.json"
DEFAULT_GAMMAS = (5.0, 3.125, 1.0, 0.3, 0.0001)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    args = parser.parse_args()
    gammas = sorted(set(args.gammas), reverse=True)
    sweep_result = json.loads((HERE / "output/new_hybrid_gamma_sweep/results.json").read_text())
    sweep_rows = {(r["date"], r["gamma"], r["fill_mode"], r["gap_inventory"]): r
                  for r in sweep_result["daily"]}
    anchors = {
        "carry": json.loads((HERE / "output/new_hybrid_month/results.json").read_text()),
        "forced_flat": json.loads((HERE / "output/new_hybrid_month_forced_flat/results.json").read_text()),
    }
    anchor_rows = {(r["date"], r["gamma"], r["fill_mode"], treatment): r
                   for treatment, source in anchors.items() for r in source["daily"]
                   if r["family"] == "GP"}
    manifest = json.loads(month.MANIFEST.read_text())
    dates = manifest["date_order"]
    base = month.gp.GPConfig(max_inventory_lots=month.GP_CAP)
    days, trades, stats = {}, {}, {}
    for date in dates:
        day, _ = month.enforce_safe_day_eligibility(month.load_day(
            date, "tolerant_mbo_gap2m", month.ROOT / manifest["book_cache_by_date"][date]
        ))
        day.features = None
        days[date] = day
        source = Path.home() / f"Downloads/7709_202607/hk07709_{date}.csv"
        trades[date] = month.load_or_extract_trades(
            source, month.ROOT / manifest["trade_cache_by_date"][date]
        )
        stats[date] = month.gp.collect_daily_calibration_stats(day, trades[date], base)
    rows = []
    for date in dates[1:]:
        with np.load(HERE / "output/new_hybrid_signal/predictions" /
                     f"day_{date}.npz", allow_pickle=False) as archive:
            available = bool(archive["l3_available"])
            prior_dates = [str(x) for x in archive["prior_dates"]]
            indices = month.gp.select_quote_indices(days[date], days[date].eligible,
                                                    base.quote_horizon_snapshots)
            if not np.array_equal(indices, archive["quote_indices"]):
                raise AssertionError(f"quote starts differ on {date}")
            if available:
                prior = {key: archive[key] for key in archive.files}
                prior["prior_expected_move_ticks"] = archive[
                    "prior_book_only_expected_move_ticks"
                ].copy()
                regime, _ = month.sweep.prior_regime(prior, prior_dates, days, base, 5)
                prediction = archive["book_only_expected_move_ticks"].copy()
                durations = archive["causal_horizon_seconds"].copy()
        if available:
            inputs = month.gp.aggregate_calibration_stats([stats[d] for d in prior_dates], base)
            tick = month.gp.infer_tick_hkd(days[date])
            states = month.gp.assign_gp_drift_states(prediction * tick / durations, regime)
        for gamma in gammas:
            config = replace(base, inventory_penalty_gamma=gamma)
            for mode in month.gp.FILL_MODES:
                book_policy = (month.gp_policy_set(inputs, tick, mode, config, regime)
                               if available and gamma not in (5.0, 3.125) else None)
                for treatment in ("carry", "forced_flat"):
                    new = sweep_rows[(date, gamma, mode, treatment)]
                    if gamma in (5.0, 3.125):
                        book_net = anchor_rows[(date, gamma, mode, treatment)]["book_drift_net_hkd"]
                    elif not available:
                        book_net = new["new_hybrid_net_pnl_hkd"]
                    else:
                        book = month.gp.simulate_gp_day(
                            days[date], trades[date], book_policy, config, mode,
                            "new_hybrid_book_gp", states,
                            flatten_on_segment_change=treatment == "forced_flat",
                        )
                        month.checked(book)
                        if book["quote_decisions"] != len(indices):
                            raise AssertionError(f"quote count differs on {date}")
                        book_net = book["net_pnl_hkd"]
                    delta = new["new_hybrid_net_pnl_hkd"] - book_net
                    if not available and abs(delta) > 1e-8:
                        raise AssertionError(f"gap-day L2 fallback differs on {date}")
                    rows.append({
                        "date": date, "gamma": gamma, "fill_mode": mode,
                        "gap_inventory": treatment, "l3_available": available,
                        "book_only_net_hkd": book_net,
                        "new_hybrid_net_hkd": new["new_hybrid_net_pnl_hkd"],
                        "l3_minus_book_hkd": delta,
                    })
            print("L3_MARGINAL", date, gamma, flush=True)
    summary = [
        {"gamma": gamma, "fill_mode": mode, "gap_inventory": treatment,
         "test_days": len(selected),
         "l3_available_days": sum(r["l3_available"] for r in selected),
         "book_only_net_hkd": sum(r["book_only_net_hkd"] for r in selected),
         "new_hybrid_net_hkd": sum(r["new_hybrid_net_hkd"] for r in selected),
         "l3_minus_book_hkd": sum(r["l3_minus_book_hkd"] for r in selected)}
        for gamma in gammas for mode in month.gp.FILL_MODES
        for treatment in ("carry", "forced_flat")
        for selected in [[r for r in rows if r["gamma"] == gamma
                          and r["fill_mode"] == mode and r["gap_inventory"] == treatment]]
    ]
    result = {"gammas": gammas, "summary": summary, "daily": rows,
              "gap_fallback_exact": True,
              "source": str((HERE / "output/new_hybrid_gamma_sweep/results.json").resolve())}
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps([s for s in summary if s["gap_inventory"] == "carry"],
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
