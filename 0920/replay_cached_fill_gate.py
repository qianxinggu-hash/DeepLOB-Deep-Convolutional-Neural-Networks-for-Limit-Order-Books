#!/usr/bin/env python3
"""Replay a new GP gamma using precomputed, strictly prior fill-quality gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import run_fill_conditioned_gate as gate


gp = gate.gp
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=3.125)
    parser.add_argument("--gate-results", type=Path,
                        default=gate.DEFAULT_OUTPUT / "results.json")
    parser.add_argument("--output", type=Path,
                        default=gate.DEFAULT_OUTPUT / "gamma_3_125.json")
    args = parser.parse_args()
    previous = json.loads(args.gate_results.read_text())
    prior_days = int(previous["design"]["rolling_prior_days"])
    by_date = {record["date"]: record for record in previous["gate_by_day"]}
    _, dates, days, trades = gate._load_inputs()
    assert dates == previous["design"]["dates"]
    config = replace(gp.GPConfig(), inventory_penalty_gamma=args.gamma)
    original = json.loads((HERE / "output/gamma_results.json").read_text())
    key = f"gamma_{args.gamma:g}"
    references = {test["test_date"]: test["results"]
                  for test in original["variants"][key]["tests"]}
    stats = {dates[0]: gp.collect_daily_calibration_stats(
        days[dates[0]], trades[dates[0]], config,
    )}
    rows = []
    for i, date in enumerate(dates[1:], start=1):
        prior = dates[max(0, i - prior_days):i]
        record = by_date[date]
        if record["prior_dates"] != prior:
            raise AssertionError(f"cached gate calibration mismatch on {date}")
        allowed = np.asarray(record["allowed"], dtype=bool)
        if allowed.shape != gate.SHAPE:
            raise AssertionError("cached gate shape mismatch")
        inputs = gp.aggregate_calibration_stats([stats[d] for d in prior], config)
        tick = gp.infer_tick_hkd(days[date])
        for mode in gp.FILL_MODES:
            base = {bucket: gp.solve_gp_policy(inputs, tick, mode, bucket, config)
                    for bucket in range(len(gp.SESSION_BUCKETS))}
            gated = {
                bucket: gp.solve_gp_policy(
                    gate.masked_inputs(inputs, mode, bucket, allowed),
                    tick, mode, bucket, config,
                ) for bucket in range(len(gp.SESSION_BUCKETS))
            }
            result_base = gp.simulate_gp_day(
                days[date], trades[date], base, config, mode, "original_gp"
            )
            result_gate = gp.simulate_gp_day(
                days[date], trades[date], gated, config, mode,
                "fill_conditioned_gate",
            )
            reference = references[date][mode]
            matched = bool(
                np.isclose(result_base["net_pnl_hkd"], reference["net_pnl_hkd"],
                           rtol=0, atol=1e-6)
                and result_base["maker_fills"] == reference["maker_fills"]
            )
            if not matched:
                raise AssertionError(f"original gamma baseline mismatch on {date} {mode}")
            rows.append({
                "date": date, "fill_mode": mode, "prior_dates": prior,
                "baseline_net_hkd": result_base["net_pnl_hkd"],
                "gated_net_hkd": result_gate["net_pnl_hkd"],
                "delta_hkd": result_gate["net_pnl_hkd"] - result_base["net_pnl_hkd"],
                "baseline_maker_fills": result_base["maker_fills"],
                "gated_maker_fills": result_gate["maker_fills"],
                "baseline_quoted_bid": result_base["quoted_bid"],
                "gated_quoted_bid": result_gate["quoted_bid"],
                "baseline_quoted_ask": result_base["quoted_ask"],
                "gated_quoted_ask": result_gate["quoted_ask"],
                "baseline_market_lots": result_base["market_order_lots"],
                "gated_market_lots": result_gate["market_order_lots"],
                "original_baseline_reconciled": matched,
            })
        # End-of-day execution calibration joins the window only after replay.
        stats[date] = gp.collect_daily_calibration_stats(days[date], trades[date], config)
        print("REPLAY", date, flush=True)

    summary = []
    for mode in gp.FILL_MODES:
        mode_rows = [row for row in rows if row["fill_mode"] == mode]
        for label, subset in (
            ("early", [r for r in mode_rows if r["date"] <= "2026-07-17"]),
            ("late", [r for r in mode_rows if r["date"] >= "2026-07-20"]),
            ("month", mode_rows),
        ):
            summary.append({
                "fill_mode": mode, "period": label, "test_days": len(subset),
                "first_date": subset[0]["date"], "last_date": subset[-1]["date"],
                "baseline_net_hkd": sum(r["baseline_net_hkd"] for r in subset),
                "gated_net_hkd": sum(r["gated_net_hkd"] for r in subset),
                "no_trade_net_hkd": 0.0,
                "delta_hkd": sum(r["delta_hkd"] for r in subset),
                "baseline_maker_fills": sum(r["baseline_maker_fills"] for r in subset),
                "gated_maker_fills": sum(r["gated_maker_fills"] for r in subset),
            })
    output = {
        "gamma": args.gamma,
        "gate_source": str(args.gate_results),
        "gate_uses_only_past_completed_days": all(
            all(p < r["date"] for p in r["prior_dates"]) for r in rows
        ),
        "all_38_baselines_match_original": all(r["original_baseline_reconciled"] for r in rows),
        "summary": summary,
        "daily": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print("SAVED", args.output, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
