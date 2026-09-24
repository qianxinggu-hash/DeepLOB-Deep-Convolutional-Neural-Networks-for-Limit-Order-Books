#!/usr/bin/env python3
"""July 2026 causal Hybrid drift sensitivity across inventory penalties.

The pre-July checkpoint and daily prior-only drift/GP calibrations are fixed
across gammas. Each gamma compares a martingale QVI with the same gamma's
Hybrid QVI, using the corrected post-fill drift reward in gp_qvi_model.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

import run_hybrid_month_experiment as month
import gp_qvi_model as gp


HERE = Path(__file__).resolve().parent
DEFAULT_GAMMAS = (10.0, 5.0, 3.125, 1.5625, 1.0, 0.3, 0.1, 0.03)
REFERENCE_KEYS = (
    "net_pnl_hkd", "gross_pnl_hkd", "all_in_fees_hkd",
    "maker_turnover_hkd", "market_turnover_hkd",
    "maker_fills", "quote_decisions", "quoted_bid", "quoted_ask",
    "market_orders", "market_order_lots",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument("--checkpoint", type=Path, default=month.DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output/hybrid_gamma_sweep")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--drift-states", type=int, default=5)
    parser.add_argument("--prior-days", type=int, default=5)
    parser.add_argument("--limit-test-days", type=int, default=None)
    args = parser.parse_args()
    if not args.gammas or any(not math.isfinite(g) or g < 0 for g in args.gammas):
        parser.error("gammas must be finite and nonnegative")
    if min(args.batch_size, args.drift_states, args.prior_days) < 1:
        parser.error("batch-size, drift-states, and prior-days must be positive")
    if args.limit_test_days is not None and args.limit_test_days < 1:
        parser.error("limit-test-days must be positive")
    args.gammas = tuple(sorted(set(args.gammas), reverse=True))
    return args


def same_result(actual: dict, reference: dict) -> bool:
    return all(
        math.isclose(actual[k], reference[k], rel_tol=0, abs_tol=1e-6)
        if isinstance(actual[k], float)
        else actual[k] == reference[k]
        for k in REFERENCE_KEYS
    )


def summarize(tests: list[dict], gamma: float, mode: str) -> dict:
    base = [d["results"][mode]["martingale_gp"] for d in tests]
    drift = [d["results"][mode]["hybrid_drift_gp"] for d in tests]
    audits = [d["counterfactual_action_audit_at_hybrid_visited_states"][mode] for d in tests]
    quotes = sum(a["visited_quotes"] for a in audits)
    delta = sum(d["results"][mode]["net_pnl_delta_hkd"] for d in tests)
    return {
        "gamma": gamma,
        "fill_mode": mode,
        "test_days": len(tests),
        "quotes": quotes,
        "baseline_net_pnl_hkd": sum(r["net_pnl_hkd"] for r in base),
        "drift_net_pnl_hkd": sum(r["net_pnl_hkd"] for r in drift),
        "net_pnl_delta_hkd": delta,
        "baseline_gross_pnl_hkd": sum(r["gross_pnl_hkd"] for r in base),
        "drift_gross_pnl_hkd": sum(r["gross_pnl_hkd"] for r in drift),
        "baseline_all_in_fees_hkd": sum(r["all_in_fees_hkd"] for r in base),
        "drift_all_in_fees_hkd": sum(r["all_in_fees_hkd"] for r in drift),
        "baseline_maker_fills": sum(r["maker_fills"] for r in base),
        "drift_maker_fills": sum(r["maker_fills"] for r in drift),
        "baseline_market_order_lots": sum(r["market_order_lots"] for r in base),
        "drift_market_order_lots": sum(r["market_order_lots"] for r in drift),
        "baseline_quoted_sides": sum(r["quoted_bid"] + r["quoted_ask"] for r in base),
        "drift_quoted_sides": sum(r["quoted_bid"] + r["quoted_ask"] for r in drift),
        "baseline_max_abs_inventory_lots": max(r["max_abs_inventory_lots"] for r in base),
        "drift_max_abs_inventory_lots": max(r["max_abs_inventory_lots"] for r in drift),
        "drift_nonzero_inventory_visited_share": sum(
            a["nonzero_inventory_visited_share"] * a["visited_quotes"] for a in audits
        ) / quotes,
        "any_action_changed_count": sum(a["any_changed_count"] for a in audits),
        "bid_changed_count": sum(a["bid_changed_count"] for a in audits),
        "ask_changed_count": sum(a["ask_changed_count"] for a in audits),
        "impulse_changed_count": sum(a["impulse_changed_count"] for a in audits),
        "days_drift_better": sum(d["results"][mode]["net_pnl_delta_hkd"] > 1e-8 for d in tests),
        "days_drift_worse": sum(d["results"][mode]["net_pnl_delta_hkd"] < -1e-8 for d in tests),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, dates, days, trades = month._load_inputs()
    if len(dates) != 20 or not all(d.startswith("2026-07") for d in dates):
        raise ValueError("expected exactly 20 clean July 2026 trading days")
    checkpoint = month.verify_checkpoint(args.checkpoint, dates[0])
    adapter = month.HybridDriftAdapter(args.checkpoint, args.device)
    base_config = gp.GPConfig(
        max_inventory_lots=10,
        inventory_penalty_gamma=5.0,
        quote_horizon_snapshots=adapter.horizon_snapshots,
    )
    needed_dates = dates[:1 + (args.limit_test_days or len(dates) - 1)]
    stats, signals = {}, {}
    for date in needed_dates:
        print("PREPARE", date, flush=True)
        day = days[date]
        day.features = None
        stats[date] = gp.collect_daily_calibration_stats(day, trades[date], base_config)
        signals[date] = month.get_day_signal(day, adapter, args.batch_size)
        print("SIGNAL", date, len(signals[date].indices), flush=True)

    reference = json.loads((HERE / "output/gamma_results.json").read_text())
    reference_by_gamma = {
        float(variant["config"]["inventory_penalty_gamma"]): {
            day["test_date"]: day["results"] for day in variant["tests"]
        }
        for variant in reference["variants"].values()
    }
    reference_checks: dict[str, bool] = {}
    tests_by_gamma: dict[float, list[dict]] = {g: [] for g in args.gammas}
    daily_rows: list[dict] = []
    for index, date in enumerate(needed_dates[1:], start=1):
        prior = dates[max(0, index - args.prior_days):index]
        gp_inputs = gp.aggregate_calibration_stats([stats[d] for d in prior], base_config)
        prior_signals = [signals[d] for d in prior]
        calibrator = month.fit_probability_move_calibrator(
            np.concatenate([s.probabilities[s.has_model_history] for s in prior_signals]),
            np.concatenate([s.true_move_ticks[s.has_model_history] for s in prior_signals]),
        )
        test_signal = signals[date]
        expected_ticks = calibrator.transform(test_signal.probabilities)
        expected_ticks[~test_signal.has_model_history] = 0.0
        drift_sequences = []
        for day_date, prior_signal in zip(prior, prior_signals):
            prior_expected = calibrator.transform(prior_signal.probabilities)
            prior_expected[~prior_signal.has_model_history] = 0.0
            prior_tick = gp.infer_tick_hkd(days[day_date])
            prior_drift = prior_expected * prior_tick / prior_signal.causal_durations_seconds
            for segment in np.unique(prior_signal.segment_ids):
                mask = (prior_signal.segment_ids == segment) & prior_signal.has_model_history
                if np.any(mask):
                    drift_sequences.append(prior_drift[mask])
        raw_regime = gp.fit_gp_drift_regime(drift_sequences, args.drift_states)
        source_interval = float(np.median(np.concatenate([
            s.quote_intervals_seconds for s in prior_signals
        ])))
        drift_regime = gp.rescale_gp_drift_regime(
            raw_regime, source_interval, base_config.dt_seconds
        )
        for gamma in args.gammas:
            config = replace(base_config, inventory_penalty_gamma=gamma)
            print("REPLAY", date, "gamma", gamma, flush=True)
            result = month.run_one_day(
                days[date], trades[date], gp_inputs, config,
                test_signal, expected_ticks, drift_regime,
            )
            result.update({
                "gamma": gamma,
                "calibration_dates": prior,
                "calibrator": asdict(calibrator),
                "drift_regime": asdict(drift_regime),
                "drift_transition_source_interval_seconds": source_interval,
            })
            tests_by_gamma[gamma].append(result)
            month.write_json(args.output_dir / f"day_{date}_gamma_{gamma:g}.json", result)
            for mode in gp.FILL_MODES:
                baseline = result["results"][mode]["martingale_gp"]
                drift = result["results"][mode]["hybrid_drift_gp"]
                audit = result["counterfactual_action_audit_at_hybrid_visited_states"][mode]
                daily_rows.append({
                    "date": date, "gamma": gamma, "fill_mode": mode,
                    "calibration_dates": "|".join(prior),
                    "baseline_net_pnl_hkd": baseline["net_pnl_hkd"],
                    "drift_net_pnl_hkd": drift["net_pnl_hkd"],
                    "net_pnl_delta_hkd": result["results"][mode]["net_pnl_delta_hkd"],
                    "baseline_maker_fills": baseline["maker_fills"],
                    "drift_maker_fills": drift["maker_fills"],
                    "baseline_market_order_lots": baseline["market_order_lots"],
                    "drift_market_order_lots": drift["market_order_lots"],
                    "baseline_max_abs_inventory_lots": baseline["max_abs_inventory_lots"],
                    "drift_max_abs_inventory_lots": drift["max_abs_inventory_lots"],
                    "drift_nonzero_inventory_visited_share": audit["nonzero_inventory_visited_share"],
                    "any_action_changed_count": audit["any_changed_count"],
                })
                if gamma in reference_by_gamma:
                    reference_checks[f"{date}_{gamma:g}_{mode}"] = same_result(
                        baseline, reference_by_gamma[gamma][date][mode]
                    )
            month.write_json(args.output_dir / "progress.json", {
                "planned_dates": needed_dates[1:],
                "gammas": args.gammas,
                "completed_date_gamma": [
                    [d["date"], g] for g, rows in tests_by_gamma.items() for d in rows
                ],
            })
            print(
                "DONE", date, gamma,
                "through", round(result["results"]["through"]["net_pnl_delta_hkd"], 3),
                "touch", round(result["results"]["touch"]["net_pnl_delta_hkd"], 3),
                flush=True,
            )

    summary = [
        summarize(tests_by_gamma[g], g, mode)
        for g in args.gammas for mode in gp.FILL_MODES
    ]
    all_prior = all(
        all(p < row["date"] for p in row["calibration_dates"])
        for rows in tests_by_gamma.values() for row in rows
    )
    reconciliation = all(reference_checks.values())
    validation = {
        "strictly_prior_calibration": all_prior,
        "pre_july_checkpoint": checkpoint["validation_date"] < dates[0],
        "baseline_reference_count": len(reference_checks),
        "baseline_matches_existing_gamma_study": reconciliation,
        "baseline_reference_failures": [k for k, ok in reference_checks.items() if not ok],
        "action_audit_complete": all(
            row["counterfactual_action_audit_at_hybrid_visited_states"][mode]["visited_quotes"]
            == row["signal"]["quotes"]
            for rows in tests_by_gamma.values() for row in rows for mode in gp.FILL_MODES
        ),
        "daily_cashflow_reconciles": all(
            math.isclose(
                result["net_pnl_hkd"],
                result["gross_pnl_hkd"] - result["all_in_fees_hkd"],
                rel_tol=0, abs_tol=1e-6,
            )
            for rows in tests_by_gamma.values() for row in rows
            for mode in gp.FILL_MODES
            for result in row["results"][mode].values() if isinstance(result, dict)
        ),
    }
    validation["all_passed"] = all(v for k, v in validation.items() if k not in (
        "baseline_reference_count", "baseline_reference_failures"
    ))
    output = {
        "experiment": "Causal pre-July Hybrid GP/QVI gamma sensitivity",
        "period": "2026-07",
        "instrument": "HK 07709",
        "checkpoint": checkpoint,
        "design": {
            "clean_dates": dates,
            "initial_calibration_date": dates[0],
            "test_dates": needed_dates[1:],
            "rolling_prior_days": args.prior_days,
            "gamma_units": "HKD per lot squared second",
            "gammas": args.gammas,
            "only_policy_parameter_varied": "inventory_penalty_gamma",
            "signal_and_gp_calibration": "identical across gamma for each test date",
            "time_denominator": "causal prior 20-snapshot elapsed time",
            "drift_qvi": "includes expected post-fill within-step drift exposure",
            "same_gamma_comparison": True,
            "gp_config_except_gamma": gp.config_asdict(base_config),
            "exploratory_selection_warning": "July was used to inspect gamma performance; a winner is not independent out-of-sample evidence",
        },
        "provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "month_replay": "0920/run_hybrid_month_experiment.py",
            "solver": "0906/gp_qvi_model.py",
            "original_gamma_reference": "0920/output/gamma_results.json",
        },
        "summary": summary,
        "validation": validation,
    }
    month.write_json(args.output_dir / "results.json", output)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (args.output_dir / "daily.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(daily_rows[0]))
        writer.writeheader()
        writer.writerows(daily_rows)
    if not validation["all_passed"]:
        raise AssertionError(f"gamma sweep validation failed: {validation}")
    print("SAVED", (args.output_dir / "results.json").resolve(), flush=True)


if __name__ == "__main__":
    main()
