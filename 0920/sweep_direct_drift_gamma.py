#!/usr/bin/env python3
"""Replay prior-day Ridge endpoint drift across GP inventory penalties.

The forecast caches are produced by run_direct_drift_signal.py. They contain
out-of-sample test-day predictions and prior-day predictions from the model
available before each test date. No test-day markout enters policy fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import run_hybrid_month_experiment as month
import gp_qvi_model as gp


HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "output/direct_drift_signal"
DEFAULT_OUTPUT = HERE / "output/direct_drift_gamma_sweep"
DEFAULT_GAMMAS = (5.0, 3.125, 1.5625, 0.5, 0.1)
REFERENCE_FIELDS_FLOAT = (
    "net_pnl_hkd", "gross_pnl_hkd", "all_in_fees_hkd",
    "maker_turnover_hkd", "market_turnover_hkd",
)
REFERENCE_FIELDS_INT = (
    "maker_fills", "quote_decisions", "quoted_bid", "quoted_ask",
    "market_orders", "market_order_lots",
)


def quote_intervals(day: object, indices: np.ndarray) -> np.ndarray:
    times = gp.compact_time_to_day_ms(day.send_times)
    parts = []
    for segment in np.unique(day.segments[indices]):
        selected = indices[day.segments[indices] == segment]
        gaps = np.diff(times[selected]) / 1000.0
        parts.append(gaps[gaps > 0])
    intervals = np.concatenate(parts)
    if not len(intervals):
        raise ValueError(f"{day.date}: no positive quote interval")
    return intervals


def valid_history(day: object, indices: np.ndarray) -> np.ndarray:
    starts = np.maximum.accumulate(np.where(
        np.r_[True, day.segments[1:] != day.segments[:-1]],
        np.arange(len(day.segments)), 0,
    ))
    return indices - starts[indices] >= 100


def prior_regime(
    archive: np.lib.npyio.NpzFile,
    prior_dates: list[str],
    days: dict[str, object],
    config: gp.GPConfig,
    state_count: int,
) -> tuple[gp.GPDriftRegime, float]:
    ordinals = archive["prior_day_ordinals"]
    indices = archive["prior_quote_indices"]
    predictions = archive["prior_expected_move_ticks"]
    durations = archive["prior_causal_horizon_seconds"]
    ticks = archive["prior_tick_hkd"]
    segments = archive["prior_segment_ids"]
    lengths = archive["prior_day_quote_counts"]
    if lengths.sum() != len(indices) or len(lengths) != len(prior_dates):
        raise AssertionError("prior signal cache lengths do not reconcile")
    sequences = []
    for ordinal, date in enumerate(prior_dates):
        day = days[date]
        mask_day = ordinals == ordinal
        prior_indices = indices[mask_day]
        expected_indices = gp.select_quote_indices(
            day, day.eligible, config.quote_horizon_snapshots
        )
        if not np.array_equal(prior_indices, expected_indices):
            raise AssertionError(f"{date}: prior quote indices differ")
        if not np.array_equal(segments[mask_day], day.segments[prior_indices]):
            raise AssertionError(f"{date}: prior segment IDs differ")
        valid = valid_history(day, prior_indices)
        drift = predictions[mask_day] * ticks[mask_day] / durations[mask_day]
        for segment in np.unique(segments[mask_day]):
            selected = (segments[mask_day] == segment) & valid
            if selected.any():
                sequences.append(drift[selected])
    raw = gp.fit_gp_drift_regime(sequences, state_count)
    source_interval = float(np.median(np.concatenate([
        quote_intervals(days[d], gp.select_quote_indices(
            days[d], days[d].eligible, config.quote_horizon_snapshots
        )) for d in prior_dates
    ])))
    return gp.rescale_gp_drift_regime(raw, source_interval, config.dt_seconds), source_interval


def baseline_matches(result: dict, reference: dict) -> bool:
    return all(math.isclose(result[k], reference[k], rel_tol=0, abs_tol=1e-6)
               for k in REFERENCE_FIELDS_FLOAT) and all(
                   result[k] == reference[k] for k in REFERENCE_FIELDS_INT
               )


def aggregate(rows: list[dict], gamma: float, mode: str, phase: str) -> dict:
    chosen = [r for r in rows if r["gamma"] == gamma and r["fill_mode"] == mode
              and (phase == "all" or r["phase"] == phase)]
    return {
        "gamma": gamma,
        "fill_mode": mode,
        "phase": phase,
        "test_days": len(chosen),
        "baseline_net_pnl_hkd": float(sum(r["baseline_net_pnl_hkd"] for r in chosen)),
        "drift_net_pnl_hkd": float(sum(r["drift_net_pnl_hkd"] for r in chosen)),
        "net_pnl_delta_hkd": float(sum(r["net_pnl_delta_hkd"] for r in chosen)),
        "baseline_maker_fills": sum(r["baseline_maker_fills"] for r in chosen),
        "drift_maker_fills": sum(r["drift_maker_fills"] for r in chosen),
        "action_changes": sum(r["any_action_changed_count"] for r in chosen),
        "quote_decisions": sum(r["quote_decisions"] for r in chosen),
        "issue_flagged_test_days": sum(r["source_issue_flagged"] for r in chosen),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument("--drift-states", type=int, default=5)
    parser.add_argument("--limit-test-days", type=int, default=None)
    parser.add_argument(
        "--clean-prior", action="store_true",
        help="use cache-selected prior days with no recorded order errors or feed gaps",
    )
    args = parser.parse_args()
    if (not args.gammas or any(not math.isfinite(g) or g < 0 for g in args.gammas)
            or args.drift_states < 1 or (args.limit_test_days is not None
            and args.limit_test_days < 1)):
        parser.error("nonnegative finite gammas, positive states/days required")
    gammas = tuple(sorted(set(args.gammas), reverse=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, dates, days, trades = month._load_inputs()
    needed_dates = dates[:1 + (args.limit_test_days or len(dates)-1)]
    base_config = gp.GPConfig(max_inventory_lots=10, inventory_penalty_gamma=5.0)
    stats = {date: gp.collect_daily_calibration_stats(days[date], trades[date], base_config)
             for date in needed_dates}
    reference = json.loads((HERE / "output/gamma_results.json").read_text())
    by_gamma = {
        float(variant["config"]["inventory_penalty_gamma"]): {
            row["test_date"]: row["results"] for row in variant["tests"]
        } for variant in reference["variants"].values()
    }
    quality = json.loads((HERE / "output/combination_study/source_quality.json").read_text())
    quality_by_date = {row["date"]: row for row in quality["days"]}
    clean_reference = (
        {row["date"]: row for row in json.loads(
            (HERE / "output/combination_study/clean_prior_gamma5.json").read_text()
        )["daily"]}
        if args.clean_prior else {}
    )

    daily_rows = []
    checks = {}
    for ordinal, date in enumerate(needed_dates[1:], start=1):
        cache_path = args.signal_cache / f"day_{date}.npz"
        with np.load(cache_path, allow_pickle=False) as cache:
            prior_dates = [str(d) for d in cache["prior_dates"]]
            expected_prior = (
                [d for d in dates[:ordinal] if d == dates[0] or
                 quality_by_date[d]["quality_group"] == "no_recorded_reconstruction_issue"][-5:]
                if args.clean_prior else dates[max(0, ordinal-5):ordinal]
            )
            if prior_dates != expected_prior:
                raise AssertionError(f"{date}: signal and GP prior dates differ")
            indices = cache["quote_indices"]
            expected_indices = gp.select_quote_indices(
                days[date], days[date].eligible, base_config.quote_horizon_snapshots
            )
            if not np.array_equal(indices, expected_indices):
                raise AssertionError(f"{date}: test quote indices differ")
            predicted = cache["expected_move_ticks"].copy()
            causal_seconds = cache["causal_horizon_seconds"].copy()
            if not np.allclose(cache["predicted_drift_hkd_per_second"],
                               predicted * gp.infer_tick_hkd(days[date]) / causal_seconds,
                               rtol=0, atol=1e-12):
                raise AssertionError(f"{date}: cached drift conversion differs")
            signal = month.DaySignal(
                date=date,
                indices=indices.copy(),
                probabilities=np.zeros((len(indices), 3), dtype=np.float64),
                true_move_ticks=cache["actual_move_ticks_diagnostic"].copy(),
                causal_durations_seconds=causal_seconds,
                realized_durations_seconds=cache["realized_horizon_seconds_diagnostic"].copy(),
                has_model_history=cache["has_100_snapshot_history"].copy(),
                segment_ids=cache["segment_ids"].copy(),
                quote_intervals_seconds=quote_intervals(days[date], indices),
            )
            regime, source_interval = prior_regime(
                cache, prior_dates, days, base_config, args.drift_states
            )
        gp_inputs = gp.aggregate_calibration_stats([stats[d] for d in prior_dates], base_config)
        for gamma in gammas:
            config = replace(base_config, inventory_penalty_gamma=gamma)
            print("REPLAY", date, gamma, flush=True)
            result = month.run_one_day(
                days[date], trades[date], gp_inputs, config,
                signal, predicted, regime,
            )
            result.update({
                "gamma": gamma,
                "calibration_dates": prior_dates,
                "drift_regime": month.asdict(regime),
                "drift_transition_source_interval_seconds": source_interval,
                "signal_source": str(cache_path),
            })
            month.write_json(args.output_dir / f"day_{date}_gamma_{gamma:g}.json", result)
            for mode in gp.FILL_MODES:
                baseline = result["results"][mode]["martingale_gp"]
                drift = result["results"][mode]["hybrid_drift_gp"]
                if gamma in by_gamma:
                    if args.clean_prior and gamma == 5.0:
                        checks[f"{date}_{gamma:g}_{mode}"] = math.isclose(
                            baseline["net_pnl_hkd"],
                            clean_reference[date][mode]["clean_prior_net_pnl_hkd"],
                            rel_tol=0, abs_tol=1e-6,
                        )
                    elif not args.clean_prior:
                        checks[f"{date}_{gamma:g}_{mode}"] = baseline_matches(
                            baseline, by_gamma[gamma][date][mode]
                        )
                audit = result["counterfactual_action_audit_at_hybrid_visited_states"][mode]
                daily_rows.append({
                    "date": date,
                    "gamma": gamma,
                    "fill_mode": mode,
                    "phase": "exploration_2026_07_03_to_17" if date <= "2026-07-17"
                             else "late_check_2026_07_20_to_30",
                    "source_issue_flagged": quality_by_date[date]["quality_group"]
                                            == "recorded_reconstruction_issue",
                    "calibration_issue_count": (
                        0 if args.clean_prior else
                        quality_by_date[date]["rolling_calibration_issue_count"]
                    ),
                    "quote_decisions": audit["visited_quotes"],
                    "baseline_net_pnl_hkd": baseline["net_pnl_hkd"],
                    "drift_net_pnl_hkd": drift["net_pnl_hkd"],
                    "net_pnl_delta_hkd": result["results"][mode]["net_pnl_delta_hkd"],
                    "baseline_maker_fills": baseline["maker_fills"],
                    "drift_maker_fills": drift["maker_fills"],
                    "any_action_changed_count": audit["any_changed_count"],
                })
            month.write_json(args.output_dir / "progress.json", {
                "gammas": gammas,
                "completed_date_gamma": [[r["date"], r["gamma"]] for r in daily_rows
                                           if r["fill_mode"] == "through"],
            })
            print("DONE", date, gamma,
                  "through", round(result["results"]["through"]["net_pnl_delta_hkd"], 3),
                  "touch", round(result["results"]["touch"]["net_pnl_delta_hkd"], 3),
                  flush=True)

    if not all(checks.values()):
        raise AssertionError(f"baseline mismatches: {[k for k,v in checks.items() if not v]}")
    summary = [aggregate(daily_rows, gamma, mode, phase)
               for gamma in gammas for mode in gp.FILL_MODES
               for phase in ("all", "exploration_2026_07_03_to_17",
                             "late_check_2026_07_20_to_30")]
    output = {
        "experiment": "Causal direct endpoint drift GP/QVI gamma sensitivity",
        "period": "2026-07",
        "design": {
            "signal_cache": str(args.signal_cache),
            "gammas": gammas,
            "drift_states": args.drift_states,
            "training": (
                "fixed Ridge on preceding up to five days with no recorded reconstruction issue"
                if args.clean_prior else
                "fixed Ridge on preceding five source-audit eligible days"
            ),
            "gp_calibration": (
                "preceding up to five days with no recorded reconstruction issue"
                if args.clean_prior else
                "preceding five source-audit eligible days"
            ),
            "clean_prior": args.clean_prior,
            "test_day_source_quality": "0920/output/combination_study/source_quality.json",
            "decision_horizon": "20 snapshots converted with observed preceding 20-snapshot duration",
            "no_test_day_labels_in_fitted_signal_or_gp": True,
            "comparison": "each drift policy versus martingale policy at the same gamma",
        },
        "summary": summary,
        "validation": {
            "baseline_reference_checks": len(checks),
            "baseline_reference": (
                "clean_prior_gamma5 sensitivity" if args.clean_prior and checks
                else "original gamma study" if checks
                else None
            ),
            "baseline_matches_reference": all(checks.values()) if checks else None,
            "calibration_strictly_prior": all(
                all(pd < row["date"] for pd in dates[max(0, dates.index(row["date"])-5):dates.index(row["date"])])
                for row in daily_rows
            ),
        },
    }
    month.write_json(args.output_dir / "results.json", output)
    with (args.output_dir / "daily.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(daily_rows[0]))
        writer.writeheader()
        writer.writerows(daily_rows)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("SAVED", (args.output_dir / "results.json").resolve(), flush=True)


if __name__ == "__main__":
    main()
