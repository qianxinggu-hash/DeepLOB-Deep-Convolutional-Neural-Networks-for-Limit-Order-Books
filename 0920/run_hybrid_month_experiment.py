#!/usr/bin/env python3
"""Leakage-free July 2026 GP/QVI replay with a pre-July Hybrid drift signal.

Use July 2 as the initial calibration day and evaluate the next 19 clean
trading days.  Every tested day refits the probability-to-move map, drift
chain, and GP execution inputs using the preceding five clean days only.
The Hybrid checkpoint must have been trained and selected before July.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0906"))
sys.path.append(str(ROOT / "0824"))

from gp_qvi_model import (  # noqa: E402
    FILL_MODES,
    SESSION_BUCKETS,
    GPConfig,
    aggregate_calibration_stats,
    assign_gp_drift_states,
    collect_daily_calibration_stats,
    compact_time_to_day_ms,
    config_asdict,
    fit_gp_drift_regime,
    infer_tick_hkd,
    rescale_gp_drift_regime,
    select_quote_indices,
    simulate_gp_day,
    solve_gp_drift_policy,
    solve_gp_policy,
)
from hybrid_drift_adapter import (  # noqa: E402
    HybridDriftAdapter,
    causal_prediction_horizon_seconds,
    fit_probability_move_calibrator,
    future_mid_move_ticks,
    prediction_horizon_seconds,
)
from run_gp_qvi_experiment import _load_inputs  # noqa: E402
from train_7709_deeplob import DayData  # noqa: E402


PRICE_COLUMNS = np.arange(0, 40, 2)
SIZE_COLUMNS = np.arange(1, 40, 2)
DEFAULT_OUTPUT = HERE / "output/hybrid_month_causal"
DEFAULT_CHECKPOINT = HERE / "output/hybrid_month/pre_july_hybrid.pt"


@dataclass
class DaySignal:
    date: str
    indices: np.ndarray
    probabilities: np.ndarray
    true_move_ticks: np.ndarray
    causal_durations_seconds: np.ndarray
    realized_durations_seconds: np.ndarray
    has_model_history: np.ndarray
    segment_ids: np.ndarray
    quote_intervals_seconds: np.ndarray


class ObservedImpulse:
    """Compare both policies counterfactually at Hybrid-visited states."""

    def __init__(self, drift_policy: Any, baseline_policy: Any, records: list[dict]):
        self.drift_policy = drift_policy
        self.baseline_policy = baseline_policy
        self.records = records

    def __getitem__(self, key: tuple[int, int, int, int]) -> Any:
        step, drift_state, spread_state, q_index = (int(value) for value in key)
        drift = self.drift_policy
        baseline = self.baseline_policy
        drift_target = int(drift.impulse_target_array[key])
        baseline_target = int(baseline.impulse_target[step, spread_state, q_index])
        drift_bid = int(drift.make_bid_action[step, drift_state, spread_state, drift_target])
        drift_ask = int(drift.make_ask_action[step, drift_state, spread_state, drift_target])
        baseline_bid = int(baseline.make_bid_action[step, spread_state, baseline_target])
        baseline_ask = int(baseline.make_ask_action[step, spread_state, baseline_target])
        self.records.append(
            {
                "inventory_before_lots": q_index - drift.config.max_inventory_lots,
                "drift_state": drift_state,
                "spread_state": spread_state,
                "step": step,
                "impulse_changed": int(drift_target != baseline_target),
                "bid_changed": int(drift_bid != baseline_bid),
                "ask_changed": int(drift_ask != baseline_ask),
                "any_changed": int(
                    (drift_target, drift_bid, drift_ask)
                    != (baseline_target, baseline_bid, baseline_ask)
                ),
                "baseline_bid": baseline_bid,
                "baseline_ask": baseline_ask,
                "drift_bid": drift_bid,
                "drift_ask": drift_ask,
            }
        )
        return drift_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--drift-states", type=int, default=5)
    parser.add_argument("--prior-days", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--limit-test-days", type=int, default=None)
    args = parser.parse_args()
    if min(args.batch_size, args.drift_states, args.prior_days) < 1:
        parser.error("batch-size, drift-states and prior-days must be positive")
    if args.gamma != 5.0:
        parser.error("this month study fixes gamma=5 to match the 0920 reference")
    if args.limit_test_days is not None and args.limit_test_days < 1:
        parser.error("limit-test-days must be positive")
    return args


def verify_checkpoint(path: Path, first_test_date: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    validation = str(config["validation_date"])
    training_dates = (
        "2025-11-06", "2025-11-13", "2025-12-09"
    ) if validation == "2025-12-19" else ()
    if not validation < first_test_date:
        raise ValueError("checkpoint model selection is not before the test month")
    # The saved training script uses every processed date strictly before its
    # validation date.  Require the dedicated pre-July split used here.
    if validation != "2025-12-19" or str(config["test_date"]) != "2026-07-09":
        raise ValueError("expected the dedicated pre-July Hybrid checkpoint")
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_dates": training_dates,
        "validation_date": validation,
        "checkpoint_test_date": str(config["test_date"]),
        "best_epoch": int(checkpoint["epoch"]),
        "checkpoint_test_macro_f1": float(checkpoint["test_metrics"]["macro_f1"]),
    }


def get_day_signal(day: Any, adapter: HybridDriftAdapter, batch_size: int) -> DaySignal:
    indices = select_quote_indices(day, day.eligible, adapter.horizon_snapshots)
    if not len(indices):
        raise ValueError(f"{day.date}: no eligible quote decisions")
    # MBO execution caches are in HKD and shares.  The Hybrid checkpoint was
    # trained on the neural reconstruction's price/100 and size/100000 units.
    model_book = day.book.astype(np.float32, copy=True)
    model_book[:, PRICE_COLUMNS] /= 100.0
    model_book[:, SIZE_COLUMNS] /= 100_000.0
    model_day = DayData(day.date, day.path, model_book, day.segments, day.metadata)
    segment_starts = np.maximum.accumulate(
        np.where(
            np.r_[True, day.segments[1:] != day.segments[:-1]],
            np.arange(len(day.segments)),
            0,
        )
    )
    # Hybrid auxiliary features use a 100-snapshot lag in addition to the
    # 100-row model window.  The first eligible quote of each session has only
    # 99 earlier snapshots.  Give that quote zero drift to avoid a negative
    # index reading the end of the day; GP replay still uses every quote.
    has_history = indices - segment_starts[indices] >= 100
    probabilities = np.full((len(indices), 3), 1.0 / 3.0, dtype=np.float64)
    probabilities[has_history] = adapter.probabilities(
        model_day, indices[has_history], batch_size
    )
    tick = infer_tick_hkd(day)
    true_moves = future_mid_move_ticks(
        model_book,
        day.segments,
        indices,
        adapter.horizon_snapshots,
        tick / 100.0,
    )
    # The time of snapshot t+k is unknown at quote time t. Estimate its
    # duration from the already observed t-k..t interval, inside the session.
    causal_durations = causal_prediction_horizon_seconds(
        day.send_times, day.segments, indices, adapter.horizon_snapshots
    )
    realized_durations = prediction_horizon_seconds(
        day.send_times, day.segments, indices, adapter.horizon_snapshots
    )
    times = compact_time_to_day_ms(day.send_times)
    interval_parts = []
    for segment in np.unique(day.segments[indices]):
        selected = indices[day.segments[indices] == segment]
        gaps = np.diff(times[selected]) / 1000.0
        interval_parts.append(gaps[gaps > 0.0])
    intervals = np.concatenate(interval_parts)
    if not len(intervals):
        raise ValueError(f"{day.date}: no positive quote intervals")
    return DaySignal(
        date=day.date,
        indices=indices,
        probabilities=probabilities,
        true_move_ticks=true_moves,
        causal_durations_seconds=causal_durations,
        realized_durations_seconds=realized_durations,
        has_model_history=has_history,
        segment_ids=day.segments[indices],
        quote_intervals_seconds=intervals,
    )


def summarize_signal(signal: DaySignal, expected_ticks: np.ndarray) -> dict[str, Any]:
    actual = signal.true_move_ticks
    predicted = np.asarray(expected_ticks, dtype=np.float64)
    valid = signal.has_model_history
    corr = (
        float(np.corrcoef(actual[valid], predicted[valid])[0, 1])
        if valid.sum() > 1 and np.std(actual[valid]) > 0 and np.std(predicted[valid]) > 0
        else None
    )
    def summary(values: np.ndarray) -> dict[str, Any]:
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "quantiles_0_1_5_25_50_75_95_99_100": np.quantile(
                values, [0, .01, .05, .25, .5, .75, .95, .99, 1]
            ).tolist(),
            "positive_share": float(np.mean(values > 1e-12)),
            "negative_share": float(np.mean(values < -1e-12)),
            "zero_share": float(np.mean(np.abs(values) <= 1e-12)),
        }
    return {
        "quotes": len(predicted),
        "neutral_history_quotes": int((~valid).sum()),
        "unscored_quote_count": int((~valid).sum()),
        "predicted_move_ticks": summary(predicted),
        "absolute_predicted_move_ticks_quantiles_0_25_50_75_90_95_99_100": (
            np.quantile(np.abs(predicted), [0, .25, .5, .75, .9, .95, .99, 1]).tolist()
        ),
        "actual_20_snapshot_mid_move_ticks": summary(actual),
        "prediction_actual_correlation_model_quotes": corr,
        "prediction_actual_mae_ticks_model_quotes": float(
            np.mean(np.abs(predicted[valid] - actual[valid]))
        ),
        "causal_horizon_seconds_median": float(
            np.median(signal.causal_durations_seconds)
        ),
        "realized_horizon_seconds_median_diagnostic_only": float(
            np.median(signal.realized_durations_seconds)
        ),
    }


def summarize_action_records(records: list[dict], expected: int) -> dict[str, Any]:
    if len(records) != expected:
        raise AssertionError(f"action audit has {len(records)} vs {expected} quotes")
    count = len(records)
    result: dict[str, Any] = {"visited_quotes": count}
    for field in ("impulse_changed", "bid_changed", "ask_changed", "any_changed"):
        changes = sum(row[field] for row in records)
        result[field + "_count"] = changes
        result[field + "_share"] = changes / count
    result["quoted_side_changed_count"] = (
        result["bid_changed_count"] + result["ask_changed_count"]
    )
    result["quoted_side_changed_share"] = result["quoted_side_changed_count"] / (2 * count)
    result["baseline_none_side_share_at_hybrid_states"] = sum(
        (row["baseline_bid"] == 0) + (row["baseline_ask"] == 0)
        for row in records
    ) / (2 * count)
    result["drift_none_side_share_at_hybrid_states"] = sum(
        (row["drift_bid"] == 0) + (row["drift_ask"] == 0)
        for row in records
    ) / (2 * count)
    result["nonzero_inventory_visited_share"] = sum(
        row["inventory_before_lots"] != 0 for row in records
    ) / count
    return result


def run_one_day(
    day: Any,
    trades: Any,
    gp_inputs: Any,
    config: GPConfig,
    signal: DaySignal,
    expected_ticks: np.ndarray,
    drift_regime: Any,
) -> dict[str, Any]:
    tick = infer_tick_hkd(day)
    states = assign_gp_drift_states(
        expected_ticks * tick / signal.causal_durations_seconds, drift_regime
    )
    outcomes: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for mode in FILL_MODES:
        baseline = {
            bucket: solve_gp_policy(gp_inputs, tick, mode, bucket, config)
            for bucket in range(len(SESSION_BUCKETS))
        }
        drift = {
            bucket: solve_gp_drift_policy(
                gp_inputs, tick, mode, bucket, config, drift_regime
            )
            for bucket in range(len(SESSION_BUCKETS))
        }
        baseline_result = simulate_gp_day(
            day, trades, baseline, config, mode, "martingale_gp"
        )
        records: list[dict] = []
        observed_drift = {}
        for bucket, policy in drift.items():
            observed = copy.copy(policy)
            observed.impulse_target_array = policy.impulse_target
            observed.impulse_target = ObservedImpulse(observed, baseline[bucket], records)
            observed_drift[bucket] = observed
        drift_result = simulate_gp_day(
            day, trades, observed_drift, config, mode, "hybrid_drift_gp", states
        )
        outcomes[mode] = {
            "martingale_gp": baseline_result,
            "hybrid_drift_gp": drift_result,
            "net_pnl_delta_hkd": (
                drift_result["net_pnl_hkd"] - baseline_result["net_pnl_hkd"]
            ),
        }
        audits[mode] = summarize_action_records(records, len(signal.indices))
    return {
        "date": day.date,
        "tick_hkd": tick,
        "signal": summarize_signal(signal, expected_ticks),
        "drift_state_counts": {
            drift_regime.labels[z]: int(np.sum(states == z))
            for z in range(len(drift_regime.labels))
        },
        "unscored_nearest_zero_drift_state": drift_regime.labels[
            int(np.argmin(np.abs(drift_regime.drift_hkd_per_second)))
        ],
        "results": outcomes,
        "counterfactual_action_audit_at_hybrid_visited_states": audits,
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, dates, days, trades = _load_inputs()
    if len(dates) != 20 or not all(d.startswith("2026-07") for d in dates):
        raise ValueError("expected exactly the 20 clean July 2026 trading days")
    checkpoint_info = verify_checkpoint(args.checkpoint, dates[0])
    adapter = HybridDriftAdapter(args.checkpoint, args.device)
    config = GPConfig(
        max_inventory_lots=10,
        inventory_penalty_gamma=args.gamma,
        quote_horizon_snapshots=adapter.horizon_snapshots,
    )
    for day in days.values():
        day.features = None  # 0824's logistic features are unused here.
    stats = {}
    signals = {}
    needed_dates = dates[: 1 + (args.limit_test_days or len(dates) - 1)]
    for date in needed_dates:
        print("PREPARE", date, flush=True)
        stats[date] = collect_daily_calibration_stats(days[date], trades[date], config)
        signals[date] = get_day_signal(days[date], adapter, args.batch_size)
        print("SIGNAL", date, len(signals[date].indices), flush=True)

    tests = []
    for index, date in enumerate(needed_dates[1:], start=1):
        prior = dates[max(0, index - args.prior_days) : index]
        gp_inputs = aggregate_calibration_stats([stats[d] for d in prior], config)
        prior_signals = [signals[d] for d in prior]
        calibrator = fit_probability_move_calibrator(
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
            prior_tick = infer_tick_hkd(days[day_date])
            prior_drift = (
                prior_expected * prior_tick / prior_signal.causal_durations_seconds
            )
            for segment in np.unique(prior_signal.segment_ids):
                mask = (prior_signal.segment_ids == segment) & prior_signal.has_model_history
                if np.any(mask):
                    drift_sequences.append(prior_drift[mask])
        raw_regime = fit_gp_drift_regime(drift_sequences, args.drift_states)
        source_interval = float(np.median(np.concatenate([
            s.quote_intervals_seconds for s in prior_signals
        ])))
        drift_regime = rescale_gp_drift_regime(
            raw_regime, source_interval, config.dt_seconds
        )
        print("REPLAY", date, "prior", prior, flush=True)
        day_result = run_one_day(
            days[date], trades[date], gp_inputs, config, test_signal,
            expected_ticks, drift_regime,
        )
        day_result.update({
            "calibration_dates": prior,
            "calibrator": asdict(calibrator),
            "drift_regime": asdict(drift_regime),
            "drift_transition_source_interval_seconds": source_interval,
        })
        tests.append(day_result)
        write_json(args.output_dir / f"day_{date}.json", day_result)
        write_json(args.output_dir / "progress.json", {
            "completed_dates": [row["date"] for row in tests],
            "planned_dates": needed_dates[1:],
        })
        print(
            "DONE", date,
            "through delta", day_result["results"]["through"]["net_pnl_delta_hkd"],
            "touch delta", day_result["results"]["touch"]["net_pnl_delta_hkd"],
            flush=True,
        )

    reference = json.loads((HERE / "output/gamma_results.json").read_text())
    reference_days = {
        day["test_date"]: day["results"]
        for day in reference["variants"]["gamma_5"]["tests"]
    }
    baseline_checks = {}
    for day in tests:
        date = day["date"]
        for mode in FILL_MODES:
            actual = day["results"][mode]["martingale_gp"]
            prior_result = reference_days[date][mode]
            baseline_checks[f"{date}_{mode}"] = bool(
                all(
                    np.isclose(actual[key], prior_result[key], rtol=0, atol=1e-6)
                    for key in (
                        "net_pnl_hkd", "gross_pnl_hkd", "all_in_fees_hkd",
                        "maker_turnover_hkd", "market_turnover_hkd",
                    )
                )
                and all(
                    actual[key] == prior_result[key]
                    for key in (
                        "maker_fills", "quote_decisions", "quoted_bid", "quoted_ask",
                        "market_orders", "market_order_lots",
                    )
                )
            )
    if not all(baseline_checks.values()):
        failed = [key for key, passed in baseline_checks.items() if not passed]
        raise AssertionError(f"baseline fails exact 0920 daily reconciliation: {failed}")
    summary = {}
    daily_rows = []
    for mode in FILL_MODES:
        base_rows = [d["results"][mode]["martingale_gp"] for d in tests]
        drift_rows = [d["results"][mode]["hybrid_drift_gp"] for d in tests]
        deltas = [d["results"][mode]["net_pnl_delta_hkd"] for d in tests]
        actions = [d["counterfactual_action_audit_at_hybrid_visited_states"][mode] for d in tests]
        total_quotes = sum(a["visited_quotes"] for a in actions)
        summary[mode] = {
            "test_days": len(tests),
            "baseline_net_pnl_hkd": float(sum(r["net_pnl_hkd"] for r in base_rows)),
            "drift_net_pnl_hkd": float(sum(r["net_pnl_hkd"] for r in drift_rows)),
            "net_pnl_delta_hkd": float(sum(deltas)),
            "days_drift_better": sum(delta > 0 for delta in deltas),
            "days_drift_worse": sum(delta < 0 for delta in deltas),
            "baseline_maker_fills": sum(r["maker_fills"] for r in base_rows),
            "drift_maker_fills": sum(r["maker_fills"] for r in drift_rows),
            "visited_quotes": total_quotes,
            "any_action_changed_count": sum(a["any_changed_count"] for a in actions),
            "any_action_changed_share": (
                sum(a["any_changed_count"] for a in actions) / total_quotes
            ),
            "bid_changed_count": sum(a["bid_changed_count"] for a in actions),
            "ask_changed_count": sum(a["ask_changed_count"] for a in actions),
            "impulse_changed_count": sum(a["impulse_changed_count"] for a in actions),
        }
        for day in tests:
            result = day["results"][mode]
            audit = day["counterfactual_action_audit_at_hybrid_visited_states"][mode]
            daily_rows.append({
                "date": day["date"], "fill_mode": mode,
                "calibration_dates": "|".join(day["calibration_dates"]),
                "quotes": day["signal"]["quotes"],
                "neutral_history_quotes": day["signal"]["neutral_history_quotes"],
                "predicted_mean_ticks": day["signal"]["predicted_move_ticks"]["mean"],
                "predicted_std_ticks": day["signal"]["predicted_move_ticks"]["std"],
                "prediction_actual_correlation": day["signal"]["prediction_actual_correlation_model_quotes"],
                "baseline_net_pnl_hkd": result["martingale_gp"]["net_pnl_hkd"],
                "drift_net_pnl_hkd": result["hybrid_drift_gp"]["net_pnl_hkd"],
                "net_pnl_delta_hkd": result["net_pnl_delta_hkd"],
                "baseline_maker_fills": result["martingale_gp"]["maker_fills"],
                "drift_maker_fills": result["hybrid_drift_gp"]["maker_fills"],
                "any_action_changed_count": audit["any_changed_count"],
                "any_action_changed_share": audit["any_changed_share"],
                "bid_changed_count": audit["bid_changed_count"],
                "ask_changed_count": audit["ask_changed_count"],
                "impulse_changed_count": audit["impulse_changed_count"],
            })
    output = {
        "experiment": "Pre-July Hybrid drift in July 2026 GP/QVI replay",
        "period": "2026-07",
        "instrument": "HK 07709",
        "checkpoint": checkpoint_info,
        "design": {
            "clean_dates": dates,
            "initial_calibration_date": dates[0],
            "test_dates": needed_dates[1:],
            "rolling_prior_days": args.prior_days,
            "price_source": "0824 tolerant MBO cache; HKD execution book converted price/100 and size/100000 only for Hybrid inference",
            "inference_boundary": "first quote of each session is unscored because the 100-lag auxiliary feature lacks full history; it is assigned the regime nearest zero, which may have nonzero drift",
            "drift_time_denominator": "elapsed time over the previous 20 snapshots in the same session, observable at the quote; realized future duration is diagnostic only",
            "action_comparison": "at each state actually visited by the Hybrid replay, compare its impulse and post-impulse bid/ask actions with the baseline policy evaluated counterfactually at that same pre-impulse inventory, time and spread",
            "gp_config": config_asdict(config),
            "drift_states": args.drift_states,
            "touch_through_are_execution_proxies": True,
        },
        "summary": summary,
        "tests": tests,
        "validation": {
            "all_calibration_dates_strictly_prior": all(
                all(prior < row["date"] for prior in row["calibration_dates"])
                for row in tests
            ),
            "baseline_matches_0920_each_day": baseline_checks,
            "baseline_matches_0920_all": all(baseline_checks.values()),
            "checkpoint_selected_pre_july": checkpoint_info["validation_date"] < dates[0],
        },
        "limitations": [
            "The Hybrid Equation (4) classification label includes a past-to-current price component already known at prediction time; classification F1 does not establish forward markout skill.",
            "The 20-snapshot training label spans about 74 seconds on the pooled pre-July training days but about 1.9 seconds on the July 9 processed test day; the transferred model targets a different clock horizon.",
            "The pre-July Hybrid checkpoint has weak July 9 classification performance; assess conditional move calibration and action changes before interpreting PnL.",
            "The Hybrid model was trained on a different LOB reconstruction; July MBO inputs are put into the same numeric units, but their feature distribution may differ.",
            "Touch/through fills are execution proxies; short-sale feasibility and borrow fees are not modeled.",
        ],
    }
    write_json(args.output_dir / "results.json", output)
    with (args.output_dir / "daily.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(daily_rows[0]))
        writer.writeheader()
        writer.writerows(daily_rows)
    print("SAVED", (args.output_dir / "results.json").resolve(), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
