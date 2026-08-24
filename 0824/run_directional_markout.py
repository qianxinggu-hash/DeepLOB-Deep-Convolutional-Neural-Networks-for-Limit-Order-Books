#!/usr/bin/env python3
"""Multi-period direction-enhancement tests with 5/10/20 second markouts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from directional_market_maker import (
    ASParameters,
    QuoteParameters,
    compact_time_to_day_ms,
    fit_alpha_calibrator,
    fit_direction_classifier,
    future_mid_move_ticks,
    load_or_extract_mbp_executions,
    load_or_extract_trades,
    predict_direction,
    select_quote_indices,
    simulate_market_maker,
)
from run_multiday_strategy import HORIZON, RAW, ensure_mbp_cache, load_day, per_day_split


HERE = Path(__file__).resolve().parent
MARKOUT_SECONDS = (5, 10, 20)


def subset_values(
    eligible: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(eligible, selected)
    if np.any(positions >= len(eligible)) or not np.array_equal(eligible[positions], selected):
        raise ValueError("selected indices are not a subset of eligible indices")
    return values[positions]


def make_validation_record(
    day: object,
    trades: object,
    eligible: np.ndarray,
    score: np.ndarray,
) -> dict[str, object]:
    return {
        "day": day,
        "trades": trades,
        "eligible": eligible,
        "score": score,
        "future_ticks": future_mid_move_ticks(day, eligible, HORIZON),
    }


def pooled_calibrator(records: list[dict[str, object]]) -> object:
    return fit_alpha_calibrator(
        np.concatenate([record["score"] for record in records]),
        np.concatenate([record["future_ticks"] for record in records]),
    )


def estimate_as_market_inputs(
    records: list[dict[str, object]],
) -> tuple[float, float, dict[str, object]]:
    """Estimate AS horizon variance and exponential fill decay on validation only."""

    future_ticks = np.concatenate([record["future_ticks"] for record in records])
    horizon_variance = float(np.var(future_ticks, ddof=1))
    distances = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    rates: list[float] = []
    rate_audit: list[dict[str, object]] = []
    for distance in distances.astype(int):
        fills = 0
        side_quotes = 0
        for record in records:
            day = record["day"]
            eligible = record["eligible"]
            quote_indices = select_quote_indices(day, eligible, HORIZON)
            neutral = QuoteParameters(
                base_distance_ticks=int(distance),
                signal_strength=0.0,
                inventory_skew_ticks=0.0,
                max_inventory_lots=999,
            )
            result, _ = simulate_market_maker(
                day,
                record["trades"],
                quote_indices,
                np.zeros(len(quote_indices), dtype=np.float64),
                neutral,
                fill_mode="through",
                strategy_name="as_intensity_probe",
            )
            fills += int(result["maker_fills"])
            side_quotes += int(result["quoted_bid"]) + int(result["quoted_ask"])
        rate = fills / max(side_quotes, 1)
        rates.append(rate)
        rate_audit.append(
            {
                "distance_ticks": int(distance),
                "fills": fills,
                "side_quotes": side_quotes,
                "fill_rate": rate,
            }
        )
    floor = 0.5 / max(sum(row["side_quotes"] for row in rate_audit), 1)
    log_rates = np.log(np.clip(np.asarray(rates), floor, 1.0))
    slope = float(np.polyfit(distances, log_rates, 1)[0])
    order_decay = max(0.05, -slope)
    return horizon_variance, order_decay, {
        "horizon_variance_ticks2": horizon_variance,
        "order_decay_per_tick": order_decay,
        "intensity_fit": "OLS slope of log empirical through-fill rate at 0/1/2 ticks; k=max(0.05,-slope)",
        "probes": rate_audit,
    }


def tune_model_parameters(
    records: list[dict[str, object]],
    calibrator: object,
    model_name: str,
) -> tuple[QuoteParameters | ASParameters, dict[str, object]]:
    if model_name == "directional":
        parameter_grid: list[QuoteParameters | ASParameters] = [
            QuoteParameters(base, strength, skew, cap)
            for base in (1, 2)
            for strength in (2.0, 4.0, 8.0, 16.0)
            for skew in (0.5, 1.0)
            for cap in (3, 5)
        ]
        market_input_audit = None
    elif model_name == "gp_style_discrete":
        parameter_grid = [
            QuoteParameters(base, 0.0, skew, cap)
            for base in (1, 2)
            for skew in (0.5, 1.0, 2.0)
            for cap in (3, 5)
        ]
        market_input_audit = None
    elif model_name == "classic_as":
        variance, decay, market_input_audit = estimate_as_market_inputs(records)
        parameter_grid = [
            ASParameters(gamma, decay, variance, cap)
            for gamma in (0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)
            for cap in (3, 5)
        ]
    else:
        raise ValueError(model_name)

    candidates: list[tuple[float, object, dict[str, object]]] = []
    for parameters in parameter_grid:
        total_net = 0.0
        total_drawdown = 0.0
        total_fills = 0
        per_period: list[dict[str, object]] = []
        for record in records:
            day = record["day"]
            eligible = record["eligible"]
            quote_indices = select_quote_indices(day, eligible, HORIZON)
            if model_name == "classic_as":
                quote_alpha = np.zeros(len(quote_indices), dtype=np.float64)
            else:
                alpha_all = calibrator.transform(record["score"])
                quote_alpha = subset_values(eligible, alpha_all, quote_indices)
            result, _ = simulate_market_maker(
                day,
                record["trades"],
                quote_indices,
                quote_alpha,
                parameters,
                fill_mode="through",
                strategy_name=model_name,
            )
            total_net += float(result["net_pnl_hkd"])
            total_drawdown += float(result["max_drawdown_hkd"])
            total_fills += int(result["maker_fills"])
            per_period.append(
                {
                    "date": day.date,
                    "net_pnl_hkd": result["net_pnl_hkd"],
                    "max_drawdown_hkd": result["max_drawdown_hkd"],
                    "maker_fills": result["maker_fills"],
                }
            )
        objective = total_net - 0.25 * total_drawdown
        if total_fills < 20:
            objective = -1e18
        candidates.append(
            (
                objective,
                parameters,
                {
                    "score": objective,
                    "net_pnl_hkd": total_net,
                    "max_drawdown_hkd": total_drawdown,
                    "maker_fills": total_fills,
                    "validation_periods": per_period,
                },
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, parameters, audit = candidates[0]
    return parameters, {
        "model": model_name,
        "objective": "all-in net PnL - 0.25 * max drawdown, using conservative through fills",
        "candidate_count": len(candidates),
        "validation_dates": [record["day"].date for record in records],
        "calibrator": asdict(calibrator) if model_name == "directional" else None,
        "as_market_inputs": market_input_audit,
        "best_parameters": asdict(parameters),
        "best": audit,
        "top_five": [
            {"parameters": asdict(item[1]), **item[2]}
            for item in candidates[:5]
        ],
    }


def fill_markouts(
    day: object,
    events: list[dict[str, object]],
    family: str,
    split: str,
) -> list[dict[str, object]]:
    snapshot_ms = compact_time_to_day_ms(day.send_times)
    mids = (day.book[:, 0].astype(np.float64) + day.book[:, 2]) / 2.0
    rows: list[dict[str, object]] = []
    for event in events:
        fill_ms = int(event["trade_time_ms"])
        fill_index = int(np.searchsorted(snapshot_ms, fill_ms, side="left"))
        if fill_index >= len(day.book):
            continue
        segment = int(day.segments[fill_index])
        execution_price = float(event["price_hkd"])
        maker_sign = 1.0 if event["side"] == "buy" else -1.0
        mid_at_fill = float(mids[fill_index])
        for seconds in MARKOUT_SECONDS:
            future_index = int(
                np.searchsorted(snapshot_ms, fill_ms + seconds * 1_000, side="left")
            )
            if future_index >= len(day.book) or int(day.segments[future_index]) != segment:
                continue
            future_mid = float(mids[future_index])
            signed_mid_move_bps = maker_sign * (future_mid - mid_at_fill) / mid_at_fill * 10_000.0
            execution_markout_bps = (
                maker_sign * (future_mid - execution_price) / execution_price * 10_000.0
            )
            row = dict(event)
            row.update(
                {
                    "family": family,
                    "split": split,
                    "horizon_seconds": seconds,
                    "mid_at_fill_hkd": mid_at_fill,
                    "future_mid_hkd": future_mid,
                    "signed_mid_move_bps": signed_mid_move_bps,
                    "adverse_selection_cost_bps": -signed_mid_move_bps,
                    "execution_markout_bps": execution_markout_bps,
                }
            )
            rows.append(row)
    return rows


def summarize_markouts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    keys = sorted(
        {
            (
                row["family"],
                row["split"],
                row["date"],
                row["fill_mode"],
                row["strategy"],
                row["side"],
                row["horizon_seconds"],
            )
            for row in rows
        }
    )
    for family, split, date, fill_mode, strategy, side, horizon in keys:
        selected = [
            row
            for row in rows
            if (
                row["family"],
                row["split"],
                row["date"],
                row["fill_mode"],
                row["strategy"],
                row["side"],
                row["horizon_seconds"],
            )
            == (family, split, date, fill_mode, strategy, side, horizon)
        ]
        adverse = np.asarray(
            [row["adverse_selection_cost_bps"] for row in selected], dtype=np.float64
        )
        execution = np.asarray(
            [row["execution_markout_bps"] for row in selected], dtype=np.float64
        )
        summaries.append(
            {
                "family": family,
                "split": split,
                "date": date,
                "fill_mode": fill_mode,
                "strategy": strategy,
                "side": side,
                "horizon_seconds": horizon,
                "fills": int(len(selected)),
                "mean_adverse_selection_cost_bps": float(np.mean(adverse)),
                "median_adverse_selection_cost_bps": float(np.median(adverse)),
                "adverse_move_share": float(np.mean(adverse > 0)),
                "mean_execution_markout_bps": float(np.mean(execution)),
                "median_execution_markout_bps": float(np.median(execution)),
            }
        )
    return summaries


def adverse_selection_deltas(
    summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    direction_rows = [row for row in summaries if row["strategy"] == "directional"]
    for direction in direction_rows:
        matches = [
            row
            for row in summaries
            if row["strategy"] == "paired_no_signal"
            and all(
                row[key] == direction[key]
                for key in (
                    "family",
                    "split",
                    "date",
                    "fill_mode",
                    "side",
                    "horizon_seconds",
                )
            )
        ]
        if len(matches) != 1:
            raise ValueError("paired markout summary missing")
        baseline = matches[0]
        results.append(
            {
                key: direction[key]
                for key in (
                    "family",
                    "split",
                    "date",
                    "fill_mode",
                    "side",
                    "horizon_seconds",
                )
            }
            | {
                "directional_fills": direction["fills"],
                "paired_no_signal_fills": baseline["fills"],
                "directional_adverse_cost_bps": direction[
                    "mean_adverse_selection_cost_bps"
                ],
                "paired_no_signal_adverse_cost_bps": baseline[
                    "mean_adverse_selection_cost_bps"
                ],
                "adverse_cost_reduction_bps": baseline[
                    "mean_adverse_selection_cost_bps"
                ]
                - direction["mean_adverse_selection_cost_bps"],
            }
        )
    return results


def evaluate_test(
    family: str,
    split: str,
    day: object,
    trades: object,
    eligible: np.ndarray,
    score: np.ndarray,
    classification: dict[str, object],
    calibrator: object,
    directional_parameters: QuoteParameters,
    gp_parameters: QuoteParameters,
    as_parameters: ASParameters,
    tuning: dict[str, dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    quote_indices = select_quote_indices(day, eligible, HORIZON)
    quote_alpha = subset_values(
        eligible, calibrator.transform(score), quote_indices
    )
    strategies = {
        "directional": directional_parameters,
        "classic_as": as_parameters,
        "gp_style_discrete": gp_parameters,
        "paired_no_signal": replace(directional_parameters, signal_strength=0.0),
    }
    strategy_results: dict[str, object] = {}
    all_events: list[dict[str, object]] = []
    markout_rows: list[dict[str, object]] = []
    for strategy, strategy_parameters in strategies.items():
        strategy_results[strategy] = {}
        for fill_mode in ("through", "touch"):
            result, events = simulate_market_maker(
                day,
                trades,
                quote_indices,
                quote_alpha,
                strategy_parameters,
                fill_mode,
                strategy,
                record_events=True,
            )
            for event in events:
                event["family"] = family
                event["split"] = split
            strategy_results[strategy][fill_mode] = result
            all_events.extend(events)
            markout_rows.extend(fill_markouts(day, events, family, split))
    comparisons: dict[str, dict[str, object]] = {}
    model_comparisons: dict[str, dict[str, dict[str, float]]] = {}
    for fill_mode in ("through", "touch"):
        direction = strategy_results["directional"][fill_mode]
        baseline = strategy_results["paired_no_signal"][fill_mode]
        comparisons[fill_mode] = {
            "incremental_gross_pnl_hkd": float(direction["gross_pnl_hkd"])
            - float(baseline["gross_pnl_hkd"]),
            "incremental_official_only_net_pnl_hkd": float(
                direction["net_pnl_official_only_hkd"]
            )
            - float(baseline["net_pnl_official_only_hkd"]),
            "incremental_all_in_net_pnl_hkd": float(direction["net_pnl_hkd"])
            - float(baseline["net_pnl_hkd"]),
            "incremental_max_drawdown_hkd": float(direction["max_drawdown_hkd"])
            - float(baseline["max_drawdown_hkd"]),
        }
        model_comparisons[fill_mode] = {}
        for baseline_name in ("classic_as", "gp_style_discrete", "paired_no_signal"):
            model_baseline = strategy_results[baseline_name][fill_mode]
            model_comparisons[fill_mode][f"directional_minus_{baseline_name}"] = {
                "gross_pnl_hkd": float(direction["gross_pnl_hkd"])
                - float(model_baseline["gross_pnl_hkd"]),
                "all_in_net_pnl_hkd": float(direction["net_pnl_hkd"])
                - float(model_baseline["net_pnl_hkd"]),
                "max_drawdown_hkd": float(direction["max_drawdown_hkd"])
                - float(model_baseline["max_drawdown_hkd"]),
                "maker_fills": int(direction["maker_fills"])
                - int(model_baseline["maker_fills"]),
            }
    return (
        {
            "family": family,
            "split": split,
            "test_date": day.date,
            "classification": classification,
            "quote_opportunities": int(len(quote_indices)),
            "tuning": tuning,
            "parameters": {
                "directional": asdict(directional_parameters),
                "classic_as": asdict(as_parameters),
                "gp_style_discrete": asdict(gp_parameters),
            },
            "strategies": strategy_results,
            "paired_incremental_pnl": comparisons,
            "model_comparisons": model_comparisons,
        },
        all_events,
        markout_rows,
    )


def run_family(
    family: str,
    days: list[object],
    trades_by_date: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    first_day, second_day, third_day = days
    first_fit, first_validation = per_day_split(first_day)
    validation_estimator, validation_alpha = fit_direction_classifier(
        [first_day], [first_fit]
    )
    validation_score, validation_metrics = predict_direction(
        validation_estimator,
        validation_alpha,
        first_day,
        first_validation,
    )
    first_record = make_validation_record(
        first_day,
        trades_by_date[first_day.date],
        first_validation,
        validation_score,
    )
    first_calibrator = pooled_calibrator([first_record])
    first_directional, first_directional_tuning = tune_model_parameters(
        [first_record], first_calibrator, "directional"
    )
    first_gp, first_gp_tuning = tune_model_parameters(
        [first_record], first_calibrator, "gp_style_discrete"
    )
    first_as, first_as_tuning = tune_model_parameters(
        [first_record], first_calibrator, "classic_as"
    )
    first_tuning = {
        "directional": first_directional_tuning,
        "classic_as": first_as_tuning,
        "gp_style_discrete": first_gp_tuning,
    }

    full_first_estimator, full_first_alpha = fit_direction_classifier(
        [first_day], [first_day.eligible]
    )
    second_score, second_metrics = predict_direction(
        full_first_estimator,
        full_first_alpha,
        second_day,
        second_day.eligible,
    )
    first_test, first_events, first_markouts = evaluate_test(
        family,
        f"{first_day.date}_to_{second_day.date}",
        second_day,
        trades_by_date[second_day.date],
        second_day.eligible,
        second_score,
        second_metrics,
        first_calibrator,
        first_directional,
        first_gp,
        first_as,
        first_tuning,
    )

    second_record = make_validation_record(
        second_day,
        trades_by_date[second_day.date],
        second_day.eligible,
        second_score,
    )
    pooled_records = [first_record, second_record]
    second_calibrator = pooled_calibrator(pooled_records)
    second_directional, second_directional_tuning = tune_model_parameters(
        pooled_records, second_calibrator, "directional"
    )
    second_gp, second_gp_tuning = tune_model_parameters(
        pooled_records, second_calibrator, "gp_style_discrete"
    )
    second_as, second_as_tuning = tune_model_parameters(
        pooled_records, second_calibrator, "classic_as"
    )
    second_tuning = {
        "directional": second_directional_tuning,
        "classic_as": second_as_tuning,
        "gp_style_discrete": second_gp_tuning,
    }
    full_estimator, full_alpha = fit_direction_classifier(
        [first_day, second_day],
        [first_day.eligible, second_day.eligible],
    )
    third_score, third_metrics = predict_direction(
        full_estimator,
        full_alpha,
        third_day,
        third_day.eligible,
    )
    second_test, second_events, second_markouts = evaluate_test(
        family,
        f"{first_day.date}_{second_day.date}_to_{third_day.date}",
        third_day,
        trades_by_date[third_day.date],
        third_day.eligible,
        third_score,
        third_metrics,
        second_calibrator,
        second_directional,
        second_gp,
        second_as,
        second_tuning,
    )
    first_test["initial_validation_classification"] = validation_metrics
    second_test["pooled_validation_periods"] = [
        f"{first_day.date} last 30%",
        f"{second_day.date} full-day out-of-sample",
    ]
    return (
        [first_test, second_test],
        first_events + second_events,
        first_markouts + second_markouts,
    )


def main() -> None:
    mbo_dates = ("2026-07-09", "2026-07-21", "2026-08-07")
    mbo_days = [
        load_day(
            date,
            "strict_mbo",
            HERE / "data" / f"hk07709_{date}_mbo_strict_s10.npz",
        )
        for date in mbo_dates
    ]
    for day in mbo_days:
        if day.metadata.get("final_state_tainted"):
            raise RuntimeError(f"tainted MBO day: {day.date}")
    mbo_trades = {
        date: load_or_extract_trades(
            RAW / f"hk07709_{date}.csv",
            HERE / "data" / f"hk07709_{date}_trades.npz",
        )
        for date in mbo_dates
    }

    mbp_dates = ("2026-07-21", "2026-07-22", "2026-08-07")
    mbp_days = [
        load_day(date, "absolute_mbp", ensure_mbp_cache(date))
        for date in mbp_dates
    ]
    mbp_trades = {
        date: load_or_extract_mbp_executions(
            RAW / f"hk07709_{date}.npz",
            HERE / "data" / f"hk07709_{date}_mbp_executions.npz",
        )
        for date in mbp_dates
    }

    all_tests: list[dict[str, object]] = []
    all_events: list[dict[str, object]] = []
    all_markouts: list[dict[str, object]] = []
    for family, days, trades in (
        ("strict_mbo", mbo_days, mbo_trades),
        ("absolute_mbp", mbp_days, mbp_trades),
    ):
        print("FAMILY", family, flush=True)
        tests, events, markouts = run_family(family, days, trades)
        all_tests.extend(tests)
        all_events.extend(events)
        all_markouts.extend(markouts)
        for test in tests:
            print(
                "TEST",
                test["split"],
                f"accuracy={test['classification']['accuracy']:.4f}",
                f"through_delta={test['paired_incremental_pnl']['through']['incremental_all_in_net_pnl_hkd']:.2f}",
                flush=True,
            )

    markout_summary = summarize_markouts(all_markouts)
    adverse_deltas = adverse_selection_deltas(markout_summary)
    output = {
        "as_of": "2026-08-24",
        "question": "Does direction-enhanced passive quoting reduce adverse selection and improve paired market-making PnL?",
        "definitions": {
            "markout": "maker-signed value of the fill measured against the future mid-price after 5/10/20 seconds",
            "adverse_selection_cost": "negative maker-signed mid-price move from immediately after fill to the future horizon; positive values are harmful",
            "paired_incremental_pnl": "directional strategy PnL minus a strategy with identical parameters and signal_strength=0 on the same test day",
            "model_level_comparison": "direction+inventory versus independently validation-tuned classic AS and GP-style discrete inventory control",
            "conservative_estimate": "through: a hypothetical passive order fills only after the market trades one tick beyond its quote",
            "optimistic_estimate": "touch: a hypothetical passive order fills when a trade reaches its quote",
        },
        "method": {
            "markout_seconds": list(MARKOUT_SECONDS),
            "parameter_selection": "each model independently maximizes all-in net PnL minus 0.25 times max drawdown on exactly the same prior validation periods under through fills",
            "classic_as": "finite-horizon closed-form reservation price and spread; horizon variance and exponential fill-decay k are estimated on validation only",
            "gp_baseline": "discrete-tick GP-style approximation with inventory skew, hard inventory cap and terminal flattening; it is not the original Guilbaud-Pham QVI numerical solution",
            "second_stage_validation": "pool the first day's final 30% and the second day's full out-of-sample predictions",
            "families_kept_separate": True,
            "mbo_dates": list(mbo_dates),
            "mbp_dates": list(mbp_dates),
            "continuous_day_limit": "Only 2026-07-21 and 2026-07-22 are adjacent, and they are available together only in the MBP family.",
        },
        "tests": all_tests,
        "markout_summary": markout_summary,
        "adverse_selection_deltas": adverse_deltas,
        "data_quality": {
            "strict_mbo": {
                day.date: {
                    "snapshots": int(len(day.book)),
                    "tainted": bool(day.metadata.get("final_state_tainted")),
                    "trade_prints": int(len(mbo_trades[day.date].price_hkd)),
                }
                for day in mbo_days
            },
            "absolute_mbp": {
                day.date: {
                    "snapshots": int(len(day.book)),
                    "execution_prints": int(len(mbp_trades[day.date].price_hkd)),
                    "source_kind": mbp_trades[day.date].metadata["source_kind"],
                }
                for day in mbp_days
            },
        },
        "limitations": [
            "Exact hypothetical queue position is unavailable; touch and through remain sensitivity assumptions.",
            "The GP-style baseline preserves the practical discrete/inventory controls but is not a full numerical solution of the original Guilbaud-Pham QVI.",
            "Touch and through are upper/lower execution sensitivities, not statistical confidence intervals and not guaranteed PnL bounds.",
            "Only one pair of adjacent dates (07-21 and 07-22) is available, and only under MBP reconstruction.",
            "The second-stage validation pool contains two periods, still far below the requested multi-week continuous validation sample.",
            "Markouts are conditional on simulated fills, whose composition differs between directional and no-signal strategies.",
        ],
    }
    output_path = HERE / "output" / "directional_markout_results.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    event_path = HERE / "output" / "directional_markout_fills.csv"
    if all_events:
        with event_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_events[0]))
            writer.writeheader()
            writer.writerows(all_events)
    markout_path = HERE / "output" / "directional_markouts.csv"
    if all_markouts:
        with markout_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_markouts[0]))
            writer.writeheader()
            writer.writerows(all_markouts)
    comparison_rows: list[dict[str, object]] = []
    for test in all_tests:
        for fill_mode, estimate_label in (
            ("through", "conservative"),
            ("touch", "optimistic"),
        ):
            strategies = test["strategies"]
            comparisons = test["model_comparisons"][fill_mode]
            comparison_rows.append(
                {
                    "family": test["family"],
                    "test_date": test["test_date"],
                    "classification_accuracy": test["classification"]["accuracy"],
                    "fill_mode": fill_mode,
                    "estimate_label": estimate_label,
                    "directional_inventory_net_pnl_hkd": strategies["directional"][fill_mode]["net_pnl_hkd"],
                    "classic_as_net_pnl_hkd": strategies["classic_as"][fill_mode]["net_pnl_hkd"],
                    "gp_style_discrete_net_pnl_hkd": strategies["gp_style_discrete"][fill_mode]["net_pnl_hkd"],
                    "paired_no_signal_net_pnl_hkd": strategies["paired_no_signal"][fill_mode]["net_pnl_hkd"],
                    "directional_minus_as_hkd": comparisons["directional_minus_classic_as"]["all_in_net_pnl_hkd"],
                    "directional_minus_gp_hkd": comparisons["directional_minus_gp_style_discrete"]["all_in_net_pnl_hkd"],
                    "directional_signal_ablation_hkd": comparisons["directional_minus_paired_no_signal"]["all_in_net_pnl_hkd"],
                }
            )
    comparison_path = HERE / "output" / "as_gp_directional_comparison.csv"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(
        "WROTE",
        output_path,
        event_path,
        markout_path,
        comparison_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
