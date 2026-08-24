#!/usr/bin/env python3
"""Independent QA for paired PnL and 5/10/20 second markouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from directional_market_maker import compact_time_to_day_ms
from run_multiday_strategy import ensure_mbp_cache, load_day


HERE = Path(__file__).resolve().parent


def main() -> None:
    result_path = HERE / "output" / "directional_markout_results.json"
    markout_path = HERE / "output" / "directional_markouts.csv"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    markouts = pd.read_csv(markout_path)
    checks: dict[str, bool] = {}

    checks["requested_horizons_present"] = set(markouts["horizon_seconds"]) == {5, 10, 20}
    checks["both_sides_present"] = set(markouts["side"]) == {"buy", "sell"}
    checks["both_fill_modes_present"] = set(markouts["fill_mode"]) == {"touch", "through"}
    checks["adverse_cost_sign_identity"] = bool(
        np.allclose(
            markouts["adverse_selection_cost_bps"],
            -markouts["signed_mid_move_bps"],
            atol=1e-9,
        )
    )
    maker_sign = np.where(markouts["side"].eq("buy"), 1.0, -1.0)
    execution_recomputed = (
        maker_sign
        * (markouts["future_mid_hkd"] - markouts["price_hkd"])
        / markouts["price_hkd"]
        * 10_000.0
    )
    checks["execution_markouts_recomputed"] = bool(
        np.allclose(execution_recomputed, markouts["execution_markout_bps"], atol=1e-7)
    )

    day_cache: dict[tuple[str, str], object] = {}
    for family, dates in {
        "strict_mbo": ("2026-07-21", "2026-08-07"),
        "absolute_mbp": ("2026-07-22", "2026-08-07"),
    }.items():
        for date in dates:
            path = (
                HERE / "data" / f"hk07709_{date}_mbo_strict_s10.npz"
                if family == "strict_mbo"
                else ensure_mbp_cache(date)
            )
            day_cache[(family, date)] = load_day(date, family, path)

    full_time_check = True
    full_mid_check = True
    for (family, date), frame in markouts.groupby(["family", "date"]):
        day = day_cache[(family, date)]
        times = compact_time_to_day_ms(day.send_times)
        mids = (day.book[:, 0].astype(np.float64) + day.book[:, 2]) / 2.0
        fill_ms = frame["trade_time_ms"].to_numpy(np.int64)
        horizons_ms = frame["horizon_seconds"].to_numpy(np.int64) * 1_000
        fill_indices = np.searchsorted(times, fill_ms, side="left")
        future_indices = np.searchsorted(times, fill_ms + horizons_ms, side="left")
        full_time_check &= bool(
            np.all(fill_indices < len(times))
            and np.all(future_indices < len(times))
            and np.all(times[future_indices] >= fill_ms + horizons_ms)
            and np.all(day.segments[fill_indices] == day.segments[future_indices])
        )
        full_mid_check &= bool(
            np.allclose(mids[fill_indices], frame["mid_at_fill_hkd"], atol=1e-7)
            and np.allclose(mids[future_indices], frame["future_mid_hkd"], atol=1e-7)
        )
    checks["markout_horizons_are_causal_and_same_session"] = full_time_check
    checks["markout_midpoints_match_source_books"] = full_mid_check

    pnl_deltas_match = True
    model_deltas_match = True
    for test in results["tests"]:
        pnl_deltas_match &= set(test["strategies"]) == {
            "directional",
            "classic_as",
            "gp_style_discrete",
            "paired_no_signal",
        }
        for fill_mode in ("through", "touch"):
            direction = test["strategies"]["directional"][fill_mode]
            baseline = test["strategies"]["paired_no_signal"][fill_mode]
            comparison = test["paired_incremental_pnl"][fill_mode]
            pnl_deltas_match &= np.isclose(
                float(comparison["incremental_gross_pnl_hkd"]),
                float(direction["gross_pnl_hkd"]) - float(baseline["gross_pnl_hkd"]),
                atol=1e-7,
            )
            pnl_deltas_match &= np.isclose(
                float(comparison["incremental_all_in_net_pnl_hkd"]),
                float(direction["net_pnl_hkd"]) - float(baseline["net_pnl_hkd"]),
                atol=1e-7,
            )
            direction_parameters = direction["parameters"]
            baseline_parameters = baseline["parameters"]
            pnl_deltas_match &= all(
                direction_parameters[key] == baseline_parameters[key]
                for key in direction_parameters
                if key != "signal_strength"
            )
            pnl_deltas_match &= baseline_parameters["signal_strength"] == 0.0
            for baseline_name in ("classic_as", "gp_style_discrete", "paired_no_signal"):
                reported_delta = test["model_comparisons"][fill_mode][
                    f"directional_minus_{baseline_name}"
                ]
                model_baseline = test["strategies"][baseline_name][fill_mode]
                model_deltas_match &= np.isclose(
                    float(reported_delta["all_in_net_pnl_hkd"]),
                    float(direction["net_pnl_hkd"])
                    - float(model_baseline["net_pnl_hkd"]),
                    atol=1e-7,
                )
                model_deltas_match &= np.isclose(
                    float(reported_delta["max_drawdown_hkd"]),
                    float(direction["max_drawdown_hkd"])
                    - float(model_baseline["max_drawdown_hkd"]),
                    atol=1e-7,
                )
    checks["paired_pnl_and_parameters_recomputed"] = bool(pnl_deltas_match)
    checks["as_gp_directional_model_deltas_recomputed"] = bool(model_deltas_match)

    summaries_match = True
    summary_frame = pd.DataFrame(results["markout_summary"])
    grouping = [
        "family",
        "split",
        "date",
        "fill_mode",
        "strategy",
        "side",
        "horizon_seconds",
    ]
    recomputed = (
        markouts.groupby(grouping, as_index=False)
        .agg(
            fills=("adverse_selection_cost_bps", "size"),
            mean_adverse_selection_cost_bps=("adverse_selection_cost_bps", "mean"),
            mean_execution_markout_bps=("execution_markout_bps", "mean"),
        )
        .sort_values(grouping)
        .reset_index(drop=True)
    )
    reported = summary_frame.sort_values(grouping).reset_index(drop=True)
    summaries_match &= recomputed[grouping].equals(reported[grouping])
    summaries_match &= bool(
        np.array_equal(recomputed["fills"], reported["fills"])
        and np.allclose(
            recomputed["mean_adverse_selection_cost_bps"],
            reported["mean_adverse_selection_cost_bps"],
            atol=1e-9,
        )
        and np.allclose(
            recomputed["mean_execution_markout_bps"],
            reported["mean_execution_markout_bps"],
            atol=1e-9,
        )
    )
    checks["markout_summaries_recomputed"] = bool(summaries_match)
    checks["reconstruction_families_not_mixed"] = all(
        all(
            set(model_tuning["validation_dates"])
            <= (
                {"2026-07-09", "2026-07-21"}
                if test["family"] == "strict_mbo"
                else {"2026-07-21", "2026-07-22"}
            )
            for model_tuning in test["tuning"].values()
        )
        for test in results["tests"]
    )
    checks["same_validation_dates_for_all_models"] = all(
        len(
            {
                tuple(model_tuning["validation_dates"])
                for model_tuning in test["tuning"].values()
            }
        )
        == 1
        for test in results["tests"]
    )
    checks["all_mbo_test_sources_untainted"] = all(
        not record["tainted"]
        for record in results["data_quality"]["strict_mbo"].values()
    )

    output = {
        "as_of": results["as_of"],
        "markout_rows_checked": int(len(markouts)),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "assessment": "Share with caveats" if all(checks.values()) else "Needs revision",
        "caveats": results["limitations"],
    }
    output_path = HERE / "output" / "directional_markout_validation.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
