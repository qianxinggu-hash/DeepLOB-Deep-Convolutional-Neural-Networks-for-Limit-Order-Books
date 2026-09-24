#!/usr/bin/env python3
"""Walk-forward maker-fill quality gate for July 2026 GP replay.

The policy for a test day uses only preceding completed days.  All candidate
best/improve quotes are probed on those earlier days, independently of the
historical GP action, to estimate signed 3-second and 20-snapshot markouts.
Quotes with weak expected 3-second net markout have their execution hazard
set to zero before solving the GP policy.  Zero hazard causes the solver to
choose ``none`` for that side/action and recomputes impulse controls.

This is a proxy-fill experiment, not an estimate of actual queue fills.
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
sys.path.append(str(ROOT / "0824"))
import gp_qvi_model as gp  # noqa: E402
from run_gp_qvi_experiment import _load_inputs  # noqa: E402


MODES = gp.FILL_MODES
SHAPE = (len(MODES), len(gp.SIDES), len(gp.QUOTE_ACTIONS),
         len(gp.SESSION_BUCKETS), 6)
DEFAULT_OUTPUT = HERE / "output/fill_conditioned_gate"


def fresh_probe() -> dict[str, np.ndarray]:
    return {field: np.zeros(SHAPE, dtype=np.float64) for field in (
        "attempts", "fills", "count_3s", "sum_3s", "sum_sq_3s",
        "post_sum_3s", "count_20snap", "sum_20snap", "sum_sq_20snap",
        "post_sum_20snap", "duration_sum_20snap",
    )}


def probe_completed_day(day, trades, tick: float, config: gp.GPConfig) -> dict[str, np.ndarray]:
    """Collect every executable L1 quote probe after the entire day has ended."""
    result = fresh_probe()
    times = gp.compact_time_to_day_ms(day.send_times)
    segments = day.segments
    mids = (day.book[:, 0].astype(np.float64) + day.book[:, 2]) / 2.0
    quote_indices = gp.select_quote_indices(day, day.eligible, config.quote_horizon_snapshots)
    for index_value in quote_indices:
        index = int(index_value)
        expiry = index + config.quote_horizon_snapshots
        if expiry >= len(times) or segments[index] != segments[expiry]:
            continue
        ask, bid = float(day.book[index, 0]), float(day.book[index, 2])
        state = gp._spread_state(ask, bid, tick, config.spread_states)
        bucket = gp._bucket_index(int(times[index]))
        spread_ticks = max(1, int(round((ask - bid) / tick)))
        for side_i, side in enumerate(gp.SIDES):
            for action_i, action in enumerate(gp.QUOTE_ACTIONS):
                if action == "improve" and spread_ticks <= 1:
                    continue
                price = (bid + (tick if action == "improve" else 0.0)
                         if side == "buy" else
                         ask - (tick if action == "improve" else 0.0))
                sign = 1 if side == "buy" else -1
                for mode_i, mode in enumerate(MODES):
                    group = (mode_i, side_i, action_i, bucket, state)
                    result["attempts"][group] += 1
                    fill_time, _ = gp._first_fill(
                        day, trades, index, expiry, price, side, mode, tick,
                        config.latency_ms, times,
                    )
                    if fill_time is None:
                        continue
                    result["fills"][group] += 1
                    fill_snapshot = int(np.searchsorted(times, fill_time, side="right") - 1)
                    if fill_snapshot < 0 or segments[fill_snapshot] != segments[index]:
                        continue
                    # A sampled midpoint more than one second old is poor
                    # evidence for the post-fill price at a trade timestamp.
                    if int(fill_time) - int(times[fill_snapshot]) > 1000:
                        continue
                    mid_asof = float(mids[fill_snapshot])
                    target_time = int(fill_time) + 3000
                    at_3s = int(np.searchsorted(times, target_time, side="left"))
                    if (at_3s < len(times)
                            and segments[at_3s] == segments[index]
                            and int(times[at_3s]) - target_time <= 1000):
                        gross = sign * gp.LOT_SIZE * (float(mids[at_3s]) - price)
                        post = sign * gp.LOT_SIZE * (float(mids[at_3s]) - mid_asof)
                        result["count_3s"][group] += 1
                        result["sum_3s"][group] += gross
                        result["sum_sq_3s"][group] += gross * gross
                        result["post_sum_3s"][group] += post
                    at_20 = fill_snapshot + config.quote_horizon_snapshots
                    if at_20 < len(times) and segments[at_20] == segments[index]:
                        gross = sign * gp.LOT_SIZE * (float(mids[at_20]) - price)
                        post = sign * gp.LOT_SIZE * (float(mids[at_20]) - mid_asof)
                        result["count_20snap"][group] += 1
                        result["sum_20snap"][group] += gross
                        result["sum_sq_20snap"][group] += gross * gross
                        result["post_sum_20snap"][group] += post
                        result["duration_sum_20snap"][group] += (
                            int(times[at_20]) - int(fill_time)
                        ) / 1000.0
    return result


def pooled(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: sum((r[key] for r in records), np.zeros(SHAPE, dtype=np.float64))
            for key in fresh_probe()}


def gate_from_prior(stats: dict[str, np.ndarray], fee_hkd: float,
                    confidence_z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shrink bucket/spread estimates to historical side/action parents."""
    count = stats["count_3s"]
    total = stats["sum_3s"]
    squared = stats["sum_sq_3s"]
    allowed = np.zeros(SHAPE, dtype=bool)
    estimate = np.full(SHAPE, np.nan, dtype=np.float64)
    support = np.zeros(SHAPE, dtype=np.float64)
    for mode in range(SHAPE[0]):
        for side in range(SHAPE[1]):
            for action in range(SHAPE[2]):
                n_global = float(count[mode, side, action].sum())
                if n_global < 30:
                    continue
                global_mean = float(total[mode, side, action].sum() / n_global)
                second = float(squared[mode, side, action].sum() / n_global)
                global_sd = float(np.sqrt(max(0.0, second - global_mean**2)))
                for spread in range(SHAPE[4]):
                    n_spread = float(count[mode, side, action, :, spread].sum())
                    sum_spread = float(total[mode, side, action, :, spread].sum())
                    spread_mean = (sum_spread + 100.0 * global_mean) / (n_spread + 100.0)
                    for bucket in range(SHAPE[3]):
                        key = (mode, side, action, bucket, spread)
                        n_cell = float(count[key])
                        mean = (float(total[key]) + 30.0 * spread_mean) / (n_cell + 30.0)
                        # Global empirical dispersion prevents a tiny bucket
                        # from appearing certain due to accidental low variance.
                        effective_n = max(1.0, min(n_global, n_cell + 30.0))
                        lower = mean - confidence_z * global_sd / np.sqrt(effective_n)
                        allowed[key] = lower > fee_hkd
                        estimate[key] = mean
                        support[key] = n_cell
    return allowed, estimate, support


def masked_inputs(inputs: gp.GPInputs, mode: str, bucket: int,
                  allowed: np.ndarray) -> gp.GPInputs:
    hazards = {key: value.copy() for key, value in inputs.execution_intensity_per_second.items()}
    mode_i = MODES.index(mode)
    # GP array is side x action x spread.  The historical gate additionally
    # uses the session bucket, fixed for each separately solved policy.
    hazards[mode][~allowed[mode_i, :, :, bucket, :]] = 0.0
    return replace(inputs, execution_intensity_per_second=hazards)


def summarize_probe(stats: dict[str, np.ndarray]) -> dict[str, dict]:
    result = {}
    for mode_i, mode in enumerate(MODES):
        result[mode] = {}
        for side_i, side in enumerate(gp.SIDES):
            for action_i, action in enumerate(gp.QUOTE_ACTIONS):
                key = f"{side}_{action}"
                n3 = int(stats["count_3s"][mode_i, side_i, action_i].sum())
                n20 = int(stats["count_20snap"][mode_i, side_i, action_i].sum())
                result[mode][key] = {
                    "attempts": int(stats["attempts"][mode_i, side_i, action_i].sum()),
                    "fills": int(stats["fills"][mode_i, side_i, action_i].sum()),
                    "valid_3s": n3,
                    "gross_3s_hkd_per_fill": (
                        float(stats["sum_3s"][mode_i, side_i, action_i].sum() / n3)
                        if n3 else None
                    ),
                    "post_fill_3s_hkd_per_fill": (
                        float(stats["post_sum_3s"][mode_i, side_i, action_i].sum() / n3)
                        if n3 else None
                    ),
                    "valid_20snap": n20,
                    "gross_20snap_hkd_per_fill": (
                        float(stats["sum_20snap"][mode_i, side_i, action_i].sum() / n20)
                        if n20 else None
                    ),
                    "post_fill_20snap_hkd_per_fill": (
                        float(stats["post_sum_20snap"][mode_i, side_i, action_i].sum() / n20)
                        if n20 else None
                    ),
                    "mean_20snap_seconds": (
                        float(stats["duration_sum_20snap"][mode_i, side_i, action_i].sum() / n20)
                        if n20 else None
                    ),
                }
    return result


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prior-days", type=int, default=5)
    parser.add_argument("--gamma", type=float, nargs="+", default=[5.0, 0.1])
    parser.add_argument("--confidence-z", type=float, default=1.0)
    parser.add_argument("--limit-test-days", type=int, default=None)
    args = parser.parse_args()
    if args.prior_days < 1 or args.confidence_z < 0 or any(g < 0 for g in args.gamma):
        parser.error("prior-days must be positive, gamma and confidence-z nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source, dates, days, trades = _load_inputs()
    if len(dates) != 20 or not all(date.startswith("2026-07") for date in dates):
        raise ValueError("expected the 20 clean July 2026 trading dates")
    if args.limit_test_days is not None:
        dates = dates[: 1 + args.limit_test_days]
    for day in days.values():
        day.features = None
    standard_config = gp.GPConfig()
    prior_stats = {}
    prior_probe = {}
    first = dates[0]
    prior_stats[first] = gp.collect_daily_calibration_stats(days[first], trades[first], standard_config)
    prior_probe[first] = probe_completed_day(
        days[first], trades[first], prior_stats[first].tick_hkd, standard_config
    )
    print("CALIBRATED", first, flush=True)

    rows = []
    conditional_rows = []
    gate_records = []
    for i, date in enumerate(dates[1:], start=1):
        prior = dates[max(0, i - args.prior_days):i]
        inputs = gp.aggregate_calibration_stats([prior_stats[d] for d in prior], standard_config)
        historic = pooled([prior_probe[d] for d in prior])
        fee = inputs.calibration_reference_price_hkd * gp.LOT_SIZE * gp.ALL_IN_FEE_RATE
        allowed, estimated, support = gate_from_prior(historic, fee, args.confidence_z)
        # Retain the original replay tick convention for exact baseline
        # reconciliation.  It infers a constant tick from the complete test
        # day and is called out as an inherited causality limitation below.
        test_tick = gp.infer_tick_hkd(days[date])
        for mode_i, mode in enumerate(MODES):
            for side_i, side in enumerate(gp.SIDES):
                for action_i, action in enumerate(gp.QUOTE_ACTIONS):
                    for bucket in range(len(gp.SESSION_BUCKETS)):
                        for spread in range(standard_config.spread_states):
                            key = (mode_i, side_i, action_i, bucket, spread)
                            n3 = int(historic["count_3s"][key])
                            n20 = int(historic["count_20snap"][key])
                            conditional_rows.append({
                                "date": date, "prior_dates": "|".join(prior),
                                "fill_mode": mode, "side": side, "action": action,
                                "bucket": bucket,
                                "bucket_label": gp.SESSION_BUCKETS[bucket][2],
                                "spread_state_ticks": spread + 1,
                                "prior_probe_attempts": int(historic["attempts"][key]),
                                "prior_probe_fills": int(historic["fills"][key]),
                                "prior_valid_3s": n3,
                                "prior_gross_3s_hkd_per_fill": (
                                    float(historic["sum_3s"][key] / n3) if n3 else ""
                                ),
                                "prior_post_fill_3s_hkd_per_fill": (
                                    float(historic["post_sum_3s"][key] / n3) if n3 else ""
                                ),
                                "prior_valid_20snap": n20,
                                "prior_gross_20snap_hkd_per_fill": (
                                    float(historic["sum_20snap"][key] / n20) if n20 else ""
                                ),
                                "prior_post_fill_20snap_hkd_per_fill": (
                                    float(historic["post_sum_20snap"][key] / n20) if n20 else ""
                                ),
                                "shrunken_gross_3s_hkd_per_fill": (
                                    float(estimated[key]) if np.isfinite(estimated[key]) else ""
                                ),
                                "maker_fee_hkd": fee,
                                "allowed": int(allowed[key]),
                            })
        for gamma in args.gamma:
            config = replace(standard_config, inventory_penalty_gamma=float(gamma))
            for mode in MODES:
                active_hazards = np.stack(
                    [inputs.execution_intensity_per_second[mode] > 0]
                    * len(gp.SESSION_BUCKETS), axis=2,
                )
                mode_allowed = allowed[MODES.index(mode)]
                base_policies = {
                    b: gp.solve_gp_policy(inputs, test_tick, mode, b, config)
                    for b in range(len(gp.SESSION_BUCKETS))
                }
                gated_policies = {
                    b: gp.solve_gp_policy(
                        masked_inputs(inputs, mode, b, allowed),
                        test_tick, mode, b, config,
                    ) for b in range(len(gp.SESSION_BUCKETS))
                }
                base = gp.simulate_gp_day(days[date], trades[date], base_policies,
                                          config, mode, "original_gp")
                gated = gp.simulate_gp_day(days[date], trades[date], gated_policies,
                                           config, mode, "fill_conditioned_gate")
                rows.append({
                    "date": date, "gamma": gamma, "fill_mode": mode,
                    "prior_dates": "|".join(prior),
                    "tick_hkd": test_tick,
                    "baseline_net_hkd": base["net_pnl_hkd"],
                    "gated_net_hkd": gated["net_pnl_hkd"],
                    "delta_hkd": gated["net_pnl_hkd"] - base["net_pnl_hkd"],
                    "baseline_maker_fills": base["maker_fills"],
                    "gated_maker_fills": gated["maker_fills"],
                    "baseline_quoted_bid": base["quoted_bid"],
                    "gated_quoted_bid": gated["quoted_bid"],
                    "baseline_quoted_ask": base["quoted_ask"],
                    "gated_quoted_ask": gated["quoted_ask"],
                    "baseline_market_lots": base["market_order_lots"],
                    "gated_market_lots": gated["market_order_lots"],
                    "allowed_cells": int(allowed[MODES.index(mode)].sum()),
                    "total_cells": int(allowed[MODES.index(mode)].size),
                    "original_positive_hazard_cells": int(active_hazards.sum()),
                    "hazard_cells_set_to_zero": int(np.count_nonzero(active_hazards & ~mode_allowed)),
                    "fee_hkd_per_maker_fill": fee,
                })
        gate_records.append({
            "date": date, "prior_dates": prior,
            "inherited_full_day_tick_hkd": test_tick,
            "maker_fee_hkd": fee,
            "allowed": allowed.tolist(),
            "estimated_3s_gross_markout_hkd": np.where(np.isfinite(estimated), estimated, 0.0).tolist(),
            "cell_3s_sample_count": support.tolist(),
        })
        print("REPLAY", date, "allowed through/touch",
              int(allowed[0].sum()), int(allowed[1].sum()), flush=True)
        # The current day's outcomes enter the calibration window only after
        # every policy variant for this day has finished replaying.
        prior_stats[date] = gp.collect_daily_calibration_stats(days[date], trades[date], standard_config)
        prior_probe[date] = probe_completed_day(
            days[date], trades[date], prior_stats[date].tick_hkd, standard_config
        )

    with (args.output_dir / "daily.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "conditional_markouts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(conditional_rows[0]))
        writer.writeheader()
        writer.writerows(conditional_rows)
    summaries = []
    for gamma in args.gamma:
        for mode in MODES:
            group = [r for r in rows if r["gamma"] == gamma and r["fill_mode"] == mode]
            for label, selected in (
                ("early", [r for r in group if r["date"] <= "2026-07-17"]),
                ("late", [r for r in group if r["date"] >= "2026-07-20"]),
                ("month", group),
            ):
                baseline_net = sum(r["baseline_net_hkd"] for r in selected)
                gated_net = sum(r["gated_net_hkd"] for r in selected)
                zeroed = sum(r["hazard_cells_set_to_zero"] for r in selected)
                active = sum(r["original_positive_hazard_cells"] for r in selected)
                summaries.append({
                    "gamma": gamma, "fill_mode": mode, "period": label,
                    "test_days": len(selected),
                    "first_date": selected[0]["date"],
                    "last_date": selected[-1]["date"],
                    "baseline_net_hkd": baseline_net,
                    "gated_net_hkd": gated_net,
                    "no_trade_net_hkd": 0.0,
                    "baseline_vs_no_trade_hkd": baseline_net,
                    "gated_vs_no_trade_hkd": gated_net,
                    "delta_hkd": sum(r["delta_hkd"] for r in selected),
                    "baseline_maker_fills": sum(r["baseline_maker_fills"] for r in selected),
                    "gated_maker_fills": sum(r["gated_maker_fills"] for r in selected),
                    "hazard_cells_set_to_zero": zeroed,
                    "original_positive_hazard_cells": active,
                    "hazard_cells_set_to_zero_share": zeroed / active if active else None,
                })
    original_reference = json.loads((HERE / "output/gamma_results.json").read_text())
    reference_gamma5 = {
        test["test_date"]: test["results"]
        for test in original_reference["variants"]["gamma_5"]["tests"]
    }
    gamma5_baseline_matches = {
        f"{row['date']}_{row['fill_mode']}": bool(
            np.isclose(
                row["baseline_net_hkd"],
                reference_gamma5[row["date"]][row["fill_mode"]]["net_pnl_hkd"],
                rtol=0, atol=1e-6,
            )
            and row["baseline_maker_fills"]
            == reference_gamma5[row["date"]][row["fill_mode"]]["maker_fills"]
        )
        for row in rows if row["gamma"] == 5.0
    }
    if not all(gamma5_baseline_matches.values()):
        raise AssertionError("gamma=5 original daily baseline reconciliation failed")
    output = {
        "design": {
            "dates": dates, "rolling_prior_days": args.prior_days,
            "gammas": args.gamma, "confidence_z": args.confidence_z,
            "markout_3s": "signed 100-share (first L1 midpoint at/after fill+3 seconds minus maker fill price); midpoint at most 1 second late; fill-asof midpoint at most 1 second old; same segment",
            "markout_20snap": "signed 100-share (midpoint at 20 snapshots after asof-fill snapshot minus maker fill price); same segment",
            "gate": "allow side/action/spread/session bucket if hierarchical historical 3s gross markout minus one maker fee exceeds z times pooled standard error; otherwise set GP execution hazard to zero and re-solve QVI",
            "candidate_fill": "identical _first_fill proxy as GP replay; all available non-overlapping quote starts, not conditioned on prior GP selection",
            "source_clean_dates": source["source_audit"]["clean_trading_dates"],
        },
        "summary": summaries,
        "gate_by_day": gate_records,
        "probes_by_day": {date: summarize_probe(prior_probe[date]) for date in dates},
        "validation": {
            "strict_prior_only": all(all(p < r["date"] for p in r["prior_dates"].split("|")) for r in rows),
            "gamma5_baseline_matches_original_by_day_and_mode": gamma5_baseline_matches,
            "gamma5_baseline_matches_original_all": all(gamma5_baseline_matches.values()),
            "legacy_full_day_tick_is_preserved_for_baseline_comparison": True,
        },
        "limitations": [
            "The same July month is used for this exploratory evaluation; no independent later holdout is available.",
            "The touch/through rules are proxy fills; queue position and realized execution are unknown.",
            "A 3-second midpoint markout is a selection statistic, not realized strategy PnL, and omits subsequent liquidation costs; replay PnL includes those costs.",
            "The historical first-fill probes condition on future prices of completed calibration days only.",
            "The inherited GP replay infers each test day's constant tick from its full-day price grid; exact exchange tick table is needed to remove this separate timing limitation.",
        ],
    }
    write_json(args.output_dir / "results.json", output)
    print("SAVED", args.output_dir / "results.json", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
