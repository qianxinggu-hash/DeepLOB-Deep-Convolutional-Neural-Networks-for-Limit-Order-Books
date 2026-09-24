#!/usr/bin/env python3
"""Repeat the 0913 gamma study with signed inventory made explicit.

Every gamma retains limit and inventory-reducing market controls. No other
parameter changes. The admissible inventory state is the symmetric interval
[-10, 10] lots, so maker sells may open or extend a short position.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0906"))
sys.path.append(str(ROOT / "0824"))
import gp_qvi_model as gp
from paper_gp_replay import simulate_gp_day
from run_gp_qvi_experiment import _load_inputs, _portable_audit_rows

ANCHORS = (5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.0)
PAPER_GRID = tuple(50.0 / 2**i for i in range(14))
DEFAULT_GAMMAS = tuple(sorted(set((*PAPER_GRID, *ANCHORS)), reverse=True))


def run_day(task: tuple) -> dict:
    date, day, trades, inputs, gammas = task
    original_fill = gp._first_fill
    cache = {}

    def cached_fill(day_, trades_, index, expiry, price, side, mode, tick, latency,
                    snapshot_times_ms=None):
        key = (index, expiry, price, side, mode, tick, latency)
        if key not in cache:
            cache[key] = original_fill(day_, trades_, index, expiry, price, side,
                                       mode, tick, latency, snapshot_times_ms)
        return cache[key]

    gp._first_fill = cached_fill
    output = {}
    tick = gp.infer_tick_hkd(day)
    try:
        for gamma in gammas:
            config = replace(gp.GPConfig(), inventory_penalty_gamma=gamma)
            outcomes, audits = {}, {}
            for mode in gp.FILL_MODES:
                policies = {bucket: gp.solve_gp_policy(
                    inputs, tick, mode, bucket, config, market_orders_enabled=True)
                    for bucket in range(len(gp.SESSION_BUCKETS))}
                outcomes[mode] = simulate_gp_day(day, trades, policies, config, mode, "gp_qvi")
                audits[mode] = {}
                for bucket, policy in policies.items():
                    targets = policy.impulse_target[-1].astype(np.int64) - config.max_inventory_lots
                    grid = np.arange(-config.max_inventory_lots, config.max_inventory_lots + 1)
                    audits[mode][str(bucket)] = {
                        **gp.policy_audit(policy),
                        "full_horizon_impulse_state_share": float(np.mean(targets != grid)),
                        "full_horizon_impulse_targets_lots_by_spread": targets.tolist(),
                        "full_horizon_bid_actions_by_spread": policy.make_bid_action[-1].tolist(),
                        "full_horizon_ask_actions_by_spread": policy.make_ask_action[-1].tolist(),
                    }
            output[f"gamma_{gamma:g}"] = {
                "test_date": date, "calibration_dates": list(inputs.calibration_dates),
                "tick_hkd": tick, "results": outcomes, "policy_audit": audits,
            }
    finally:
        gp._first_fill = original_fill
    return {"date": date, "variants": output, "calibration": inputs.audit}


SUM_KEYS = (
    "quote_decisions", "quoted_bid", "quoted_ask", "maker_fills", "buy_fills", "sell_fills",
    "trade_price_fills", "best_quote_fills", "bid_cancels", "ask_cancels", "expired_orders",
    "market_orders", "market_order_lots", "policy_market_orders", "policy_market_order_lots",
    "terminal_flatten_orders", "terminal_flatten_lots", "gross_pnl_hkd", "all_in_fees_hkd",
    "net_pnl_hkd", "maker_turnover_hkd", "market_turnover_hkd", "feeable_turnover_hkd",
    "exposure_duration_seconds", "absolute_inventory_lot_seconds",
    "squared_inventory_lot2_seconds", "realized_inventory_penalty_hkd",
    "negative_inventory_time_seconds", "positive_inventory_time_seconds",
    "short_inventory_lot_seconds", "long_inventory_lot_seconds")


def aggregate(tests: list[dict]) -> dict:
    output = {}
    for mode in gp.FILL_MODES:
        rows = [t["results"][mode] for t in tests]
        sums = {k: sum(r[k] for r in rows) for k in SUM_KEYS}
        net = np.asarray([r["net_pnl_hkd"] for r in rows])
        cumulative = np.concatenate([[0.0], np.cumsum(net)])
        duration = max(sums["exposure_duration_seconds"], 1e-9)
        sums.update({
            "test_days": len(rows), "maker_fills_per_day": sums["maker_fills"] / len(rows),
            "mean_daily_net_pnl_hkd": float(np.mean(net)),
            "daily_net_pnl_std_hkd": float(np.std(net, ddof=1)) if len(rows) > 1 else 0.0,
            "profitable_days": int(np.sum(net > 0)), "active_days": sum(r["maker_fills"] > 0 for r in rows),
            "maker_fill_rate": sums["maker_fills"] / max(sums["quoted_bid"] + sums["quoted_ask"], 1),
            "quoted_side_share": (sums["quoted_bid"] + sums["quoted_ask"]) / max(2 * sums["quote_decisions"], 1),
            "max_abs_inventory_lots": max(r["max_abs_inventory_lots"] for r in rows),
            "min_inventory_lots": min(r["min_inventory_lots"] for r in rows),
            "max_inventory_lots": max(r["max_inventory_lots"] for r in rows),
            "time_mean_abs_inventory_lots": sums["absolute_inventory_lot_seconds"] / duration,
            "time_rms_inventory_lots": math.sqrt(sums["squared_inventory_lot2_seconds"] / duration),
            "nonzero_inventory_time_share": sum(r["nonzero_inventory_time_share"] * r["exposure_duration_seconds"] for r in rows) / duration,
            "negative_inventory_time_share": sums["negative_inventory_time_seconds"] / duration,
            "positive_inventory_time_share": sums["positive_inventory_time_seconds"] / duration,
            "day_close_max_drawdown_hkd": float(np.max(np.maximum.accumulate(cumulative) - cumulative)),
            "worst_intraday_drawdown_hkd": max(r["max_drawdown_hkd"] for r in rows),
            "mean_full_horizon_impulse_state_share": float(np.mean([
                a["full_horizon_impulse_state_share"] for t in tests
                for a in t["policy_audit"][mode].values()])),
            "action_counts": {k: sum(r["action_counts"][k] for r in rows) for k in rows[0]["action_counts"]},
        })
        output[mode] = sums
    return output


def validate(tests: list[dict], totals: dict, config: gp.GPConfig) -> dict:
    rows = [r for t in tests for r in t["results"].values()]
    checks = {
        "strictly_prior_rolling_calibration": all(all(d < t["test_date"] for d in t["calibration_dates"]) for t in tests),
        "all_policies_enable_market_control": all(a["market_orders_enabled"] for t in tests for audits in t["policy_audit"].values() for a in audits.values()),
        "order_lifecycle_reconciles": all(r["quoted_bid"] == r["buy_fills"] + r["bid_cancels"] and r["quoted_ask"] == r["sell_fills"] + r["ask_cancels"] for r in rows),
        "fill_evidence_reconciles": all(r["maker_fills"] == r["trade_price_fills"] + r["best_quote_fills"] for r in rows),
        "all_days_end_flat": all(r["end_inventory_lots"] == 0 for r in rows),
        "signed_inventory_bounds_are_symmetric": all(
            -config.max_inventory_lots <= r["min_inventory_lots"] <= 0
            and 0 <= r["max_inventory_lots"] <= config.max_inventory_lots for r in rows),
        "negative_inventory_is_observed": any(
            r["min_inventory_lots"] < 0 and r["negative_inventory_time_seconds"] > 0 for r in rows),
        "positive_inventory_is_observed": any(
            r["max_inventory_lots"] > 0 and r["positive_inventory_time_seconds"] > 0 for r in rows),
        "fees_and_cash_reconcile": all(math.isclose(r["net_pnl_hkd"], r["gross_pnl_hkd"] - r["all_in_fees_hkd"], abs_tol=1e-6) and math.isclose(r["all_in_fees_hkd"], r["feeable_turnover_hkd"] * gp.ALL_IN_FEE_RATE, abs_tol=1e-6) for r in rows),
        "inventory_exposure_reconciles": all(
            math.isclose(sum(r["time_at_inventory_lots_seconds"].values()), r["exposure_duration_seconds"], abs_tol=1e-6)
            and math.isclose(sum(float(q)**2 * sec for q, sec in r["time_at_inventory_lots_seconds"].items()), r["squared_inventory_lot2_seconds"], abs_tol=1e-6)
            and math.isclose(sum(abs(float(q)) * sec for q, sec in r["time_at_inventory_lots_seconds"].items()), r["absolute_inventory_lot_seconds"], abs_tol=1e-6)
            and math.isclose(sum(sec for q, sec in r["time_at_inventory_lots_seconds"].items() if float(q) < 0), r["negative_inventory_time_seconds"], abs_tol=1e-6)
            and math.isclose(sum(sec for q, sec in r["time_at_inventory_lots_seconds"].items() if float(q) > 0), r["positive_inventory_time_seconds"], abs_tol=1e-6)
            and math.isclose(sum(-float(q) * sec for q, sec in r["time_at_inventory_lots_seconds"].items() if float(q) < 0), r["short_inventory_lot_seconds"], abs_tol=1e-6)
            and math.isclose(sum(float(q) * sec for q, sec in r["time_at_inventory_lots_seconds"].items() if float(q) > 0), r["long_inventory_lot_seconds"], abs_tol=1e-6)
            and r["squared_inventory_lot2_seconds"] <= config.max_inventory_lots**2 * r["exposure_duration_seconds"] + 1e-6
            and math.isclose(r["realized_inventory_penalty_hkd"], config.inventory_penalty_gamma * r["squared_inventory_lot2_seconds"], abs_tol=1e-6) for r in rows),
        "monthly_aggregate_reconciles": all(math.isclose(totals[m]["net_pnl_hkd"], sum(t["results"][m]["net_pnl_hkd"] for t in tests), abs_tol=1e-6) for m in gp.FILL_MODES),
    }
    checks["all_passed"] = all(checks.values())
    return checks


def reconcile(tests: list[dict], reference: dict) -> dict:
    keys = ("maker_fills", "quoted_bid", "quoted_ask", "market_order_lots", "max_abs_inventory_lots",
            "trade_price_fills", "best_quote_fills", "net_pnl_hkd", "gross_pnl_hkd", "all_in_fees_hkd")
    mismatches = []
    for t in tests:
        for m in gp.FILL_MODES:
            for k in keys:
                old, new = reference[t["test_date"]][m][k], t["results"][m][k]
                if not math.isclose(old, new, rel_tol=1e-10, abs_tol=1e-6):
                    mismatches.append({"date": t["test_date"], "mode": m, "metric": k, "old": old, "new": new})
    return {"all_passed": not mismatches, "mismatches": mismatches}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gammas", type=float, nargs="+", default=list(DEFAULT_GAMMAS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit-test-days", type=int)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    if any(not math.isfinite(g) or g < 0 for g in args.gammas) or args.workers < 1 or (args.limit_test_days is not None and args.limit_test_days < 1):
        parser.error("finite nonnegative gammas and positive workers/day count required")
    gammas = sorted(set([5.0, *args.gammas]), reverse=True)
    source, dates, days, trades = _load_inputs()
    base_config = gp.GPConfig()
    stats = {d: gp.collect_daily_calibration_stats(days[d], trades[d], base_config) for d in dates}
    test_dates = dates[1:][:args.limit_test_days]
    tasks = [(d, days[d], trades[d], gp.aggregate_calibration_stats(
        [stats[p] for p in dates[max(0, i - 5):i]], base_config), gammas)
        for i, d in enumerate(test_dates, start=1)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(run_day, tasks):
            daily.append(item)
            print("FINISHED", item["date"], " | ".join(
                f"{k}: {v['results']['through']['maker_fills']} fills" for k, v in item["variants"].items()), flush=True)
            (args.output_dir / "progress.json").write_text(json.dumps({"completed_dates": [r["date"] for r in daily], "planned_dates": test_dates}, indent=2))
    result = {"as_of": datetime.now().astimezone().isoformat(), "instrument": "HK 07709", "period": "2026-07",
        "design": {"method": "Guilbaud-Pham mean criterion U(x)=x, g(q)=q²; limit and inventory-reducing market QVI controls",
            "only_variable": "inventory_penalty_gamma", "gamma_units": "HKD per lot²-second",
            "inventory_domain_lots": [-base_config.max_inventory_lots, base_config.max_inventory_lots],
            "negative_inventory": "allowed; maker asks may open/extend shorts to the lower state bound",
            "paper_grid": "50/2^i, i=0..13, follows Table 6 halving structure; plus previous anchors",
            "rolling_prior_days": 5, "test_dates": test_dates, "initial_calibration_date": dates[0],
            "all_in_fee_bps_per_side": 2.77, "inventory_exposure_clock": "first quote+5ms to terminal expiry+5ms; includes lunch while inventory held",
            "exposure_warning": "whole-day diagnostic, not the value of the receding 300s Bellman objective",
            "selection": "sensitivity study; no gamma chosen using monthly PnL"},
        "provenance": {"model": "0906/gp_qvi_model.py", "model_sha256": hashlib.sha256((ROOT / "0906/gp_qvi_model.py").read_bytes()).hexdigest(),
            "diagnostic_replay": "0920/paper_gp_replay.py", "diagnostic_replay_sha256": hashlib.sha256((HERE / "paper_gp_replay.py").read_bytes()).hexdigest(),
            "baseline": "0906/output/july_2026_gp_qvi_new_fill_results.json",
            "comparison_baseline": "0913/output/gamma_results.json",
            "paper": "0906/papers/Guilbaud_Pham_2011_Optimal_High_Frequency_Trading_with_Limit_and_Market_Orders.pdf",
            "clean_trading_dates": dates, "reconstruction": _portable_audit_rows(source["source_audit"]["reconstruction"]),
            "book_caches": [str(days[d].path.relative_to(ROOT)) for d in dates],
            "trade_caches": [f"0824/data/july_2026/hk07709_{d}_trades.npz" for d in dates]},
        "daily_calibration": {r["date"]: r["calibration"] for r in daily}, "variants": {}}
    checks, summary_rows, daily_rows, surface_rows = {}, [], [], []
    for gamma in gammas:
        key = f"gamma_{gamma:g}"
        config = replace(base_config, inventory_penalty_gamma=gamma)
        tests = [r["variants"][key] for r in daily]
        totals = aggregate(tests)
        checks[key] = validate(tests, totals, config)
        result["variants"][key] = {"config": gp.config_asdict(config), "aggregate": totals, "tests": tests, "validation": checks[key]}
        for mode in gp.FILL_MODES:
            summary_rows.append({"variant": key, "gamma": gamma, "fill_mode": mode,
                **{k: v for k, v in totals[mode].items() if k != "action_counts"}})
        for t in tests:
            for mode, r in t["results"].items():
                daily_rows.append({"gamma": gamma, "calibration_dates": "|".join(t["calibration_dates"]),
                    **{k: v for k, v in r.items() if k not in ("execution_cost", "action_counts", "time_at_inventory_lots_seconds")}})
                for bucket, audit in t["policy_audit"][mode].items():
                    for spread, targets in enumerate(audit["full_horizon_impulse_targets_lots_by_spread"], start=1):
                        for qi, target in enumerate(targets):
                            surface_rows.append({"date": t["test_date"], "gamma": gamma, "fill_mode": mode,
                                "bucket": bucket, "spread_ticks": spread, "inventory_lots": qi - config.max_inventory_lots,
                                "impulse_target_lots": target, "bid_action": audit["full_horizon_bid_actions_by_spread"][spread - 1][qi],
                                "ask_action": audit["full_horizon_ask_actions_by_spread"][spread - 1][qi]})
    baseline = json.loads((ROOT / "0906/output/july_2026_gp_qvi_new_fill_results.json").read_text())
    reference = {t["test_date"]: t["strategies"]["gp_qvi"] for t in baseline["tests"]}
    checks["baseline_matches_0906"] = reconcile(result["variants"]["gamma_5"]["tests"], reference)
    comparison_path = ROOT / "0913/output/gamma_results.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text())
        for key, variant in result["variants"].items():
            if key in comparison["variants"]:
                reference = {
                    t["test_date"]: t["results"]
                    for t in comparison["variants"][key]["tests"]
                }
                checks[f"matches_0913_{key}"] = reconcile(variant["tests"], reference)
    checks["all_passed"] = all(c["all_passed"] for c in checks.values())
    result["validation"] = checks
    (args.output_dir / "gamma_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    (args.output_dir / "gamma_validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    write_csv(args.output_dir / "gamma_summary.csv", summary_rows)
    write_csv(args.output_dir / "gamma_daily.csv", daily_rows)
    write_csv(args.output_dir / "gamma_policy_surface.csv", surface_rows)
    print("WROTE", args.output_dir, "all_passed:", checks["all_passed"], flush=True)
    if not checks["all_passed"]:
        raise AssertionError(checks)


if __name__ == "__main__":
    main()
