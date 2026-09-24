#!/usr/bin/env python3
"""Reconcile GP replay losses to execution edge, quote repricing, carry and fees.

Diagnostics observe the original replay without changing its actions or cashflows.
Markouts use the first sampled midpoint at/after each horizon, within the same
segment and at most one second late. They are not additional PnL components.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0906"))
sys.path.append(str(ROOT / "0824"))
import gp_qvi_model as gp
from paper_gp_replay import simulate_gp_day
from run_gp_qvi_experiment import _load_inputs

OUT = HERE / "output/loss_diagnostics"
GAMMAS = (5.0, 0.1, 0.01, 0.001, 0.0)
HORIZONS = (1, 3, 10, 30)


def near(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-5)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(day, inputs, config, mode, result, events, decisions):
    times = gp.compact_time_to_day_ms(day.send_times)
    mids = (day.book[:, 0].astype(float) + day.book[:, 2].astype(float)) / 2
    parts = dict(maker_quote_edge_hkd=0.0, maker_quote_to_fill_hkd=0.0,
                 market_execution_edge_hkd=0.0, terminal_execution_edge_hkd=0.0,
                 inventory_carry_hkd=0.0)
    signed_inventory, last_mid = 0, None
    ledger_cash, ledger_fees = 0.0, 0.0
    markouts = []
    for event in events:
        assert event["inventory_before_lots"] == signed_inventory
        side, price, mid = event["signed_lots"], event["price_hkd"], event["mid_asof_hkd"]
        if last_mid is not None:
            parts["inventory_carry_hkd"] += signed_inventory * gp.LOT_SIZE * (mid - last_mid)
        edge = side * gp.LOT_SIZE * (mid - price)
        if event["kind"] == "maker":
            parts["maker_quote_edge_hkd"] += side * gp.LOT_SIZE * (event["quote_mid_hkd"] - price)
            parts["maker_quote_to_fill_hkd"] += side * gp.LOT_SIZE * (mid - event["quote_mid_hkd"])
            index = event["quote_index"]
            event["quote_to_fill_seconds"] = (event["time_ms"] - int(times[index]) - config.latency_ms) / 1000
            for horizon in HORIZONS:
                target_time = event["time_ms"] + 1000 * horizon
                target = int(np.searchsorted(times, target_time, side="left"))
                if (target >= len(times) or int(day.segments[target]) != event["segment"]
                        or int(times[target]) - target_time > 1000 or event["snapshot_age_ms"] > 1000):
                    continue
                markouts.append({"horizon_seconds": horizon,
                    "gross_markout_hkd": side * gp.LOT_SIZE * (mids[target] - price),
                    "net_markout_hkd": side * gp.LOT_SIZE * (mids[target] - price) - event["fee_hkd"],
                    "post_fill_move_hkd": side * gp.LOT_SIZE * (mids[target] - mid)})
        else:
            parts[f"{event['kind']}_execution_edge_hkd"] += edge
            event["quote_to_fill_seconds"] = None
        signed_inventory += side
        ledger_cash -= side * gp.LOT_SIZE * price
        ledger_fees += event["fee_hkd"]
        last_mid = mid
    assert signed_inventory == 0
    assert near(ledger_cash, result["gross_pnl_hkd"])
    assert near(ledger_fees, result["all_in_fees_hkd"])
    assert near(sum(parts.values()), result["gross_pnl_hkd"]), (parts, result)
    maker = [e for e in events if e["kind"] == "maker"]
    probabilities = dict(order_count=0, actual_fills=0, expected_fills_dt3=0.0,
                         expected_fills_lifetime=0.0, both_quoted=0, both_filled=0,
                         expected_both_lifetime=0.0, positive_edge_orders=0)
    lifecycle = [r["lifetime_seconds"] for r in decisions]
    calibration_fee = inputs.calibration_reference_price_hkd * gp.LOT_SIZE * gp.ALL_IN_FEE_RATE
    for decision in decisions:
        ps = []
        for side, name in enumerate(("bid", "ask")):
            action = decision[f"{name}_action"]
            if action == 0:
                ps.append(0.0)
                continue
            state = decision["spread_state"]
            intensity = inputs.execution_intensity_per_second[mode][side, action - 1, state]
            probability = -np.expm1(-intensity * decision["lifetime_seconds"])
            ps.append(probability)
            probabilities["order_count"] += 1
            probabilities["actual_fills"] += int(decision["buy_filled" if side == 0 else "sell_filled"])
            probabilities["expected_fills_dt3"] += -np.expm1(-intensity * config.dt_seconds)
            probabilities["expected_fills_lifetime"] += probability
            half_spread = (state + 1) * result["tick_hkd"] / 2
            reward = (half_spread - (result["tick_hkd"] if action == gp.ACTION_IMPROVE else 0)) * gp.LOT_SIZE - calibration_fee
            probabilities["positive_edge_orders"] += int(reward > 0)
        if decision["bid_action"] and decision["ask_action"]:
            probabilities["both_quoted"] += 1
            probabilities["both_filled"] += int(decision["buy_filled"] and decision["sell_filled"])
            probabilities["expected_both_lifetime"] += ps[0] * ps[1]
    row = {"date": str(day.date), "gamma": config.inventory_penalty_gamma,
           "fill_mode": mode, **parts,
           **{key: result[key] for key in ("gross_pnl_hkd", "all_in_fees_hkd", "net_pnl_hkd",
               "net_pnl_official_only_hkd", "maker_fills", "policy_market_order_lots",
               "terminal_flatten_lots", "max_abs_inventory_lots", "exposure_duration_seconds",
               "squared_inventory_lot2_seconds")}, **probabilities,
           "maker_negative_edge_at_fill": sum(e["signed_lots"] * (e["mid_asof_hkd"] - e["price_hkd"]) < -1e-7 for e in maker),
           "maker_stale_mid_over_1s": sum(e["snapshot_age_ms"] > 1000 for e in maker),
           "maker_age_ms_sum": sum(e["snapshot_age_ms"] for e in maker),
           "lifecycle_median_seconds": float(np.median(lifecycle)),
           "lifecycle_p95_seconds": float(np.quantile(lifecycle, .95)),
           "lifecycle_under_3_seconds": sum(t < 3 for t in lifecycle),
           "quote_decisions": len(decisions), "decomposition_passed": True}
    groups = []
    for horizon in HORIZONS:
        samples = [r for r in markouts if r["horizon_seconds"] == horizon]
        groups.append({"date": str(day.date), "gamma": config.inventory_penalty_gamma,
                       "fill_mode": mode, "horizon_seconds": horizon, "count": len(samples),
                       **{key: sum(r[key] for r in samples) for key in
                          ("gross_markout_hkd", "net_markout_hkd", "post_fill_move_hkd")}})
    return row, groups


def run_day(task):
    date, day, trades, inputs, reference = task
    original_fill, cache = gp._first_fill, {}

    def cached_fill(day_, trades_, index, expiry, price, side, mode, tick, latency, snapshot_times_ms=None):
        key = (index, expiry, price, side, mode, tick, latency)
        if key not in cache:
            cache[key] = original_fill(day_, trades_, index, expiry, price, side, mode, tick, latency, snapshot_times_ms)
        return cache[key]

    gp._first_fill = cached_fill
    rows, markouts, all_events, checks = [], [], [], []
    try:
        for gamma in GAMMAS:
            config = replace(gp.GPConfig(), inventory_penalty_gamma=gamma)
            for mode in gp.FILL_MODES:
                tick = gp.infer_tick_hkd(day)
                policies = {b: gp.solve_gp_policy(inputs, tick, mode, b, config) for b in range(len(gp.SESSION_BUCKETS))}
                events, decisions = [], []
                result = simulate_gp_day(day, trades, policies, config, mode, "gp_qvi", events, decisions)
                old = reference[(gamma, mode)]
                for key in ("maker_fills", "buy_fills", "sell_fills", "quoted_bid", "quoted_ask",
                            "net_pnl_hkd", "gross_pnl_hkd", "all_in_fees_hkd", "market_order_lots",
                            "max_abs_inventory_lots", "squared_inventory_lot2_seconds"):
                    assert near(result[key], old[key]), (date, gamma, mode, key, result[key], old[key])
                if gamma == 5 and mode == "through":
                    assert result == simulate_gp_day(day, trades, policies, config, mode, "gp_qvi")
                row, marks = analyze(day, inputs, config, mode, result, events, decisions)
                rows.append(row)
                markouts.extend(marks)
                all_events.extend({"gamma": gamma, "fill_mode": mode, **e} for e in events)
                checks.append({"date": date, "gamma": gamma, "fill_mode": mode,
                               "original_results_match": True, "cashflow_decomposition_matches": True})
        with gzip.open(OUT / "events" / f"{date}.csv.gz", "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_events[0]))
            writer.writeheader()
            writer.writerows(all_events)
        return {"date": date, "daily": rows, "markouts": markouts, "checks": checks}
    finally:
        gp._first_fill = original_fill


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "events").mkdir(exist_ok=True)
    source, dates, days, trades = _load_inputs()
    reference = {}
    for filename in (HERE / "output/gamma_daily.csv", HERE / "output/gamma_0_001_grid/gamma_daily.csv"):
        with filename.open(encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                reference[(row["date"], float(row["gamma"]), row["fill_mode"])] = row
    config = gp.GPConfig()
    stats = {date: gp.collect_daily_calibration_stats(days[date], trades[date], config) for date in dates}
    tasks = []
    for i, date in enumerate(dates[1:], start=1):
        inputs = gp.aggregate_calibration_stats([stats[p] for p in dates[max(0, i - 5):i]], config)
        refs = {(g, m): reference[(date, g, m)] for g in GAMMAS for m in gp.FILL_MODES}
        tasks.append((date, days[date], trades[date], inputs, refs))
    daily, markouts, checks = [], [], []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for item in pool.map(run_day, tasks):
            daily.extend(item["daily"])
            markouts.extend(item["markouts"])
            checks.extend(item["checks"])
            print("FINISHED", item["date"], flush=True)
    reconstruction = {r["date"]: r for r in source["source_audit"]["reconstruction"]}
    for row in daily:
        record = reconstruction[row["date"]]
        row["source_flagged"] = bool(record.get("tainted") or record.get("order_errors") or record.get("gap_events"))
    sums = []
    excluded = {"date", "gamma", "fill_mode", "max_abs_inventory_lots", "lifecycle_median_seconds",
                "lifecycle_p95_seconds", "decomposition_passed", "source_flagged"}
    for g in GAMMAS:
        for m in gp.FILL_MODES:
            rows = [r for r in daily if r["gamma"] == g and r["fill_mode"] == m]
            totals = {key: sum(r[key] for r in rows) for key in rows[0] if key not in excluded}
            sums.append({"gamma": g, "fill_mode": m, **totals,
                         "test_days": len(rows), "profitable_days": sum(r["net_pnl_hkd"] > 0 for r in rows),
                         "max_abs_inventory_lots": max(r["max_abs_inventory_lots"] for r in rows),
                         "unflagged_test_days": sum(not r["source_flagged"] for r in rows),
                         "unflagged_net_pnl_hkd": sum(r["net_pnl_hkd"] for r in rows if not r["source_flagged"]),
                         "unflagged_gross_pnl_hkd": sum(r["gross_pnl_hkd"] for r in rows if not r["source_flagged"]),
                         "inventory_rms_lots": math.sqrt(totals["squared_inventory_lot2_seconds"] / totals["exposure_duration_seconds"])})
    mark_summary = []
    for g in GAMMAS:
        for m in gp.FILL_MODES:
            for h in HORIZONS:
                rows = [r for r in markouts if r["gamma"] == g and r["fill_mode"] == m and r["horizon_seconds"] == h]
                count = sum(r["count"] for r in rows)
                mark_summary.append({"gamma": g, "fill_mode": m, "horizon_seconds": h, "count": count,
                    **{key.replace("_hkd", "_per_fill_hkd"): sum(r[key] for r in rows) / count if count else None
                       for key in ("gross_markout_hkd", "net_markout_hkd", "post_fill_move_hkd")}})
    write_csv(OUT / "daily_decomposition.csv", daily)
    write_csv(OUT / "summary.csv", sums)
    write_csv(OUT / "markouts.csv", mark_summary)
    write_csv(OUT / "daily_markouts.csv", markouts)
    paths = [Path(__file__), HERE / "paper_gp_replay.py", ROOT / "0906/gp_qvi_model.py"]
    validation = {"all_passed": True, "runs": len(checks), "checks": checks,
                  "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  "caveats": ["Markouts use sampled L1 midpoints, not exact event-time exchange quotes.",
                              "Book evidence and trade evidence are proxy fills; queue position and real fill probability are unobserved.",
                              "Zero-fee cash PnL rescoring does not re-optimize the policy.",
                              "Unflagged test-day subset still may use flagged prior calibration days."]}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(sums, indent=2), flush=True)


if __name__ == "__main__":
    main()
