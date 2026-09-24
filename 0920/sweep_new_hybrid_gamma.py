#!/usr/bin/env python3
"""Sweep GP/QVI inventory penalties for the July new-hybrid replay.

The 5 and 3.125 anchors come from the reconciled month replay. New penalties
use the same prior-only calibration, quote starts, fill proxies, and fees.
Each date/penalty is saved separately so a long sweep can resume safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import run_new_hybrid_month as month

gp = month.gp
sweep = month.sweep

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "output/new_hybrid_gamma_sweep"
GAMMAS = (5.0, 3.125, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)
TREATMENTS = ("carry", "forced_flat")
MEASURES = (
    "net_pnl_hkd", "gross_pnl_hkd", "all_in_fees_hkd", "maker_fills",
    "market_order_lots", "quoted_bid", "quoted_ask", "max_abs_inventory_lots",
)


def load_anchor(date: str, gamma: float, available: bool, count: int,
                references: dict[str, dict]) -> list[dict]:
    rows = []
    for treatment in TREATMENTS:
        for mode in gp.FILL_MODES:
            reference = next(row for row in references[treatment]["daily"]
                             if row["date"] == date and row["family"] == "GP"
                             and row["gamma"] == gamma and row["fill_mode"] == mode)
            if (reference["quote_decisions"] != count
                    or reference["l3_available"] != available):
                raise AssertionError(f"anchor metadata differs on {date}")
            row = {"date": date, "gamma": gamma, "fill_mode": mode,
                   "gap_inventory": treatment, "l3_available": available,
                   "quote_decisions": count, "source": "reconciled_month_anchor"}
            for prefix, original in (("baseline", "baseline"),
                                     ("new_hybrid", "new_hybrid")):
                for key, ref_key in (("net_pnl_hkd", "net_hkd"),
                                     ("all_in_fees_hkd", "all_in_fees_hkd"),
                                     ("maker_fills", "maker_fills")):
                    row[f"{prefix}_{key}"] = reference[f"{original}_{ref_key}"]
                row[f"{prefix}_gross_pnl_hkd"] = (
                    row[f"{prefix}_net_pnl_hkd"] + row[f"{prefix}_all_in_fees_hkd"]
                )
            row["new_minus_baseline_hkd"] = (
                row["new_hybrid_net_pnl_hkd"] - row["baseline_net_pnl_hkd"]
            )
            rows.append(row)
    return rows


def result_row(date: str, gamma: float, mode: str, treatment: str,
               available: bool, count: int, baseline: dict, new: dict) -> dict:
    for item in (baseline, new):
        month.checked(item)
        if item["quote_decisions"] != count:
            raise AssertionError(f"quote decisions differ on {date}")
    row = {"date": date, "gamma": gamma, "fill_mode": mode,
           "gap_inventory": treatment, "l3_available": available,
           "quote_decisions": count, "source": "new_replay"}
    for prefix, result in (("baseline", baseline), ("new_hybrid", new)):
        row.update({f"{prefix}_{key}": result[key] for key in MEASURES})
    row["new_minus_baseline_hkd"] = (
        new["net_pnl_hkd"] - baseline["net_pnl_hkd"]
    )
    return row


def summarize(rows: list[dict], gamma: float, mode: str, treatment: str,
              subset: str) -> dict:
    selected = [r for r in rows if r["gamma"] == gamma and r["fill_mode"] == mode
                and r["gap_inventory"] == treatment
                and (subset == "all" or r["l3_available"] == (subset == "l3_available"))]
    value = {"gamma": gamma, "fill_mode": mode, "gap_inventory": treatment,
             "subset": subset, "days": len(selected),
             "l3_days": sum(r["l3_available"] for r in selected),
             "quote_decisions": sum(r["quote_decisions"] for r in selected),
             "positive_new_days": sum(r["new_hybrid_net_pnl_hkd"] > 0 for r in selected),
             "new_better_than_baseline_days": sum(r["new_minus_baseline_hkd"] > 1e-8
                                                   for r in selected)}
    for prefix in ("baseline", "new_hybrid"):
        for field in ("net_pnl_hkd", "gross_pnl_hkd", "all_in_fees_hkd", "maker_fills"):
            value[f"{prefix}_{field}"] = sum(r[f"{prefix}_{field}"] for r in selected)
        # Anchor runs did not record these secondary diagnostics in their rows.
        for field in ("market_order_lots", "quoted_bid", "quoted_ask"):
            key = f"{prefix}_{field}"
            value[key] = (sum(r[key] for r in selected) if all(key in r for r in selected)
                          else None)
    value["new_minus_baseline_hkd"] = sum(r["new_minus_baseline_hkd"] for r in selected)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-dir", type=Path, default=month.DEFAULT_SIGNAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gammas", type=float, nargs="+", default=GAMMAS)
    parser.add_argument("--limit-test-days", type=int)
    parser.add_argument("--date-min", type=str, help="first test date to compute (inclusive)")
    parser.add_argument("--date-max", type=str, help="last test date to compute (inclusive)")
    parser.add_argument("--partial-only", action="store_true",
                        help="save per-day files without writing an incomplete month summary")
    args = parser.parse_args()
    if (not args.gammas or any(not math.isfinite(g) or g <= 0 for g in args.gammas)
            or (args.limit_test_days is not None and args.limit_test_days < 1)):
        parser.error("positive finite gammas and positive test-day limit required")
    gammas = sorted(set(args.gammas), reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_day = args.output_dir / "per_day"
    per_day.mkdir(exist_ok=True)
    manifest = json.loads(month.MANIFEST.read_text())
    dates = manifest["date_order"][:1 + args.limit_test_days] if args.limit_test_days else manifest["date_order"]
    base = gp.GPConfig(max_inventory_lots=month.GP_CAP)
    references = {
        "carry": json.loads((HERE / "output/new_hybrid_month/results.json").read_text()),
        "forced_flat": json.loads((HERE / "output/new_hybrid_month_forced_flat/results.json").read_text()),
    }
    if any(ref["test_dates"] != manifest["date_order"][1:] for ref in references.values()):
        raise AssertionError("anchor test dates differ")

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
        stats[date] = gp.collect_daily_calibration_stats(day, trades[date], base)
        print("CALIBRATION", date, flush=True)

    rows = []
    for ordinal, date in enumerate(dates[1:], start=1):
        if ((args.date_min is not None and date < args.date_min)
                or (args.date_max is not None and date > args.date_max)):
            continue
        with np.load(args.signal_dir / "predictions" / f"day_{date}.npz",
                     allow_pickle=False) as archive:
            prior_dates = [str(x) for x in archive["prior_dates"]]
            clean_prior = [d for d in dates[:ordinal]
                           if not days[d].metadata.get("gap_events")
                           and not any((days[d].metadata.get("order_errors") or {}).values())][-5:]
            if prior_dates != clean_prior:
                raise AssertionError(f"prior dates differ on {date}: {prior_dates} vs {clean_prior}")
            indices = gp.select_quote_indices(days[date], days[date].eligible,
                                               base.quote_horizon_snapshots)
            if not np.array_equal(indices, archive["quote_indices"]):
                raise AssertionError(f"quote starts differ on {date}")
            available = bool(archive["l3_available"])
            expected_available = (not days[date].metadata.get("gap_events")
                                  and not any((days[date].metadata.get("order_errors") or {}).values()))
            if available != expected_available:
                raise AssertionError(f"L3 availability differs on {date}")
            prediction = archive["expected_move_ticks"].copy()
            book_prediction = archive["book_only_expected_move_ticks"].copy()
            durations = archive["causal_horizon_seconds"].copy()
            if np.any(durations <= 0) or not np.isfinite(prediction).all():
                raise AssertionError(f"invalid forecast on {date}")
            if not available and not np.array_equal(prediction, book_prediction):
                raise AssertionError(f"nonexact L2 fallback on {date}")
            if available:
                regime, _ = sweep.prior_regime(archive, prior_dates, days, base, 5)
            else:
                prior_archive = {key: archive[key] for key in archive.files}
                prior_archive["prior_expected_move_ticks"] = archive[
                    "prior_book_only_expected_move_ticks"
                ].copy()
                regime, _ = sweep.prior_regime(prior_archive, prior_dates, days, base, 5)
        inputs = gp.aggregate_calibration_stats([stats[d] for d in prior_dates], base)
        tick = gp.infer_tick_hkd(days[date])
        states = gp.assign_gp_drift_states(prediction * tick / durations, regime)
        for gamma in gammas:
            file = per_day / f"{date}_gamma_{gamma:g}.json"
            if file.exists():
                saved = json.loads(file.read_text())
                if saved["date"] != date or saved["gamma"] != gamma:
                    raise AssertionError(f"stale per-day file {file}")
                day_rows = saved["rows"]
            elif gamma in (5.0, 3.125):
                day_rows = load_anchor(date, gamma, available, len(indices), references)
                file.write_text(json.dumps({"date": date, "gamma": gamma, "rows": day_rows},
                                           allow_nan=False, indent=2) + "\n")
            else:
                config = replace(base, inventory_penalty_gamma=gamma)
                day_rows = []
                for mode in gp.FILL_MODES:
                    baseline_policy = month.gp_policy_set(inputs, tick, mode, config, None)
                    new_policy = month.gp_policy_set(inputs, tick, mode, config, regime)
                    for treatment in TREATMENTS:
                        forced = treatment == "forced_flat"
                        baseline = gp.simulate_gp_day(
                            days[date], trades[date], baseline_policy, config, mode,
                            "martingale_gp", flatten_on_segment_change=forced,
                        )
                        new = gp.simulate_gp_day(
                            days[date], trades[date], new_policy, config, mode,
                            "new_hybrid_gp", states, flatten_on_segment_change=forced,
                        )
                        day_rows.append(result_row(
                            date, gamma, mode, treatment, available, len(indices), baseline, new
                        ))
                file.write_text(json.dumps({"date": date, "gamma": gamma, "rows": day_rows},
                                           allow_nan=False, indent=2) + "\n")
            if len(day_rows) != len(TREATMENTS) * len(gp.FILL_MODES):
                raise AssertionError(f"incomplete result on {date} gamma {gamma}")
            rows.extend(day_rows)
            print("REPLAY", date, "gamma", gamma, flush=True)

    if args.partial_only:
        print("PARTIAL_COMPLETE", len(rows), flush=True)
        return

    summary = [summarize(rows, gamma, mode, treatment, subset)
               for gamma in gammas for mode in gp.FILL_MODES
               for treatment in TREATMENTS
               for subset in ("all", "l3_available", "l3_unavailable")]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "daily.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "name": "new hybrid GP gamma sweep", "dates": dates, "test_dates": dates[1:],
        "gammas": gammas, "gap_policy": references["carry"]["gap_policy"],
        "signal_result": str((args.signal_dir / "results.json").resolve()),
        "anchor_results": {key: str((HERE / "output" /
                             ("new_hybrid_month" if key == "carry"
                              else "new_hybrid_month_forced_flat") / "results.json").resolve())
                           for key in TREATMENTS},
        "source_note": "5 and 3.125 from reconciled month replay; all other gamma values replayed here",
        "fee_and_execution": references["carry"]["fee_and_execution"],
        "test_labels_not_used_in_current_day_fit": True,
        "summary": summary, "daily": rows,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    print(json.dumps([s for s in summary if s["subset"] == "all"
                      and s["gap_inventory"] == "carry"],
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
