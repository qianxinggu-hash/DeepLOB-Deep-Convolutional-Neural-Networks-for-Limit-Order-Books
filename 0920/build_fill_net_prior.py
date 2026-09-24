#!/usr/bin/env python3
"""Build causal fill-conditioned QVI reward priors on the two-minute gap caches.

The candidate probe is intentionally independent of the historical GP action:
at every eligible, non-overlapping quote start it tests both sides, both quote
prices and both execution proxies.  A candidate contributes a 3-second label
only when it fills before quote start + 3 seconds and a fresh midpoint is
available at that step boundary in the same reconstruction segment.

For a completed historical day the label is
    sign * (mid[quote_start + 3s] - mid[quote_start]) / historical_tick.
It is a conditional price move in ticks *after the quote decision*, not a
future-price forecast at all quote starts.  The GP already books the quote's
spread capture, maker fee and future impulse/terminal costs.  The only amount
to add to a maker fill in its Bellman reward is this conditional price move,
converted to HKD per 100-share lot using the TEST day's tick.  A separate
hypothetical immediate-exit diagnostic subtracts maker fee, expected half
spread and taker fee once each; that number must not be injected into QVI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0906"))
sys.path.insert(0, str(ROOT / "0824"))
from directional_market_maker import LOT_SIZE, ALL_IN_FEE_RATE, load_or_extract_trades  # noqa: E402
from run_multiday_strategy import load_day  # noqa: E402
import gp_qvi_model as gp  # noqa: E402
from gap_window_filter import enforce_safe_day_eligibility  # noqa: E402


DEFAULT_MANIFEST = HERE / "output/gap_2m_books/filtered_manifest.json"
DEFAULT_OUTPUT = HERE / "output/fill_net_prior_gap2m"
QUALITY = HERE / "output/combination_study/source_quality.json"
SHAPE = (len(gp.FILL_MODES), len(gp.SIDES), len(gp.QUOTE_ACTIONS),
         len(gp.SESSION_BUCKETS), gp.GPConfig().spread_states)
FIELDS = ("attempts", "proxy_fills", "step_fills", "valid_3s",
          "sum_move_ticks", "sum_sq_move_ticks", "sum_gross_ticks")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_cache(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_manifest(path: Path) -> tuple[dict, list[str], dict[str, Path], dict[str, Path]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    books = {date: _resolve_cache(value)
             for date, value in manifest["book_cache_by_date"].items()}
    trades = {date: _resolve_cache(value)
              for date, value in manifest["trade_cache_by_date"].items()}
    dates = sorted(books)
    if len(dates) != 20 or dates != sorted(trades):
        raise ValueError("expected matching book/trade caches for 20 July dates")
    if any(not date.startswith("2026-07-") for date in dates):
        raise ValueError("manifest must contain July 2026 dates only")
    for date in dates:
        if not books[date].is_file() or not trades[date].is_file():
            raise FileNotFoundError(f"{date}: missing book or trade cache")
    return manifest, dates, books, trades


def _load_day(date: str, book_path: Path, trade_path: Path):
    day = load_day(date, "tolerant_mbo_gap2m", book_path)
    day, audit = enforce_safe_day_eligibility(
        day, horizon_snapshots=20, history_snapshots=100,
    )
    # Existing trade caches must be reused; never re-extract into a stale
    # path from another source when the manifest is incomplete.
    if not trade_path.is_file():
        raise FileNotFoundError(trade_path)
    source = ROOT / "7709_tickdata" / f"hk07709_{date}.csv"
    if not source.is_file():
        source = Path.home() / "Downloads/7709_202607" / f"hk07709_{date}.csv"
    trades = load_or_extract_trades(source, trade_path)
    return day, trades, audit


def probe_day(day, trades, tick_hkd: float, config: gp.GPConfig) -> dict[str, np.ndarray]:
    """Reprobe one filtered completed day; outputs sufficient statistics."""

    stats = {field: np.zeros(SHAPE, dtype=np.float64) for field in FIELDS}
    times = gp.compact_time_to_day_ms(day.send_times)
    mids = (day.book[:, 0].astype(np.float64) + day.book[:, 2]) / 2.0
    quote_indices = gp.select_quote_indices(
        day, day.eligible, config.quote_horizon_snapshots,
    )
    for raw_index in quote_indices:
        index = int(raw_index)
        expiry = index + config.quote_horizon_snapshots
        if expiry >= len(times) or day.segments[index] != day.segments[expiry]:
            continue
        start = int(times[index])
        step_end = start + int(round(config.dt_seconds * 1000))
        target = int(np.searchsorted(times, step_end, side="left"))
        valid_target = (
            target < len(times)
            and day.segments[target] == day.segments[index]
            and int(times[target]) - step_end <= 1000
        )
        ask, bid = float(day.book[index, 0]), float(day.book[index, 2])
        spread = gp._spread_state(ask, bid, tick_hkd, config.spread_states)
        spread_ticks = max(1, int(round((ask - bid) / tick_hkd)))
        bucket = gp._bucket_index(start)
        for side_i, side in enumerate(gp.SIDES):
            sign = 1 if side == "buy" else -1
            for action_i, action in enumerate(gp.QUOTE_ACTIONS):
                if action == "improve" and spread_ticks <= 1:
                    continue
                price = (bid + (tick_hkd if action == "improve" else 0.0)
                         if side == "buy" else
                         ask - (tick_hkd if action == "improve" else 0.0))
                for mode_i, mode in enumerate(gp.FILL_MODES):
                    key = (mode_i, side_i, action_i, bucket, spread)
                    stats["attempts"][key] += 1
                    fill_time, _ = gp._first_fill(
                        day, trades, index, expiry, price, side, mode,
                        tick_hkd, config.latency_ms, times,
                    )
                    if fill_time is None:
                        continue
                    stats["proxy_fills"][key] += 1
                    if int(fill_time) > step_end:
                        continue
                    stats["step_fills"][key] += 1
                    if not valid_target:
                        continue
                    move_ticks = sign * (float(mids[target]) - float(mids[index])) / tick_hkd
                    gross_ticks = sign * (float(mids[target]) - price) / tick_hkd
                    stats["valid_3s"][key] += 1
                    stats["sum_move_ticks"][key] += move_ticks
                    stats["sum_sq_move_ticks"][key] += move_ticks * move_ticks
                    stats["sum_gross_ticks"][key] += gross_ticks
    return stats


def pooled(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("at least one prior day is required")
    return {field: np.sum([record[field] for record in records], axis=0)
            for field in FIELDS}


def hierarchical_move_ticks(stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Shrink cell mean to spread parent (30) and side/action parent (100)."""

    count = stats["valid_3s"]
    total = stats["sum_move_ticks"]
    estimate = np.zeros(SHAPE, dtype=np.float64)
    supported = np.zeros(SHAPE, dtype=bool)
    for mode in range(SHAPE[0]):
        for side in range(SHAPE[1]):
            for action in range(SHAPE[2]):
                n_global = float(count[mode, side, action].sum())
                if n_global < 30:
                    continue
                global_mean = float(total[mode, side, action].sum() / n_global)
                for spread in range(SHAPE[4]):
                    n_spread = float(count[mode, side, action, :, spread].sum())
                    sum_spread = float(total[mode, side, action, :, spread].sum())
                    spread_mean = (sum_spread + 100.0 * global_mean) / (n_spread + 100.0)
                    for bucket in range(SHAPE[3]):
                        key = (mode, side, action, bucket, spread)
                        estimate[key] = (
                            float(total[key]) + 30.0 * spread_mean
                        ) / (float(count[key]) + 30.0)
                        supported[key] = True
    # The GP solver never offers a one-tick improvement at a one-tick spread;
    # keep this impossible state out of exported alpha and support counts.
    estimate[:, :, gp.QUOTE_ACTIONS.index("improve"), :, 0] = 0.0
    supported[:, :, gp.QUOTE_ACTIONS.index("improve"), :, 0] = False
    return estimate, supported


def day_block_sem(records: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Leave-one-completed-day-out jackknife; NaN for one prior day."""

    if len(records) < 2:
        return np.full(SHAPE, np.nan, dtype=np.float64)
    deleted = []
    for omitted in range(len(records)):
        theta, _ = hierarchical_move_ticks(
            pooled([record for index, record in enumerate(records) if index != omitted])
        )
        deleted.append(theta)
    values = np.stack(deleted)
    center = values.mean(axis=0)
    return np.sqrt((len(records) - 1.0) / len(records)
                   * np.sum((values - center) ** 2, axis=0))


def _exit_cost(inputs: gp.GPInputs, tick_hkd: float, dt_seconds: float) -> np.ndarray:
    """Hypothetical market flatten at step end; array [bucket, spread]."""

    m = inputs.transition_probabilities.shape[0]
    spread_ticks = np.arange(1, m + 1, dtype=np.float64)
    maker_or_taker_fee = (
        inputs.calibration_reference_price_hkd * LOT_SIZE * ALL_IN_FEE_RATE
    )
    costs = np.empty((len(gp.SESSION_BUCKETS), m), dtype=np.float64)
    for bucket in range(len(gp.SESSION_BUCKETS)):
        generator = inputs.clock_intensity_per_second[bucket] * (
            inputs.transition_probabilities - np.eye(m)
        )
        transition = expm(generator * dt_seconds)
        transition = np.clip(transition, 0, 1)
        transition /= transition.sum(axis=1, keepdims=True)
        costs[bucket] = (
            transition @ spread_ticks * tick_hkd * LOT_SIZE / 2.0
            + maker_or_taker_fee
        )
    return costs


def build_day_prior(
    records: list[dict[str, np.ndarray]], inputs: gp.GPInputs,
    tick_hkd: float, config: gp.GPConfig,
) -> dict[str, np.ndarray]:
    combined = pooled(records)
    mean_ticks, supported = hierarchical_move_ticks(combined)
    sem_ticks = day_block_sem(records)
    lot_tick_hkd = LOT_SIZE * tick_hkd
    alpha_mean = mean_ticks * lot_tick_hkd
    # With one prior day, a day-block SE does not exist. Keep adverse mean
    # adjustments, but suppress unsupported positive reward. This is a
    # conservative fallback, not a statistical lower confidence bound.
    alpha_lcb = np.where(np.isfinite(sem_ticks),
                         (mean_ticks - sem_ticks) * lot_tick_hkd,
                         np.minimum(mean_ticks, 0.0) * lot_tick_hkd)
    alpha_lcb = np.where(supported, alpha_lcb, 0.0)
    spread = np.arange(1, config.spread_states + 1, dtype=np.float64)
    base_gain = np.empty((len(gp.QUOTE_ACTIONS), config.spread_states))
    for action_i, action in enumerate(gp.QUOTE_ACTIONS):
        base_gain[action_i] = (
            spread / 2.0 - (1 if action == "improve" else 0)
        ) * lot_tick_hkd
    base_gain = np.broadcast_to(base_gain[None, None, :, None, :], SHAPE)
    maker_fee = inputs.calibration_reference_price_hkd * LOT_SIZE * ALL_IN_FEE_RATE
    exit_cost = _exit_cost(inputs, tick_hkd, config.dt_seconds)
    expected_exit = np.broadcast_to(exit_cost[None, None, None, :, :], SHAPE)
    result = {
        "alpha_mean_hkd_per_lot": alpha_mean,
        "alpha_lcb_hkd_per_lot": alpha_lcb,
        "move_mean_ticks": mean_ticks,
        "move_sem_day_block_ticks": sem_ticks,
        "valid_fill_count": combined["valid_3s"],
        "supported": supported,
        "modeled_quote_spread_gain_hkd": base_gain,
        "hypothetical_gross_mean_hkd": base_gain + alpha_mean,
        "maker_fee_hkd": np.full(SHAPE, maker_fee),
        "hypothetical_exit_cost_hkd": expected_exit,
        "hypothetical_net_mean_hkd": base_gain + alpha_mean - maker_fee - expected_exit,
        "hypothetical_net_lcb_hkd": base_gain + alpha_lcb - maker_fee - expected_exit,
    }
    return result


def load_prior_alpha(
    date: str, mode: str, bucket: int, inputs: gp.GPInputs,
    tick: float, config: gp.GPConfig, *,
    output_dir: Path = DEFAULT_OUTPUT, estimate: str = "mean",
) -> np.ndarray:
    """Return side x action x spread HKD/lot add-on for one GP solve."""

    if estimate not in {"mean", "lcb"}:
        raise ValueError("estimate must be mean or lcb")
    if mode not in gp.FILL_MODES or not 0 <= bucket < len(gp.SESSION_BUCKETS):
        raise ValueError("invalid fill mode or session bucket")
    path = Path(output_dir) / f"day_{date}.npz"
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta_json"]))
        if (meta["date"] != date
                or not np.isclose(meta["tick_hkd"], tick, rtol=0, atol=1e-12)
                or tuple(meta["prior_dates"]) != tuple(inputs.calibration_dates)
                or meta["spread_states"] != config.spread_states
                or not np.isclose(meta["dt_seconds"], config.dt_seconds)):
            raise ValueError(f"{date}: prior cache differs from GP calibration or tick")
        key = f"alpha_{estimate}_hkd_per_lot"
        alpha = archive[key][gp.FILL_MODES.index(mode), :, :, bucket, :].copy()
    if alpha.shape != (len(gp.SIDES), len(gp.QUOTE_ACTIONS), config.spread_states):
        raise AssertionError("prior alpha shape mismatch")
    return alpha


def _quality_issue_dates() -> set[str]:
    data = json.loads(QUALITY.read_text(encoding="utf-8"))
    return {row["date"] for row in data["days"]
            if row["quality_group"] == "recorded_reconstruction_issue"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prior-days", type=int, default=5)
    parser.add_argument("--prior-quality", choices=("all", "zero-recorded-errors"),
                        default="all")
    parser.add_argument("--limit-test-days", type=int, default=None)
    parser.add_argument("--force-probe", action="store_true")
    args = parser.parse_args()
    if args.prior_days < 1 or (args.limit_test_days is not None
                              and args.limit_test_days < 1):
        parser.error("prior-days and limit-test-days must be positive")

    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()

    manifest, dates, books, trade_paths = _load_manifest(args.manifest)
    issue_dates = _quality_issue_dates() if args.prior_quality == "zero-recorded-errors" else set()
    if dates[0] in issue_dates:
        raise ValueError("first calibration day is excluded by quality rule")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_tests = dates[1: 1 + args.limit_test_days] if args.limit_test_days else dates[1:]
    required_dates = dates[:dates.index(selected_tests[-1]) + 1]
    config = gp.GPConfig()
    daily_probe = {}
    daily_calibration = {}
    probe_audits = []
    cell_rows = []

    def get_completed(date: str) -> None:
        if date in daily_probe:
            return
        book_path, trade_path = books[date], trade_paths[date]
        book_sha, trade_sha = _sha256(book_path), _sha256(trade_path)
        cache_path = args.output_dir / f"probe_{date}.npz"
        day, trades, eligibility = _load_day(date, book_path, trade_path)
        daily_calibration[date] = gp.collect_daily_calibration_stats(day, trades, config)
        tick = daily_calibration[date].tick_hkd
        cached = False
        if cache_path.is_file() and not args.force_probe:
            with np.load(cache_path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["meta_json"]))
                if (meta.get("book_sha256") == book_sha
                        and meta.get("trade_sha256") == trade_sha
                        and meta.get("date") == date
                        and np.isclose(meta.get("tick_hkd", np.nan), tick)
                        and meta.get("dt_seconds") == config.dt_seconds):
                    daily_probe[date] = {field: archive[field].copy() for field in FIELDS}
                    cached = True
        if not cached:
            daily_probe[date] = probe_day(day, trades, tick, config)
            meta = {"date": date, "book_sha256": book_sha,
                    "trade_sha256": trade_sha, "book_cache": str(book_path),
                    "trade_cache": str(trade_path), "tick_hkd": tick,
                    "dt_seconds": config.dt_seconds,
                    "eligibility": eligibility}
            np.savez_compressed(cache_path,
                                **daily_probe[date],
                                meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
        probe_audits.append({
            "date": date, "probe_cache": str(cache_path.relative_to(ROOT)),
            "cached": cached, "tick_hkd": tick,
            "eligible_quote_count": len(day.eligible),
            "attempts": int(daily_probe[date]["attempts"].sum()),
            "proxy_fills": int(daily_probe[date]["proxy_fills"].sum()),
            "within_step_fills": int(daily_probe[date]["step_fills"].sum()),
            "valid_3s": int(daily_probe[date]["valid_3s"].sum()),
            "eligibility_audit": eligibility,
        })
        day.features = None

    get_completed(required_dates[0])
    prior_records = []
    for date in selected_tests:
        position = dates.index(date)
        prior_dates = [d for d in dates[:position] if d not in issue_dates][-args.prior_days:]
        if not prior_dates or any(prior >= date for prior in prior_dates):
            raise AssertionError("empty or noncausal prior")
        for prior in prior_dates:
            get_completed(prior)
        # The test day's tick is an inherited legacy convention for QVI
        # reconciliation; it is not a price label used to fit alpha.
        day, _, _ = _load_day(date, books[date], trade_paths[date])
        test_tick = gp.infer_tick_hkd(day)
        inputs = gp.aggregate_calibration_stats(
            [daily_calibration[d] for d in prior_dates], config,
        )
        prior = build_day_prior([daily_probe[d] for d in prior_dates],
                                inputs, test_tick, config)
        meta = {
            "date": date, "prior_dates": prior_dates,
            "prior_quality": args.prior_quality, "prior_days": args.prior_days,
            "tick_hkd": test_tick, "spread_states": config.spread_states,
            "dt_seconds": config.dt_seconds,
            "input_reference_price_hkd": inputs.calibration_reference_price_hkd,
            "book_cache": str(books[date].relative_to(ROOT)),
            "label": "signed (mid at quote_start+3s minus mid at quote_start) / historical_tick, conditional on proxy fill by quote_start+3s",
            "lcb": "mean minus one leave-one-prior-day-out jackknife standard error; with one prior day, clip positive alpha to zero and retain negative mean (not a statistical LCB)",
        }
        destination = args.output_dir / f"day_{date}.npz"
        np.savez_compressed(destination, **prior,
                            meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
        prior_records.append({
            "date": date, "prior_dates": prior_dates,
            "test_tick_hkd": test_tick,
            "cache": str(destination.relative_to(ROOT)),
            "prior_probe_valid_3s": int(sum(daily_probe[d]["valid_3s"].sum()
                                            for d in prior_dates)),
            "alpha_mean_min_hkd": float(np.min(prior["alpha_mean_hkd_per_lot"])),
            "alpha_mean_max_hkd": float(np.max(prior["alpha_mean_hkd_per_lot"])),
            "alpha_lcb_min_hkd": float(np.min(prior["alpha_lcb_hkd_per_lot"])),
            "alpha_lcb_max_hkd": float(np.max(prior["alpha_lcb_hkd_per_lot"])),
            "positive_hypothetical_net_cells": int(np.count_nonzero(
                prior["hypothetical_net_mean_hkd"] > 0)),
        })
        for mode_i, mode in enumerate(gp.FILL_MODES):
            for side_i, side in enumerate(gp.SIDES):
                for action_i, action in enumerate(gp.QUOTE_ACTIONS):
                    for bucket in range(len(gp.SESSION_BUCKETS)):
                        for spread in range(config.spread_states):
                            key = (mode_i, side_i, action_i, bucket, spread)
                            sem = float(prior["move_sem_day_block_ticks"][key])
                            cell_rows.append({
                                "date": date, "prior_dates": "|".join(prior_dates),
                                "mode": mode, "side": side, "action": action,
                                "bucket": bucket, "spread_state_ticks": spread + 1,
                                "valid_prior_fills": int(prior["valid_fill_count"][key]),
                                "supported": int(prior["supported"][key]),
                                "move_mean_ticks": float(prior["move_mean_ticks"][key]),
                                "move_sem_day_block_ticks": sem if np.isfinite(sem) else "",
                                "alpha_mean_hkd_per_lot": float(prior["alpha_mean_hkd_per_lot"][key]),
                                "alpha_lcb_hkd_per_lot": float(prior["alpha_lcb_hkd_per_lot"][key]),
                                "modeled_quote_spread_gain_hkd": float(prior["modeled_quote_spread_gain_hkd"][key]),
                                "hypothetical_gross_mean_hkd": float(prior["hypothetical_gross_mean_hkd"][key]),
                                "maker_fee_hkd": float(prior["maker_fee_hkd"][key]),
                                "hypothetical_exit_cost_hkd": float(prior["hypothetical_exit_cost_hkd"][key]),
                                "hypothetical_net_mean_hkd": float(prior["hypothetical_net_mean_hkd"][key]),
                                "hypothetical_net_lcb_hkd": float(prior["hypothetical_net_lcb_hkd"][key]),
                            })
        print("PRIOR", date, "history", len(prior_dates),
              "valid", prior_records[-1]["prior_probe_valid_3s"], flush=True)
        if date in required_dates[:-1] and date not in issue_dates:
            get_completed(date)
        day.features = None

    output = {
        "design": {
            "manifest": str(args.manifest),
            "manifest_rule": manifest.get("rule"),
            "selected_test_dates": selected_tests,
            "prior_quality": args.prior_quality,
            "rolling_prior_days": args.prior_days,
            "probe_horizon_snapshots": config.quote_horizon_snapshots,
            "gp_step_seconds": config.dt_seconds,
            "markout_clock": "quote_start+3s, not fill+3s",
            "markout_midpoint": "first snapshot at or after step end, at most 1s late, same reconstruction segment",
            "fill_condition": "first touch/through proxy fill by step end and within 20-snapshot quote life",
            "tick_normalization": "historical conditional midpoint move divided by that day's tick; multiply shrunken ticks by current test tick and 100 shares",
            "qvi_reward": "add only conditional quote-start-to-step-end midpoint move; baseline GP already counts spread gain and maker fee, and continuation/terminal/impulse costs",
            "hypothetical_net": "GP modeled spread gain + alpha - maker fee - Markov expected halfspread at 3s - taker fee; diagnostic immediate flatten only",
            "uncertainty": "leave-one-completed-prior-day-out jackknife; one prior day has undefined SE, so the LCB variant clips positive alpha to zero and retains negative mean without claiming confidence coverage",
            "shrinkage": "within mode/side/action, spread parent 100 fills toward global, cell 30 fills toward spread parent; parent requires at least 30 fills",
        },
        "prior_by_test_day": prior_records,
        "completed_day_probes": probe_audits,
        "validation": {
            "all_prior_dates_before_test": all(all(p < r["date"] for p in r["prior_dates"])
                                           for r in prior_records),
            "all_probe_sources_from_filtered_manifest": all(
                books[r["date"]].is_file() for r in probe_audits),
        },
        "limitations": [
            "Touch and through are proxy fills without queue position.",
            "The 3-second midpoint can be up to one second late; this residual timing error is audited by the fresh-midpoint rule.",
            "The immediate market exit in the net diagnostic is hypothetical; the QVI instead optimizes continuation and impulse decisions.",
            "Reused GP test-day tick inference is based on the complete test-day price grid; a venue tick table is needed for strict live causality.",
            "Only five prior trading days are available for day-block uncertainty; overlapping quote probes within a day are not independent samples.",
        ],
    }
    with (args.output_dir / "daily_cells.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    (args.output_dir / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("SAVED", args.output_dir / "results.json", flush=True)


if __name__ == "__main__":
    main()
