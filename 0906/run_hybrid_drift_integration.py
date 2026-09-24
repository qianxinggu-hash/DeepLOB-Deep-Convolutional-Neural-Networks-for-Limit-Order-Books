#!/usr/bin/env python3
"""Causal Hybrid-DeepLOB drift integration for AS and GP/QVI quotes.

The default experiment calibrates the probability-to-move map and market
inputs on 2026-08-04, then evaluates the untouched 2026-08-07 day.  It compares
classic AS vs AS with a shifted reservation price, and martingale GP/QVI vs a
GP/QVI value function augmented with a finite-state DeepLOB drift process.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEGACY_DIR = ROOT / "0824"
for path in (ROOT, HERE, LEGACY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from directional_market_maker import (  # noqa: E402
    ASDriftParameters,
    ASParameters,
    load_or_extract_trades,
    simulate_market_maker,
)
from gp_qvi_model import (  # noqa: E402
    FILL_MODES,
    SESSION_BUCKETS,
    GPConfig,
    aggregate_calibration_stats,
    assign_gp_drift_states,
    collect_daily_calibration_stats,
    config_asdict,
    fit_gp_drift_regime,
    infer_tick_hkd,
    policy_audit,
    rescale_gp_drift_regime,
    select_quote_indices,
    simulate_gp_day,
    solve_gp_drift_policy,
    solve_gp_policy,
)
from hybrid_drift_adapter import (  # noqa: E402
    HybridDriftAdapter,
    fit_probability_move_calibrator,
    future_mid_move_ticks,
)
from train_7709_deeplob import (  # noqa: E402
    DayData,
    eligible_targets,
    equation4_returns,
)


PRICE_COLUMNS = np.arange(0, 40, 2)
SIZE_COLUMNS = np.arange(1, 40, 2)


@dataclass
class ReplayDay:
    date: str
    book: np.ndarray
    send_times: np.ndarray
    segments: np.ndarray
    eligible: np.ndarray
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT / "data/processed/7709"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=ROOT / "7709_tickdata"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/7709/deeplob_7709_hybrid_k20_best.pt",
    )
    parser.add_argument("--calibration-date", default="2026-08-04")
    parser.add_argument("--test-date", default="2026-08-07")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--drift-states", type=int, default=5)
    parser.add_argument("--as-gamma", type=float, default=0.02)
    parser.add_argument("--as-order-decay", type=float, default=1.0)
    parser.add_argument(
        "--gp-gamma",
        type=float,
        default=5.0,
        help="GP running inventory penalty; select only on pre-test data",
    )
    parser.add_argument("--max-inventory", type=int, default=5)
    parser.add_argument(
        "--cache-dir", type=Path, default=HERE / "data/hybrid_drift"
    )
    parser.add_argument(
        "--max-quotes",
        type=int,
        default=0,
        help="diagnostic cap per day; 0 runs every non-overlapping quote",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "output/hybrid_deeplob_as_gp_integration.json",
    )
    args = parser.parse_args()
    if min(args.batch_size, args.drift_states, args.max_inventory) < 1:
        parser.error("batch size, drift states and inventory cap must be positive")
    if args.as_gamma <= 0.0 or args.as_order_decay <= 0.0 or args.gp_gamma < 0.0:
        parser.error("AS gamma/decay must be positive and GP gamma non-negative")
    if args.max_quotes < 0:
        parser.error("max-quotes cannot be negative")
    return args


def load_model_day(processed_dir: Path, date: str) -> tuple[DayData, np.ndarray]:
    path = processed_dir / f"hk07709_{date}_s10_lob.npz"
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=False)
        send_times = archive["send_times"].astype(np.int64, copy=False)
        sessions = archive["sessions"].astype(np.int16, copy=False)
        metadata = json.loads(str(archive["metadata"]))
    day = DayData(date, path, features, sessions, metadata)
    return day, send_times


def execution_scale(metadata: dict[str, Any]) -> tuple[float, float]:
    """Recover HKD prices and share quantities from neural reconstruction units."""

    feature_scale = float(metadata.get("feature_scale", 100_000.0))
    # Raw HK tick files encode price in thousandths of HKD; the neural
    # reconstruction divides all 40 columns by feature_scale.
    return feature_scale / 1_000.0, feature_scale


def make_replay_day(
    model_day: DayData,
    send_times: np.ndarray,
    sequence_length: int,
    horizon: int,
    max_quotes: int,
) -> ReplayDay:
    price_scale, size_scale = execution_scale(model_day.metadata)
    book = model_day.features.astype(np.float64, copy=True)
    book[:, PRICE_COLUMNS] *= price_scale
    book[:, SIZE_COLUMNS] *= size_scale
    returns = equation4_returns(model_day, horizon)
    eligible = eligible_targets(model_day, returns, sequence_length, 1)
    provisional = ReplayDay(
        model_day.date,
        book,
        send_times,
        model_day.sessions,
        eligible,
        model_day.metadata,
    )
    quote_indices = select_quote_indices(provisional, eligible, horizon)
    if max_quotes:
        quote_indices = quote_indices[:max_quotes]
    provisional.eligible = quote_indices
    return provisional


def quote_interval_seconds(day: ReplayDay, quote_indices: np.ndarray) -> float:
    compact = quote_indices
    times = day.send_times.astype(np.int64) % 1_000_000_000
    hour = times // 10_000_000
    minute = (times // 100_000) % 100
    second = (times // 1_000) % 100
    millisecond = times % 1_000
    day_ms = ((hour * 60 + minute) * 60 + second) * 1_000 + millisecond
    gaps: list[np.ndarray] = []
    for session in np.unique(day.segments[compact]):
        selected = compact[day.segments[compact] == session]
        if len(selected) > 1:
            gaps.append(np.diff(day_ms[selected]) / 1_000.0)
    if not gaps:
        raise ValueError("not enough quote decisions to estimate signal transitions")
    values = np.concatenate(gaps)
    values = values[values > 0.0]
    if not len(values):
        raise ValueError("quote decision intervals are not positive")
    return float(np.median(values))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    calibration_model_day, calibration_times = load_model_day(
        args.processed_dir, args.calibration_date
    )
    test_model_day, test_times = load_model_day(args.processed_dir, args.test_date)
    adapter = HybridDriftAdapter(args.checkpoint, args.device)
    horizon = adapter.horizon_snapshots
    calibration_day = make_replay_day(
        calibration_model_day,
        calibration_times,
        adapter.sequence_length,
        horizon,
        args.max_quotes,
    )
    test_day = make_replay_day(
        test_model_day,
        test_times,
        adapter.sequence_length,
        horizon,
        args.max_quotes,
    )
    calibration_quotes = select_quote_indices(
        calibration_day, calibration_day.eligible, horizon
    )
    test_quotes = select_quote_indices(test_day, test_day.eligible, horizon)

    calibration_probabilities = adapter.probabilities(
        calibration_model_day, calibration_quotes, args.batch_size
    )
    calibration_tick = infer_tick_hkd(calibration_day)
    test_tick = infer_tick_hkd(test_day)
    calibration_price_scale, _ = execution_scale(calibration_model_day.metadata)
    calibration_moves = future_mid_move_ticks(
        calibration_model_day.features,
        calibration_model_day.sessions,
        calibration_quotes,
        horizon,
        calibration_tick / calibration_price_scale,
    )
    calibrator = fit_probability_move_calibrator(
        calibration_probabilities, calibration_moves
    )
    calibration_forecast = adapter.forecast(
        calibration_model_day,
        calibration_quotes,
        calibrator,
        calibration_tick,
        args.batch_size,
    )
    test_forecast = adapter.forecast(
        test_model_day,
        test_quotes,
        calibrator,
        test_tick,
        args.batch_size,
    )

    cache_dir = args.cache_dir
    calibration_trades = load_or_extract_trades(
        args.raw_dir / f"hk07709_{args.calibration_date}.csv",
        cache_dir / f"hk07709_{args.calibration_date}_trades.npz",
    )
    test_trades = load_or_extract_trades(
        args.raw_dir / f"hk07709_{args.test_date}.csv",
        cache_dir / f"hk07709_{args.test_date}_trades.npz",
    )

    variance = float(np.var(calibration_moves, ddof=1))
    classic_as = ASParameters(
        args.as_gamma,
        args.as_order_decay,
        variance,
        args.max_inventory,
        horizon,
    )
    drift_as = ASDriftParameters(
        args.as_gamma,
        args.as_order_decay,
        variance,
        1.0,
        args.max_inventory,
        horizon,
    )
    as_results: dict[str, Any] = {}
    for fill_mode in FILL_MODES:
        as_results[fill_mode] = {}
        as_results[fill_mode]["classic_as"], _ = simulate_market_maker(
            test_day,
            test_trades,
            test_quotes,
            np.zeros(len(test_quotes)),
            classic_as,
            fill_mode,
            "classic_as",
        )
        as_results[fill_mode]["hybrid_drift_as"], _ = simulate_market_maker(
            test_day,
            test_trades,
            test_quotes,
            test_forecast.expected_move_ticks,
            drift_as,
            fill_mode,
            "hybrid_drift_as",
        )

    gp_config = GPConfig(
        max_inventory_lots=args.max_inventory,
        quote_horizon_snapshots=horizon,
        inventory_penalty_gamma=args.gp_gamma,
    )
    gp_inputs = aggregate_calibration_stats(
        [collect_daily_calibration_stats(calibration_day, calibration_trades, gp_config)],
        gp_config,
    )
    sequences = [
        calibration_forecast.drift_hkd_per_second[
            calibration_day.segments[calibration_quotes] == session
        ]
        for session in np.unique(calibration_day.segments[calibration_quotes])
    ]
    raw_regime = fit_gp_drift_regime(sequences, args.drift_states)
    source_interval = quote_interval_seconds(calibration_day, calibration_quotes)
    drift_regime = rescale_gp_drift_regime(
        raw_regime, source_interval, gp_config.dt_seconds
    )
    test_drift_states = assign_gp_drift_states(
        test_forecast.drift_hkd_per_second, drift_regime
    )

    gp_results: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for fill_mode in FILL_MODES:
        baseline_policies = {
            bucket: solve_gp_policy(
                gp_inputs, test_tick, fill_mode, bucket, gp_config
            )
            for bucket in range(len(SESSION_BUCKETS))
        }
        drift_policies = {
            bucket: solve_gp_drift_policy(
                gp_inputs,
                test_tick,
                fill_mode,
                bucket,
                gp_config,
                drift_regime,
            )
            for bucket in range(len(SESSION_BUCKETS))
        }
        gp_results[fill_mode] = {
            "martingale_gp": simulate_gp_day(
                test_day,
                test_trades,
                baseline_policies,
                gp_config,
                fill_mode,
                "martingale_gp",
            ),
            "hybrid_drift_gp": simulate_gp_day(
                test_day,
                test_trades,
                drift_policies,
                gp_config,
                fill_mode,
                "hybrid_drift_gp",
                test_drift_states,
            ),
        }
        audits[fill_mode] = policy_audit(drift_policies[0])

    output = {
        "experiment": "Hybrid DeepLOB drift integrated with AS and GP/QVI",
        "calibration_date": args.calibration_date,
        "test_date": args.test_date,
        "diagnostic_quote_cap": args.max_quotes or None,
        "chronological": args.calibration_date < args.test_date,
        "units": {
            "neural_price_scale_to_execution": execution_scale(
                calibration_model_day.metadata
            )[0],
            "calibration_tick_hkd": calibration_tick,
            "test_tick_hkd": test_tick,
            "as_signal": "expected move in execution ticks over 20 snapshots",
            "gp_signal": "expected HKD price drift per second",
        },
        "probability_calibrator": asdict(calibrator),
        "signal": {
            "calibration_quotes": len(calibration_quotes),
            "test_quotes": len(test_quotes),
            "test_expected_move_ticks_mean": float(
                np.mean(test_forecast.expected_move_ticks)
            ),
            "test_expected_move_ticks_std": float(
                np.std(test_forecast.expected_move_ticks)
            ),
            "test_horizon_seconds_median": float(
                np.median(test_forecast.horizon_seconds)
            ),
            "drift_transition_source_interval_seconds": source_interval,
            "gp_dt_seconds": gp_config.dt_seconds,
            "drift_regime": asdict(drift_regime),
        },
        "as_parameters": {
            "classic": asdict(classic_as),
            "drift": asdict(drift_as),
        },
        "gp_config": config_asdict(gp_config),
        "as_results": as_results,
        "gp_results": gp_results,
        "gp_policy_audit": audits,
        "limitations": [
            "AS order-decay and gamma use explicit command-line defaults; tune them only on pre-test data.",
            "The GP drift chain assumes conditional independence from the spread chain.",
            "Touch/through fills remain queue-position proxies, not exchange-exact fills.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(jsonable(output), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {args.output.resolve()}")
    for fill_mode in FILL_MODES:
        print(
            fill_mode,
            "AS",
            as_results[fill_mode]["classic_as"]["net_pnl_hkd"],
            "->",
            as_results[fill_mode]["hybrid_drift_as"]["net_pnl_hkd"],
            "GP",
            gp_results[fill_mode]["martingale_gp"]["net_pnl_hkd"],
            "->",
            gp_results[fill_mode]["hybrid_drift_gp"]["net_pnl_hkd"],
        )


if __name__ == "__main__":
    main()
