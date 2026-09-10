#!/usr/bin/env python3
"""Guilbaud--Pham mean-criterion/QVI approximation for HK 07709.

The paper's reduced value function is ``v=x+yp+phi(t,y,s)``.  This module
solves the corresponding finite-state dynamic program in the paper's state
variables: time-to-go, inventory and discrete spread.  Limit controls are
``none / best / improve one tick`` on each side and the impulse control is a
market order that reduces inventory.  The continuous Cox/Markov dynamics are
discretised at the empirical 20-snapshot order lifetime.

Calibration and replay deliberately use the 0906 execution proxy: after 5 ms,
the first qualifying future trade or opposite L1 quote fills the order, and an
unfilled order expires after snapshot t+20.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
from scipy.linalg import expm

from directional_market_maker import (
    ALL_IN_FEE_RATE,
    LATENCY_MS,
    LOT_SIZE,
    OFFICIAL_FIXED_FEE_RATE,
    TradeData,
    compact_time_to_day_ms,
    select_quote_indices,
)


FILL_MODES = ("through", "touch")
SIDES = ("buy", "sell")
QUOTE_ACTIONS = ("best", "improve")
ACTION_NONE = 0
ACTION_BEST = 1
ACTION_IMPROVE = 2
SESSION_BUCKETS = (
    (34_200_000, 37_800_000, "09:30-10:30"),
    (37_800_000, 41_400_000, "10:30-11:30"),
    (41_400_000, 43_200_000, "11:30-12:00"),
    (46_800_000, 50_400_000, "13:00-14:00"),
    (50_400_000, 54_000_000, "14:00-15:00"),
    (54_000_000, 57_600_000, "15:00-16:00"),
)


@dataclass(frozen=True)
class GPConfig:
    spread_states: int = 6
    max_inventory_lots: int = 10
    quote_horizon_snapshots: int = 20
    latency_ms: int = LATENCY_MS
    horizon_seconds: float = 300.0
    time_steps: int = 100
    # Paper Table 4 uses gamma=5.  For the 7709 replay the inventory coordinate
    # is one 100-share board lot, so the reported coefficient is per lot^2-sec.
    inventory_penalty_gamma: float = 5.0
    execution_prior_seconds: float = 30.0
    clock_prior_seconds: float = 300.0

    @property
    def dt_seconds(self) -> float:
        return self.horizon_seconds / self.time_steps


@dataclass
class DailyCalibrationStats:
    date: str
    tick_hkd: float
    reference_price_hkd: float
    transition_counts: np.ndarray
    clock_jumps: np.ndarray
    clock_exposure_seconds: np.ndarray
    execution_fills: dict[str, np.ndarray]
    execution_exposure_seconds: dict[str, np.ndarray]
    lifecycle_seconds_sum: float
    lifecycle_count: int
    quoted_probes: int
    capped_spread_share: float


@dataclass(frozen=True)
class GPInputs:
    transition_probabilities: np.ndarray
    clock_intensity_per_second: np.ndarray
    execution_intensity_per_second: dict[str, np.ndarray]
    calibration_dates: tuple[str, ...]
    calibration_reference_price_hkd: float
    mean_lifecycle_seconds: float
    audit: dict[str, Any]


@dataclass
class GPPolicy:
    values: np.ndarray
    bid_action: np.ndarray
    ask_action: np.ndarray
    impulse_target: np.ndarray
    make_bid_action: np.ndarray
    make_ask_action: np.ndarray
    config: GPConfig
    tick_hkd: float
    fill_mode: str
    clock_bucket: int
    market_orders_enabled: bool


def infer_tick_hkd(day: Any) -> float:
    """Infer the largest exchange-grid step supported by essentially all L1 prices."""

    prices = np.concatenate(
        [day.book[:, 0].astype(np.float64), day.book[:, 2].astype(np.float64)]
    )
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if not len(prices):
        raise ValueError("no positive L1 prices")
    candidates = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
    for candidate in candidates:
        residual = np.abs(prices / candidate - np.rint(prices / candidate))
        # A tiny fraction of reconstructed snapshots can retain off-grid prices
        # around a tick-band transition; require the dominant legal grid rather
        # than letting those records force an artificially small tick.
        if float(np.mean(residual < 2e-4)) >= 0.995:
            return float(candidate)
    raise ValueError("unable to infer a stable tick grid")


def _bucket_index(time_ms: int) -> int:
    for index, (start, end, _) in enumerate(SESSION_BUCKETS):
        if start <= time_ms < end:
            return index
    if 43_200_000 <= time_ms < 46_800_000:
        return 2
    return len(SESSION_BUCKETS) - 1


def _spread_state(best_ask: float, best_bid: float, tick: float, m: int) -> int:
    raw = max(1, int(round((best_ask - best_bid) / tick)))
    return min(raw, m) - 1


def _first_fill(
    day: Any,
    trades: TradeData,
    quote_index: int,
    expiry_index: int,
    quote_hkd: float,
    side: str,
    fill_mode: str,
    tick_hkd: float,
    latency_ms: int,
    snapshot_times_ms: np.ndarray | None = None,
) -> tuple[int | None, str | None]:
    times = (
        compact_time_to_day_ms(day.send_times)
        if snapshot_times_ms is None
        else snapshot_times_ms
    )
    active_after = int(times[quote_index]) + latency_ms
    expiry_ms = int(times[expiry_index]) + latency_ms
    future = np.arange(quote_index + 1, expiry_index + 1, dtype=np.int64)
    future = future[times[future] > active_after]
    extra = tick_hkd if fill_mode == "through" else 0.0

    candidates: list[tuple[int, str]] = []
    if len(future):
        if side == "buy":
            matches = np.flatnonzero(day.book[future, 0] <= quote_hkd - extra + 1e-8)
            source = "best_ask"
        else:
            matches = np.flatnonzero(day.book[future, 2] >= quote_hkd + extra - 1e-8)
            source = "best_bid"
        if len(matches):
            candidates.append((int(times[int(future[int(matches[0])])]), source))

    left = int(np.searchsorted(trades.trade_time_ms, active_after, side="right"))
    right = int(np.searchsorted(trades.trade_time_ms, expiry_ms, side="right"))
    trade_prices = trades.price_hkd[left:right]
    if side == "buy":
        trade_matches = np.flatnonzero(trade_prices <= quote_hkd - extra + 1e-8)
    else:
        trade_matches = np.flatnonzero(trade_prices >= quote_hkd + extra - 1e-8)
    if len(trade_matches):
        trade_index = left + int(trade_matches[0])
        candidates.append((int(trades.trade_time_ms[trade_index]), "trade_price"))
    if not candidates:
        return None, None
    # Match the 0906 replay tie break: book evidence sorts before trade evidence.
    return min(candidates, key=lambda item: (item[0], item[1]))


def collect_daily_calibration_stats(
    day: Any,
    trades: TradeData,
    config: GPConfig,
) -> DailyCalibrationStats:
    """Collect sufficient statistics for causal rolling GP calibration."""

    tick = infer_tick_hkd(day)
    times = compact_time_to_day_ms(day.send_times)
    m = config.spread_states
    states = np.asarray(
        [
            _spread_state(float(row[0]), float(row[2]), tick, m)
            for row in day.book
        ],
        dtype=np.int16,
    )
    raw_spreads = np.maximum(
        1, np.rint((day.book[:, 0] - day.book[:, 2]) / tick).astype(np.int64)
    )
    transition_counts = np.zeros((m, m), dtype=np.float64)
    clock_jumps = np.zeros(len(SESSION_BUCKETS), dtype=np.float64)
    clock_exposure = np.zeros(len(SESSION_BUCKETS), dtype=np.float64)
    for index in range(len(day.book) - 1):
        if int(day.segments[index]) != int(day.segments[index + 1]):
            continue
        elapsed = (int(times[index + 1]) - int(times[index])) / 1000.0
        if not (0 < elapsed <= 30.0):
            continue
        bucket = _bucket_index(int(times[index]))
        clock_exposure[bucket] += elapsed
        if raw_spreads[index + 1] != raw_spreads[index]:
            clock_jumps[bucket] += 1
        source_state = int(states[index])
        target_state = int(states[index + 1])
        if source_state != target_state:
            transition_counts[source_state, target_state] += 1

    execution_fills = {
        mode: np.zeros((len(SIDES), len(QUOTE_ACTIONS), m), dtype=np.float64)
        for mode in FILL_MODES
    }
    execution_exposure = {
        mode: np.zeros((len(SIDES), len(QUOTE_ACTIONS), m), dtype=np.float64)
        for mode in FILL_MODES
    }
    quote_indices = select_quote_indices(day, day.eligible, config.quote_horizon_snapshots)
    lifecycle_sum = 0.0
    lifecycle_count = 0
    quoted_probes = 0
    for index_value in quote_indices:
        index = int(index_value)
        expiry = index + config.quote_horizon_snapshots
        if expiry >= len(day.book) or day.segments[index] != day.segments[expiry]:
            continue
        active_after = int(times[index]) + config.latency_ms
        expiry_ms = int(times[expiry]) + config.latency_ms
        full_exposure = max((expiry_ms - active_after) / 1000.0, 0.001)
        lifecycle_sum += full_exposure
        lifecycle_count += 1
        state = int(states[index])
        best_ask = float(day.book[index, 0])
        best_bid = float(day.book[index, 2])
        spread_ticks = int(raw_spreads[index])
        for side_index, side in enumerate(SIDES):
            for action_index, action in enumerate(QUOTE_ACTIONS):
                if action == "improve" and spread_ticks <= 1:
                    continue
                if side == "buy":
                    quote = best_bid + (tick if action == "improve" else 0.0)
                else:
                    quote = best_ask - (tick if action == "improve" else 0.0)
                quoted_probes += 1
                for mode in FILL_MODES:
                    fill_time, _ = _first_fill(
                        day,
                        trades,
                        index,
                        expiry,
                        quote,
                        side,
                        mode,
                        tick,
                        config.latency_ms,
                        times,
                    )
                    exposure = (
                        max((fill_time - active_after) / 1000.0, 0.001)
                        if fill_time is not None
                        else full_exposure
                    )
                    execution_exposure[mode][side_index, action_index, state] += exposure
                    if fill_time is not None:
                        execution_fills[mode][side_index, action_index, state] += 1

    mids = (day.book[:, 0].astype(np.float64) + day.book[:, 2]) / 2.0
    return DailyCalibrationStats(
        date=str(day.date),
        tick_hkd=tick,
        reference_price_hkd=float(np.median(mids)),
        transition_counts=transition_counts,
        clock_jumps=clock_jumps,
        clock_exposure_seconds=clock_exposure,
        execution_fills=execution_fills,
        execution_exposure_seconds=execution_exposure,
        lifecycle_seconds_sum=lifecycle_sum,
        lifecycle_count=lifecycle_count,
        quoted_probes=quoted_probes,
        capped_spread_share=float(np.mean(raw_spreads > m)),
    )


def aggregate_calibration_stats(
    records: Iterable[DailyCalibrationStats],
    config: GPConfig,
) -> GPInputs:
    records = list(records)
    if not records:
        raise ValueError("at least one calibration day is required")
    m = config.spread_states
    transition_counts = sum(
        (record.transition_counts for record in records),
        np.zeros((m, m), dtype=np.float64),
    )
    rho = np.zeros((m, m), dtype=np.float64)
    global_dest = transition_counts.sum(axis=0)
    global_dest = global_dest + 0.05
    for state in range(m):
        row = transition_counts[state].copy()
        row[state] = 0.0
        if row.sum() == 0:
            row = global_dest.copy()
            row[state] = 0.0
        row += np.asarray([0.0 if j == state else 0.05 for j in range(m)])
        rho[state] = row / row.sum()

    jumps = sum(
        (record.clock_jumps for record in records),
        np.zeros(len(SESSION_BUCKETS), dtype=np.float64),
    )
    clock_exposure = sum(
        (record.clock_exposure_seconds for record in records),
        np.zeros(len(SESSION_BUCKETS), dtype=np.float64),
    )
    global_clock = float(jumps.sum() / max(clock_exposure.sum(), 1e-9))
    clock = (jumps + config.clock_prior_seconds * global_clock) / (
        clock_exposure + config.clock_prior_seconds
    )

    intensities: dict[str, np.ndarray] = {}
    execution_audit: dict[str, Any] = {}
    for mode in FILL_MODES:
        fills = sum(
            (record.execution_fills[mode] for record in records),
            np.zeros((len(SIDES), len(QUOTE_ACTIONS), m), dtype=np.float64),
        )
        exposure = sum(
            (record.execution_exposure_seconds[mode] for record in records),
            np.zeros((len(SIDES), len(QUOTE_ACTIONS), m), dtype=np.float64),
        )
        intensity = np.zeros_like(fills)
        for side_index in range(len(SIDES)):
            for action_index in range(len(QUOTE_ACTIONS)):
                total_rate = float(
                    fills[side_index, action_index].sum()
                    / max(exposure[side_index, action_index].sum(), 1e-9)
                )
                intensity[side_index, action_index] = (
                    fills[side_index, action_index]
                    + config.execution_prior_seconds * total_rate
                ) / (
                    exposure[side_index, action_index]
                    + config.execution_prior_seconds
                )
            # Enforce the paper's priority assumption lambda(best)<lambda(improve).
            intensity[side_index, 1] = np.maximum(
                intensity[side_index, 1], intensity[side_index, 0] * 1.001
            )
        intensities[mode] = intensity
        execution_audit[mode] = {
            "fills": fills.tolist(),
            "exposure_seconds": exposure.tolist(),
            "intensity_per_second": intensity.tolist(),
        }

    lifecycle_sum = sum(record.lifecycle_seconds_sum for record in records)
    lifecycle_count = sum(record.lifecycle_count for record in records)
    weights = np.asarray([record.lifecycle_count for record in records], dtype=np.float64)
    reference_price = float(
        np.average(
            [record.reference_price_hkd for record in records],
            weights=np.maximum(weights, 1.0),
        )
    )
    return GPInputs(
        transition_probabilities=rho,
        clock_intensity_per_second=clock,
        execution_intensity_per_second=intensities,
        calibration_dates=tuple(record.date for record in records),
        calibration_reference_price_hkd=reference_price,
        mean_lifecycle_seconds=float(lifecycle_sum / max(lifecycle_count, 1)),
        audit={
            "transition_counts": transition_counts.tolist(),
            "transition_probabilities": rho.tolist(),
            "clock_jumps": jumps.tolist(),
            "clock_exposure_seconds": clock_exposure.tolist(),
            "clock_intensity_per_second": clock.tolist(),
            "clock_buckets": [label for _, _, label in SESSION_BUCKETS],
            "execution": execution_audit,
            "mean_lifecycle_seconds": float(lifecycle_sum / max(lifecycle_count, 1)),
            "calibration_tick_hkd_by_day": {
                record.date: record.tick_hkd for record in records
            },
            "spread_state_capped_share_by_day": {
                record.date: record.capped_spread_share for record in records
            },
        },
    )


def solve_gp_policy(
    inputs: GPInputs,
    tick_hkd: float,
    fill_mode: str,
    clock_bucket: int,
    config: GPConfig,
    market_orders_enabled: bool = True,
) -> GPPolicy:
    """Solve the finite-state Bellman/QVI approximation backward in time."""

    if fill_mode not in FILL_MODES:
        raise ValueError(fill_mode)
    m = config.spread_states
    cap = config.max_inventory_lots
    inventories = np.arange(-cap, cap + 1, dtype=np.int16)
    q_count = len(inventories)
    n_steps = config.time_steps
    dt = config.dt_seconds
    fee_per_fill = (
        inputs.calibration_reference_price_hkd * LOT_SIZE * ALL_IN_FEE_RATE
    )
    generator = inputs.clock_intensity_per_second[clock_bucket] * (
        inputs.transition_probabilities - np.eye(m)
    )
    spread_transition = expm(generator * dt)
    spread_transition = np.clip(spread_transition, 0.0, 1.0)
    spread_transition /= spread_transition.sum(axis=1, keepdims=True)
    hazards = inputs.execution_intensity_per_second[fill_mode]

    values = np.zeros((n_steps + 1, m, q_count), dtype=np.float64)
    bid_action = np.zeros((n_steps + 1, m, q_count), dtype=np.int8)
    ask_action = np.zeros_like(bid_action)
    make_bid_action = np.zeros_like(bid_action)
    make_ask_action = np.zeros_like(bid_action)
    impulse_target = np.broadcast_to(
        np.arange(q_count, dtype=np.int16), (n_steps + 1, m, q_count)
    ).copy()

    for spread in range(m):
        half_spread = (spread + 1) * tick_hkd / 2.0
        taker_cost = half_spread * LOT_SIZE + fee_per_fill
        values[0, spread] = -np.abs(inventories) * taker_cost

    action_codes = (ACTION_NONE, ACTION_BEST, ACTION_IMPROVE)
    for step in range(1, n_steps + 1):
        next_by_current_spread = spread_transition @ values[step - 1]
        for spread in range(m):
            spread_ticks = spread + 1
            half_spread = spread_ticks * tick_hkd / 2.0
            make_values = np.full(q_count, -np.inf, dtype=np.float64)
            make_bids = np.zeros(q_count, dtype=np.int8)
            make_asks = np.zeros(q_count, dtype=np.int8)
            for q_index, inventory in enumerate(inventories):
                best_value = -np.inf
                best_bid = ACTION_NONE
                best_ask = ACTION_NONE
                for bid in action_codes:
                    if inventory >= cap and bid != ACTION_NONE:
                        continue
                    if spread_ticks == 1 and bid == ACTION_IMPROVE:
                        continue
                    for ask in action_codes:
                        if inventory <= -cap and ask != ACTION_NONE:
                            continue
                        if spread_ticks == 1 and ask == ACTION_IMPROVE:
                            continue
                        if bid == ACTION_NONE:
                            p_buy = 0.0
                            buy_gain = 0.0
                        else:
                            action_index = bid - 1
                            p_buy = 1.0 - np.exp(-hazards[0, action_index, spread] * dt)
                            buy_gain = (
                                half_spread
                                - (tick_hkd if bid == ACTION_IMPROVE else 0.0)
                            ) * LOT_SIZE - fee_per_fill
                        if ask == ACTION_NONE:
                            p_sell = 0.0
                            sell_gain = 0.0
                        else:
                            action_index = ask - 1
                            p_sell = 1.0 - np.exp(-hazards[1, action_index, spread] * dt)
                            sell_gain = (
                                half_spread
                                - (tick_hkd if ask == ACTION_IMPROVE else 0.0)
                            ) * LOT_SIZE - fee_per_fill
                        expected = 0.0
                        outcomes = (
                            ((1-p_buy)*(1-p_sell), 0, 0.0),
                            (p_buy*(1-p_sell), 1, buy_gain),
                            ((1-p_buy)*p_sell, -1, sell_gain),
                            (p_buy*p_sell, 0, buy_gain + sell_gain),
                        )
                        for probability, inventory_delta, gain in outcomes:
                            target = q_index + inventory_delta
                            if 0 <= target < q_count:
                                expected += probability * (
                                    next_by_current_spread[spread, target] + gain
                                )
                        running_penalty = (
                            config.inventory_penalty_gamma
                            * float(inventory * inventory)
                            * dt
                        )
                        expected -= running_penalty
                        if expected > best_value:
                            best_value = expected
                            best_bid = bid
                            best_ask = ask
                make_values[q_index] = best_value
                make_bids[q_index] = best_bid
                make_asks[q_index] = best_ask

            final_values = make_values.copy()
            targets = np.arange(q_count, dtype=np.int16)
            if market_orders_enabled:
                taker_cost = half_spread * LOT_SIZE + fee_per_fill
                zero = cap
                for q_index in range(zero + 1, q_count):
                    candidate = final_values[q_index - 1] - taker_cost
                    if candidate > final_values[q_index]:
                        final_values[q_index] = candidate
                        targets[q_index] = targets[q_index - 1]
                for q_index in range(zero - 1, -1, -1):
                    candidate = final_values[q_index + 1] - taker_cost
                    if candidate > final_values[q_index]:
                        final_values[q_index] = candidate
                        targets[q_index] = targets[q_index + 1]

            values[step, spread] = final_values
            impulse_target[step, spread] = targets
            make_bid_action[step, spread] = make_bids
            make_ask_action[step, spread] = make_asks
            for q_index in range(q_count):
                target = int(targets[q_index])
                bid_action[step, spread, q_index] = make_bids[target]
                ask_action[step, spread, q_index] = make_asks[target]

    return GPPolicy(
        values=values,
        bid_action=bid_action,
        ask_action=ask_action,
        impulse_target=impulse_target,
        make_bid_action=make_bid_action,
        make_ask_action=make_ask_action,
        config=config,
        tick_hkd=tick_hkd,
        fill_mode=fill_mode,
        clock_bucket=clock_bucket,
        market_orders_enabled=market_orders_enabled,
    )


def _action_name(code: int) -> str:
    return {ACTION_NONE: "none", ACTION_BEST: "best", ACTION_IMPROVE: "improve"}[
        int(code)
    ]


def simulate_gp_day(
    day: Any,
    trades: TradeData,
    policies: dict[int, GPPolicy] | None,
    config: GPConfig,
    fill_mode: str,
    strategy_name: str,
) -> dict[str, Any]:
    """Replay one day with either a GP policy or the constant-best benchmark."""

    tick = infer_tick_hkd(day)
    times = compact_time_to_day_ms(day.send_times)
    quote_indices = select_quote_indices(day, day.eligible, config.quote_horizon_snapshots)
    inventory = 0
    cash = 0.0
    fees = 0.0
    official_fees = 0.0
    maker_turnover = 0.0
    market_turnover = 0.0
    maker_fills = buy_fills = sell_fills = 0
    trade_price_fills = best_ask_fills = best_bid_fills = 0
    quoted_bid = quoted_ask = bid_cancels = ask_cancels = 0
    market_orders = market_order_lots = 0
    max_abs_inventory = 0
    equity_path = [0.0]
    action_counts = {
        "bid_none": 0, "bid_best": 0, "bid_improve": 0,
        "ask_none": 0, "ask_best": 0, "ask_improve": 0,
    }
    final_session_ms = max(int(times[index]) for index in day.eligible)

    def transact_market(target_inventory: int, best_bid: float, best_ask: float) -> None:
        nonlocal inventory, cash, fees, official_fees, market_turnover
        nonlocal market_orders, market_order_lots
        delta = target_inventory - inventory
        if delta == 0:
            return
        market_orders += 1
        market_order_lots += abs(delta)
        if delta > 0:
            notional = delta * LOT_SIZE * best_ask
            cash -= notional
        else:
            notional = -delta * LOT_SIZE * best_bid
            cash += notional
        market_turnover += notional
        fees += notional * ALL_IN_FEE_RATE
        official_fees += notional * OFFICIAL_FIXED_FEE_RATE
        inventory = target_inventory

    for index_value in quote_indices:
        index = int(index_value)
        expiry = index + config.quote_horizon_snapshots
        if expiry >= len(day.book) or day.segments[index] != day.segments[expiry]:
            continue
        best_ask = float(day.book[index, 0])
        best_bid = float(day.book[index, 2])
        spread_state = _spread_state(best_ask, best_bid, tick, config.spread_states)
        bucket = _bucket_index(int(times[index]))
        if policies is None:
            bid_code = ask_code = ACTION_BEST
        else:
            remaining = max((final_session_ms - int(times[index])) / 1000.0, 0.0)
            step = min(
                config.time_steps,
                max(1, int(np.ceil(remaining / config.dt_seconds))),
            )
            policy = policies[bucket]
            q_index = inventory + config.max_inventory_lots
            target_index = int(policy.impulse_target[step, spread_state, q_index])
            target_inventory = target_index - config.max_inventory_lots
            if target_inventory != inventory:
                transact_market(target_inventory, best_bid, best_ask)
                q_index = target_index
            bid_code = int(policy.make_bid_action[step, spread_state, q_index])
            ask_code = int(policy.make_ask_action[step, spread_state, q_index])

        if inventory >= config.max_inventory_lots:
            bid_code = ACTION_NONE
        if inventory <= -config.max_inventory_lots:
            ask_code = ACTION_NONE
        spread_ticks = max(1, int(round((best_ask - best_bid) / tick)))
        if spread_ticks <= 1:
            if bid_code == ACTION_IMPROVE:
                bid_code = ACTION_BEST
            if ask_code == ACTION_IMPROVE:
                ask_code = ACTION_BEST
        bid_name = _action_name(bid_code)
        ask_name = _action_name(ask_code)
        action_counts[f"bid_{bid_name}"] += 1
        action_counts[f"ask_{ask_name}"] += 1
        place_bid = bid_code != ACTION_NONE
        place_ask = ask_code != ACTION_NONE
        quoted_bid += int(place_bid)
        quoted_ask += int(place_ask)
        bid_quote = best_bid + (tick if bid_code == ACTION_IMPROVE else 0.0)
        ask_quote = best_ask - (tick if ask_code == ACTION_IMPROVE else 0.0)

        pending: list[tuple[int, str, float, str]] = []
        if place_bid:
            fill_time, source = _first_fill(
                day, trades, index, expiry, bid_quote, "buy", fill_mode,
                tick, config.latency_ms, times,
            )
            if fill_time is None:
                bid_cancels += 1
            else:
                pending.append((fill_time, "buy", bid_quote, str(source)))
        if place_ask:
            fill_time, source = _first_fill(
                day, trades, index, expiry, ask_quote, "sell", fill_mode,
                tick, config.latency_ms, times,
            )
            if fill_time is None:
                ask_cancels += 1
            else:
                pending.append((fill_time, "sell", ask_quote, str(source)))
        pending.sort(key=lambda item: (item[0], item[1]))
        for _, side, price, source in pending:
            notional = price * LOT_SIZE
            if side == "buy":
                cash -= notional
                inventory += 1
                buy_fills += 1
            else:
                cash += notional
                inventory -= 1
                sell_fills += 1
            maker_fills += 1
            maker_turnover += notional
            fees += notional * ALL_IN_FEE_RATE
            official_fees += notional * OFFICIAL_FIXED_FEE_RATE
            if source == "trade_price":
                trade_price_fills += 1
            elif source == "best_ask":
                best_ask_fills += 1
            elif source == "best_bid":
                best_bid_fills += 1
            else:
                raise AssertionError(source)
            max_abs_inventory = max(max_abs_inventory, abs(inventory))
        expiry_mid = float((day.book[expiry, 0] + day.book[expiry, 2]) / 2.0)
        equity_path.append(cash + inventory * LOT_SIZE * expiry_mid - fees)

    final_index = int(quote_indices[-1] + config.quote_horizon_snapshots)
    final_bid = float(day.book[final_index, 2])
    final_ask = float(day.book[final_index, 0])
    terminal_lots = abs(inventory)
    terminal_side = "none"
    policy_market_orders = market_orders
    policy_market_order_lots = market_order_lots
    if inventory > 0:
        terminal_side = "sell"
    elif inventory < 0:
        terminal_side = "buy"
    if inventory != 0:
        transact_market(0, final_bid, final_ask)
    gross_pnl = cash
    net_pnl = cash - fees
    equity_path.append(net_pnl)
    equity = np.asarray(equity_path, dtype=np.float64)
    drawdown = float(np.max(np.maximum.accumulate(equity) - equity))
    best_quote_fills = best_ask_fills + best_bid_fills
    if quoted_bid != buy_fills + bid_cancels:
        raise AssertionError("bid lifecycle mismatch")
    if quoted_ask != sell_fills + ask_cancels:
        raise AssertionError("ask lifecycle mismatch")
    if maker_fills != trade_price_fills + best_quote_fills:
        raise AssertionError("fill evidence mismatch")
    feeable_turnover = maker_turnover + market_turnover
    if not np.isclose(fees, feeable_turnover * ALL_IN_FEE_RATE, atol=1e-6):
        raise AssertionError("fee accounting mismatch")
    return {
        "date": str(day.date),
        "strategy": strategy_name,
        "fill_mode": fill_mode,
        "tick_hkd": tick,
        "quote_decisions": int(len(quote_indices)),
        "quoted_bid": quoted_bid,
        "quoted_ask": quoted_ask,
        "maker_fills": maker_fills,
        "buy_fills": buy_fills,
        "sell_fills": sell_fills,
        "trade_price_fills": trade_price_fills,
        "best_quote_fills": best_quote_fills,
        "best_ask_fills": best_ask_fills,
        "best_bid_fills": best_bid_fills,
        "bid_cancels": bid_cancels,
        "ask_cancels": ask_cancels,
        "expired_orders": bid_cancels + ask_cancels,
        "market_orders": market_orders,
        "market_order_lots": market_order_lots,
        "policy_market_orders": policy_market_orders,
        "policy_market_order_lots": policy_market_order_lots,
        "terminal_flatten_orders": int(terminal_lots > 0),
        "terminal_flatten_side": terminal_side,
        "terminal_flatten_lots": terminal_lots,
        "end_inventory_lots": inventory,
        "max_abs_inventory_lots": max_abs_inventory,
        "gross_pnl_hkd": gross_pnl,
        "official_only_fees_hkd": official_fees,
        "all_in_fees_hkd": fees,
        "net_pnl_official_only_hkd": gross_pnl - official_fees,
        "net_pnl_hkd": net_pnl,
        "maker_turnover_hkd": maker_turnover,
        "market_turnover_hkd": market_turnover,
        "feeable_turnover_hkd": feeable_turnover,
        "gross_bps_of_feeable_turnover": (
            gross_pnl / feeable_turnover * 10_000 if feeable_turnover else None
        ),
        "net_bps_of_feeable_turnover": (
            net_pnl / feeable_turnover * 10_000 if feeable_turnover else None
        ),
        "max_drawdown_hkd": drawdown,
        "action_counts": action_counts,
        "execution_cost": {
            "official_fixed_fee_bps_per_side": OFFICIAL_FIXED_FEE_RATE * 10_000,
            "brokerage_assumption_bps_per_side": (
                ALL_IN_FEE_RATE - OFFICIAL_FIXED_FEE_RATE
            ) * 10_000,
            "all_in_bps_per_side": ALL_IN_FEE_RATE * 10_000,
        },
    }


def policy_audit(policy: GPPolicy) -> dict[str, Any]:
    step = policy.config.time_steps
    center = policy.config.max_inventory_lots
    return {
        "fill_mode": policy.fill_mode,
        "clock_bucket": policy.clock_bucket,
        "market_orders_enabled": policy.market_orders_enabled,
        "tick_hkd": policy.tick_hkd,
        "initial_zero_inventory_actions_by_spread": [
            {
                "spread_ticks": spread + 1,
                "bid": _action_name(policy.bid_action[step, spread, center]),
                "ask": _action_name(policy.ask_action[step, spread, center]),
            }
            for spread in range(policy.config.spread_states)
        ],
        "near_terminal_market_targets_by_inventory_at_two_tick_spread": {
            str(inventory): int(
                policy.impulse_target[
                    1,
                    min(1, policy.config.spread_states - 1),
                    inventory + center,
                ]
                - center
            )
            for inventory in range(-center, center + 1)
        },
    }


def config_asdict(config: GPConfig) -> dict[str, Any]:
    return asdict(config)
