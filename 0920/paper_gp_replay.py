#!/usr/bin/env python3
"""Signed-inventory GP replay plus long/short exposure bookkeeping.

Quote decisions, cashflows, fees and event matching are unchanged. The
inventory clock aligns policy market events and terminal flattening to
snapshot+latency, matching the inherited order expiry clock. Risk diagnostics
include wall-clock gaps, including lunch, while inventory is held. Negative
inventory is explicitly allowed down to the symmetric GP state-space bound.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import gp_qvi_model as gp
from gp_qvi_model import (GPConfig, GPPolicy, TradeData, infer_tick_hkd,
    compact_time_to_day_ms, select_quote_indices, _spread_state, _bucket_index,
    ACTION_NONE, ACTION_BEST, ACTION_IMPROVE, _action_name, LOT_SIZE,
    ALL_IN_FEE_RATE, OFFICIAL_FIXED_FEE_RATE)


@dataclass
class InventoryExposure:
    start_ms: int
    last_ms: int = field(init=False)
    absolute_ms: int = 0
    square_ms: int = 0
    time_by_inventory_ms: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self.last_ms = self.start_ms

    def advance(self, time_ms: int, inventory: int) -> None:
        elapsed = int(time_ms) - self.last_ms
        if elapsed < 0:
            raise ValueError("Inventory events must be chronological")
        q = int(inventory)
        self.absolute_ms += elapsed * abs(q)
        self.square_ms += elapsed * q * q
        self.time_by_inventory_ms[q] = self.time_by_inventory_ms.get(q, 0) + elapsed
        self.last_ms = int(time_ms)

    def metrics(self) -> dict:
        duration_ms = self.last_ms - self.start_ms
        negative_time_ms = sum(ms for q, ms in self.time_by_inventory_ms.items() if q < 0)
        positive_time_ms = sum(ms for q, ms in self.time_by_inventory_ms.items() if q > 0)
        short_lot_ms = sum(-q * ms for q, ms in self.time_by_inventory_ms.items() if q < 0)
        long_lot_ms = sum(q * ms for q, ms in self.time_by_inventory_ms.items() if q > 0)
        return {
            "exposure_duration_seconds": duration_ms / 1000.0,
            "absolute_inventory_lot_seconds": self.absolute_ms / 1000.0,
            "squared_inventory_lot2_seconds": self.square_ms / 1000.0,
            "time_mean_abs_inventory_lots": self.absolute_ms / max(duration_ms, 1),
            "time_rms_inventory_lots": float(np.sqrt(self.square_ms / max(duration_ms, 1))),
            "nonzero_inventory_time_share": 1.0 - self.time_by_inventory_ms.get(0, 0) / max(duration_ms, 1),
            "negative_inventory_time_seconds": negative_time_ms / 1000.0,
            "positive_inventory_time_seconds": positive_time_ms / 1000.0,
            "short_inventory_lot_seconds": short_lot_ms / 1000.0,
            "long_inventory_lot_seconds": long_lot_ms / 1000.0,
            "negative_inventory_time_share": negative_time_ms / max(duration_ms, 1),
            "positive_inventory_time_share": positive_time_ms / max(duration_ms, 1),
            "time_at_inventory_lots_seconds": {str(q): ms / 1000.0 for q, ms in sorted(self.time_by_inventory_ms.items())},
        }


def simulate_gp_day(
    day: Any,
    trades: TradeData,
    policies: dict[int, GPPolicy] | None,
    config: GPConfig,
    fill_mode: str,
    strategy_name: str,
    event_log: list[dict[str, Any]] | None = None,
    decision_log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replay original cashflows; optional logs observe, but do not alter, decisions."""

    tick = infer_tick_hkd(day)
    times = compact_time_to_day_ms(day.send_times)
    quote_indices = select_quote_indices(day, day.eligible, config.quote_horizon_snapshots)
    inventory = 0
    min_inventory = 0
    max_inventory = 0
    exposure = InventoryExposure(int(times[int(quote_indices[0])]) + config.latency_ms)
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

    def record_event(kind: str, time_ms: int, signed_lots: int, price: float,
                     quote_mid: float, quote_index: int | None, source: str) -> None:
        if event_log is None:
            return
        snapshot = max(0, int(np.searchsorted(times, time_ms, side="right")) - 1)
        event_log.append({
            "date": str(day.date), "kind": kind, "time_ms": int(time_ms),
            "signed_lots": int(signed_lots), "price_hkd": float(price),
            "quote_mid_hkd": float(quote_mid), "quote_index": quote_index,
            "inventory_before_lots": int(inventory), "source": source,
            "mid_asof_hkd": float((float(day.book[snapshot, 0]) + float(day.book[snapshot, 2])) / 2),
            "snapshot_index": snapshot, "snapshot_age_ms": int(time_ms - times[snapshot]),
            "segment": int(day.segments[snapshot]),
            "fee_hkd": abs(signed_lots) * LOT_SIZE * price * ALL_IN_FEE_RATE,
        })

    def transact_market(target_inventory: int, best_bid: float, best_ask: float, event_time_ms: int,
                        kind: str = "market") -> None:
        nonlocal inventory, cash, fees, official_fees, market_turnover
        nonlocal market_orders, market_order_lots
        nonlocal min_inventory, max_inventory
        delta = target_inventory - inventory
        if delta == 0:
            return
        record_event(kind, event_time_ms, delta, best_ask if delta > 0 else best_bid,
                     (best_bid + best_ask) / 2, None, "opposite_best")
        exposure.advance(event_time_ms, inventory)
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
        min_inventory = min(min_inventory, inventory)
        max_inventory = max(max_inventory, inventory)

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
                transact_market(target_inventory, best_bid, best_ask, int(times[index]) + config.latency_ms)
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
            fill_time, source = gp._first_fill(
                day, trades, index, expiry, bid_quote, "buy", fill_mode,
                tick, config.latency_ms, times,
            )
            if fill_time is None:
                bid_cancels += 1
            else:
                pending.append((fill_time, "buy", bid_quote, str(source)))
        if place_ask:
            fill_time, source = gp._first_fill(
                day, trades, index, expiry, ask_quote, "sell", fill_mode,
                tick, config.latency_ms, times,
            )
            if fill_time is None:
                ask_cancels += 1
            else:
                pending.append((fill_time, "sell", ask_quote, str(source)))
        pending.sort(key=lambda item: (item[0], item[1]))
        if decision_log is not None:
            decision_log.append({
                "date": str(day.date), "index": index, "expiry": expiry,
                "time_ms": int(times[index]), "bucket": bucket,
                "spread_state": spread_state, "spread_ticks": spread_ticks,
                "bid_action": int(bid_code), "ask_action": int(ask_code),
                "buy_filled": any(item[1] == "buy" for item in pending),
                "sell_filled": any(item[1] == "sell" for item in pending),
                "lifetime_seconds": float((times[expiry] - times[index]) / 1000),
                "inventory_lots": int(inventory),
            })
        for fill_time_ms, side, price, source in pending:
            record_event("maker", fill_time_ms, 1 if side == "buy" else -1, price,
                         (best_bid + best_ask) / 2, index, source)
            exposure.advance(fill_time_ms, inventory)
            notional = price * LOT_SIZE
            if side == "buy":
                cash -= notional
                inventory += 1
                buy_fills += 1
            else:
                cash += notional
                inventory -= 1
                sell_fills += 1
            min_inventory = min(min_inventory, inventory)
            max_inventory = max(max_inventory, inventory)
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
    exposure.advance(int(times[final_index]) + config.latency_ms, inventory)
    terminal_lots = abs(inventory)
    terminal_side = "none"
    policy_market_orders = market_orders
    policy_market_order_lots = market_order_lots
    if inventory > 0:
        terminal_side = "sell"
    elif inventory < 0:
        terminal_side = "buy"
    if inventory != 0:
        transact_market(0, final_bid, final_ask, int(times[final_index]) + config.latency_ms,
                        kind="terminal")
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
        "min_inventory_lots": min_inventory,
        "max_inventory_lots": max_inventory,
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
        **exposure.metrics(),
        "realized_inventory_penalty_hkd": config.inventory_penalty_gamma * exposure.square_ms / 1000.0,
        "execution_cost": {
            "official_fixed_fee_bps_per_side": OFFICIAL_FIXED_FEE_RATE * 10_000,
            "brokerage_assumption_bps_per_side": (
                ALL_IN_FEE_RATE - OFFICIAL_FIXED_FEE_RATE
            ) * 10_000,
            "all_in_bps_per_side": ALL_IN_FEE_RATE * 10_000,
        },
    }
