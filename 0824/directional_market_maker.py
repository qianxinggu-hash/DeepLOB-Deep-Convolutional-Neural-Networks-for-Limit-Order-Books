#!/usr/bin/env python3
"""Direction-aware, discrete AS/GP-style market-making components.

The prediction is used only to move the reservation price.  Limit-order fills
are inferred from subsequent trade prints under two queue assumptions:
``touch`` (optimistic: a trade at our price fills us) and ``through``
(conservative: price must trade one tick beyond us).  The exact queue position
is not present in the source feed, so neither mode is presented as an exact
exchange replay.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_improved_experiment import causal_features, make_labels, metrics_from_prediction
from lob_reconstruction import ASK_EXECUTION, BID_EXECUTION, hk_session_from_utc_ms


TICK_HKD = 0.02
LOT_SIZE = 100
LATENCY_MS = 5
OFFICIAL_FIXED_FEE_RATE = 0.000127  # 1.27 bps per execution side
BROKERAGE_ASSUMPTION_RATE = 0.000150  # 1.50 bps per execution side
ALL_IN_FEE_RATE = OFFICIAL_FIXED_FEE_RATE + BROKERAGE_ASSUMPTION_RATE


@dataclass(frozen=True)
class QuoteParameters:
    base_distance_ticks: int
    signal_strength: float
    inventory_skew_ticks: float
    max_inventory_lots: int
    quote_horizon_snapshots: int = 20


@dataclass(frozen=True)
class ASParameters:
    """Finite-horizon Avellaneda--Stoikov parameters in tick/lot units.

    ``horizon_variance_ticks2`` is the validation-period variance of the
    mid-price change over one quote lifetime.  ``risk_aversion_per_tick`` and
    ``order_decay_per_tick`` therefore operate in the same discrete tick unit.
    """

    risk_aversion_per_tick: float
    order_decay_per_tick: float
    horizon_variance_ticks2: float
    max_inventory_lots: int
    quote_horizon_snapshots: int = 20


@dataclass
class TradeData:
    trade_time_ms: np.ndarray
    send_time_ms: np.ndarray
    price_hkd: np.ndarray
    quantity: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class AlphaCalibrator:
    intercept_ticks: float
    slope_ticks: float
    clip_ticks: float
    samples: int
    correlation: float

    def transform(self, score: np.ndarray) -> np.ndarray:
        values = self.intercept_ticks + self.slope_ticks * score
        return np.clip(values, -self.clip_ticks, self.clip_ticks)


def compact_time_to_day_ms(values: np.ndarray | int) -> np.ndarray:
    """Convert YYYYMMDDhhmmssfff integers to milliseconds since midnight."""

    compact = np.asarray(values, dtype=np.int64) % 1_000_000_000
    hour = compact // 10_000_000
    minute = (compact // 100_000) % 100
    second = (compact // 1_000) % 100
    millisecond = compact % 1_000
    return ((hour * 60 + minute) * 60 + second) * 1_000 + millisecond


def _price_to_hkd(raw: str) -> float:
    return float(raw) if "." in raw else int(raw) / 1_000.0


def load_or_extract_trades(
    csv_path: Path,
    cache_path: Path,
    security_id: int = 7709,
) -> TradeData:
    """Extract continuous-session trade prints and cache compact arrays."""

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            return TradeData(
                trade_time_ms=archive["trade_time_ms"].astype(np.int64, copy=False),
                send_time_ms=archive["send_time_ms"].astype(np.int64, copy=False),
                price_hkd=archive["price_hkd"].astype(np.float64, copy=False),
                quantity=archive["quantity"].astype(np.int64, copy=False),
                metadata=json.loads(str(archive["metadata"])),
            )

    trade_times: list[int] = []
    send_times: list[int] = []
    prices: list[float] = []
    quantities: list[int] = []
    rows = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            rows += 1
            if int(row["SecurityId"]) != security_id or int(row["MsgType"]) != 50:
                continue
            send_time = int(row["SendTime"])
            trade_time = int(row["TradeTime"] or send_time)
            trade_ms = int(compact_time_to_day_ms(trade_time))
            if not (34_200_000 <= trade_ms < 43_200_000 or 46_800_000 <= trade_ms < 57_600_000):
                continue
            trade_times.append(trade_ms)
            send_times.append(int(compact_time_to_day_ms(send_time)))
            prices.append(_price_to_hkd(row["Price"]))
            quantities.append(int(row["Quantity"]))

    order = np.lexsort((np.asarray(send_times), np.asarray(trade_times)))
    trade_time_array = np.asarray(trade_times, dtype=np.int64)[order]
    send_time_array = np.asarray(send_times, dtype=np.int64)[order]
    price_array = np.asarray(prices, dtype=np.float64)[order]
    quantity_array = np.asarray(quantities, dtype=np.int64)[order]
    metadata: dict[str, object] = {
        "source": str(csv_path.resolve()),
        "source_rows_scanned": rows,
        "trade_prints": int(len(price_array)),
        "security_id": security_id,
        "time_basis": "TradeTime; SendTime retained for diagnostics",
        "continuous_sessions": ["09:30-12:00", "13:00-16:00"],
        "price_unit": "HKD",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        trade_time_ms=trade_time_array,
        send_time_ms=send_time_array,
        price_hkd=price_array,
        quantity=quantity_array,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    return TradeData(
        trade_time_ms=trade_time_array,
        send_time_ms=send_time_array,
        price_hkd=price_array,
        quantity=quantity_array,
        metadata=metadata,
    )


def load_or_extract_mbp_executions(
    npz_path: Path,
    cache_path: Path,
) -> TradeData:
    """Extract trade executions from the absolute MBP event stream."""

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            return TradeData(
                trade_time_ms=archive["trade_time_ms"].astype(np.int64, copy=False),
                send_time_ms=archive["send_time_ms"].astype(np.int64, copy=False),
                price_hkd=archive["price_hkd"].astype(np.float64, copy=False),
                quantity=archive["quantity"].astype(np.int64, copy=False),
                metadata=json.loads(str(archive["metadata"])),
            )

    with np.load(npz_path, allow_pickle=False) as archive:
        events = archive["data"]
    execution_mask = np.isin(
        events["ev"].astype(np.uint64, copy=False),
        np.asarray([ASK_EXECUTION, BID_EXECUTION], dtype=np.uint64),
    )
    selected = events[execution_mask]
    epoch_ms = selected["exch_ts"].astype(np.int64, copy=False) // 1_000_000
    session_mask = np.fromiter(
        (hk_session_from_utc_ms(int(value)) > 0 for value in epoch_ms),
        dtype=bool,
        count=len(epoch_ms),
    )
    selected = selected[session_mask]
    epoch_ms = epoch_ms[session_mask]
    day_ms = (epoch_ms + 8 * 60 * 60 * 1_000) % (24 * 60 * 60 * 1_000)
    event_codes = selected["ev"].astype(np.uint64, copy=False)
    metadata: dict[str, object] = {
        "source": str(npz_path.resolve()),
        "source_kind": "MBP execution events",
        "trade_prints": int(len(selected)),
        "ask_execution_prints": int(np.sum(event_codes == ASK_EXECUTION)),
        "bid_execution_prints": int(np.sum(event_codes == BID_EXECUTION)),
        "time_basis": "exch_ts converted to Hong Kong milliseconds since midnight",
        "continuous_sessions": ["09:30-12:00", "13:00-16:00"],
        "price_unit": "HKD",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        trade_time_ms=day_ms.astype(np.int64, copy=False),
        send_time_ms=day_ms.astype(np.int64, copy=False),
        price_hkd=selected["px"].astype(np.float64, copy=False),
        quantity=selected["qty"].astype(np.int64, copy=False),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    return TradeData(
        trade_time_ms=day_ms.astype(np.int64, copy=False),
        send_time_ms=day_ms.astype(np.int64, copy=False),
        price_hkd=selected["px"].astype(np.float64, copy=False),
        quantity=selected["qty"].astype(np.int64, copy=False),
        metadata=metadata,
    )


def classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1_000,
            solver="lbfgs",
            random_state=20260824,
        ),
    )


def fit_direction_classifier(
    train_days: list[object],
    train_indices: list[np.ndarray],
    stationary_share: float = 1.0 / 3.0,
) -> tuple[object, float]:
    train_returns = np.concatenate(
        [day.returns[index] for day, index in zip(train_days, train_indices)]
    )
    alpha = float(np.quantile(np.abs(train_returns), stationary_share))
    features = np.concatenate(
        [causal_features(day.book, index) for day, index in zip(train_days, train_indices)]
    )
    labels = np.concatenate(
        [make_labels(day.returns, index, alpha) for day, index in zip(train_days, train_indices)]
    )
    estimator = classifier()
    estimator.fit(features, labels)
    return estimator, alpha


def predict_direction(
    estimator: object,
    alpha: float,
    day: object,
    indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    features = causal_features(day.book, indices)
    probabilities = estimator.predict_proba(features)
    classes = estimator.named_steps["logisticregression"].classes_
    positions = {int(value): position for position, value in enumerate(classes)}
    score = probabilities[:, positions[2]] - probabilities[:, positions[0]]
    prediction = classes[np.argmax(probabilities, axis=1)]
    truth = make_labels(day.returns, indices, alpha)
    metrics = metrics_from_prediction(truth, prediction.astype(np.int64))
    metrics["alpha"] = alpha
    metrics["samples"] = int(len(indices))
    metrics["mean_probability_edge"] = float(np.mean(np.abs(score)))
    return score.astype(np.float64), metrics


def future_mid_move_ticks(day: object, indices: np.ndarray, horizon: int = 20) -> np.ndarray:
    mids = (day.book[:, 0].astype(np.float64) + day.book[:, 2]) / 2.0
    future = indices + horizon
    valid = future < len(mids)
    if not np.all(valid) or not np.all(day.segments[future] == day.segments[indices]):
        raise ValueError("future move crosses a day/session boundary")
    return (mids[future] - mids[indices]) / TICK_HKD


def fit_alpha_calibrator(
    score: np.ndarray,
    future_ticks: np.ndarray,
    clip_ticks: float = 2.0,
) -> AlphaCalibrator:
    """Fit a train/validation-only linear map from class edge to expected ticks."""

    if len(score) != len(future_ticks) or len(score) < 2:
        raise ValueError("insufficient calibration observations")
    target = np.clip(future_ticks.astype(np.float64), -5.0, 5.0)
    centered = score - np.mean(score)
    variance = float(centered @ centered)
    slope = float(centered @ (target - np.mean(target)) / variance) if variance else 0.0
    slope = max(0.0, slope)  # a validation failure switches alpha off; it never inverts it
    intercept = float(np.mean(target) - slope * np.mean(score))
    correlation = float(np.corrcoef(score, target)[0, 1]) if np.std(score) and np.std(target) else 0.0
    return AlphaCalibrator(intercept, slope, clip_ticks, int(len(score)), correlation)


def select_quote_indices(
    day: object,
    eligible: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Select non-overlapping quote starts within each continuous session."""

    selected: list[int] = []
    for segment in np.unique(day.segments[eligible]):
        candidates = eligible[day.segments[eligible] == segment]
        next_allowed = -1
        for value in candidates:
            index = int(value)
            if index < next_allowed or index + horizon >= len(day.book):
                continue
            if day.segments[index + horizon] != segment:
                continue
            selected.append(index)
            next_allowed = index + horizon
    return np.asarray(selected, dtype=np.int64)


def _rounded_tick(value: float) -> int:
    return int(math.floor(value + 0.5))


def quote_prices(
    best_bid_hkd: float,
    best_ask_hkd: float,
    alpha_ticks: float,
    inventory_lots: int,
    parameters: QuoteParameters,
) -> tuple[float, float, float]:
    best_bid_tick = _rounded_tick(best_bid_hkd / TICK_HKD)
    best_ask_tick = _rounded_tick(best_ask_hkd / TICK_HKD)
    inventory_shift = -parameters.inventory_skew_ticks * (
        inventory_lots / parameters.max_inventory_lots
    )
    center_shift = parameters.signal_strength * alpha_ticks + inventory_shift
    bid_tick = _rounded_tick(
        best_bid_tick - parameters.base_distance_ticks + center_shift
    )
    ask_tick = _rounded_tick(
        best_ask_tick + parameters.base_distance_ticks + center_shift
    )
    bid_tick = min(bid_tick, best_ask_tick - 1)
    ask_tick = max(ask_tick, best_bid_tick + 1)
    if bid_tick >= ask_tick:
        raise AssertionError("crossed quote")
    return bid_tick * TICK_HKD, ask_tick * TICK_HKD, center_shift


def classic_as_quote_prices(
    best_bid_hkd: float,
    best_ask_hkd: float,
    inventory_lots: int,
    parameters: ASParameters,
) -> tuple[float, float, float]:
    """Return passive classic AS quotes using a fixed quote-lifetime horizon.

    In tick units, the reservation price and total spread are

        r = mid - q * gamma * sigma_h^2
        delta_a + delta_b = gamma * sigma_h^2
                            + 2/gamma * log(1 + gamma/k).

    Quotes are rounded outwards and clipped so they never cross the current
    opposite best price.
    """

    best_bid_tick = _rounded_tick(best_bid_hkd / TICK_HKD)
    best_ask_tick = _rounded_tick(best_ask_hkd / TICK_HKD)
    mid_tick = (best_bid_tick + best_ask_tick) / 2.0
    gamma = parameters.risk_aversion_per_tick
    decay = parameters.order_decay_per_tick
    variance = max(parameters.horizon_variance_ticks2, 0.0)
    inventory_shift = -inventory_lots * gamma * variance
    reservation_tick = mid_tick + inventory_shift
    total_spread_ticks = gamma * variance + 2.0 / gamma * math.log1p(gamma / decay)
    half_spread = total_spread_ticks / 2.0
    bid_tick = int(math.floor(reservation_tick - half_spread))
    ask_tick = int(math.ceil(reservation_tick + half_spread))
    bid_tick = min(bid_tick, best_ask_tick - 1)
    ask_tick = max(ask_tick, best_bid_tick + 1)
    if bid_tick >= ask_tick:
        raise AssertionError("crossed AS quote")
    return bid_tick * TICK_HKD, ask_tick * TICK_HKD, inventory_shift


def simulate_market_maker(
    day: object,
    trades: TradeData,
    quote_indices: np.ndarray,
    alpha_ticks: np.ndarray,
    parameters: QuoteParameters | ASParameters,
    fill_mode: str,
    strategy_name: str,
    record_events: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if fill_mode not in {"touch", "through"}:
        raise ValueError(fill_mode)
    if len(quote_indices) != len(alpha_ticks):
        raise ValueError("quote/signal length mismatch")
    quote_times = compact_time_to_day_ms(day.send_times)
    inventory = 0
    cash = 0.0
    fees = 0.0
    maker_fills = 0
    buy_fills = 0
    sell_fills = 0
    quoted_bid = 0
    quoted_ask = 0
    max_abs_inventory = 0
    maker_turnover = 0.0
    flatten_turnover = 0.0
    equity_path: list[float] = [0.0]
    events: list[dict[str, object]] = []

    for quote_number, (index, alpha_value) in enumerate(zip(quote_indices, alpha_ticks)):
        index = int(index)
        end_index = index + parameters.quote_horizon_snapshots
        if end_index >= len(day.book) or day.segments[index] != day.segments[end_index]:
            continue
        best_ask = float(day.book[index, 0])
        best_bid = float(day.book[index, 2])
        if isinstance(parameters, ASParameters):
            bid_quote, ask_quote, center_shift = classic_as_quote_prices(
                best_bid, best_ask, inventory, parameters
            )
        else:
            bid_quote, ask_quote, center_shift = quote_prices(
                best_bid, best_ask, float(alpha_value), inventory, parameters
            )
        place_bid = inventory < parameters.max_inventory_lots
        place_ask = inventory > -parameters.max_inventory_lots
        quoted_bid += int(place_bid)
        quoted_ask += int(place_ask)
        start_ms = int(quote_times[index]) + LATENCY_MS
        end_ms = int(quote_times[end_index]) + LATENCY_MS
        left = int(np.searchsorted(trades.trade_time_ms, start_ms, side="right"))
        right = int(np.searchsorted(trades.trade_time_ms, end_ms, side="right"))
        prices = trades.price_hkd[left:right]
        if fill_mode == "touch":
            bid_hits = np.flatnonzero(prices <= bid_quote + 1e-9) if place_bid else np.empty(0, dtype=np.int64)
            ask_hits = np.flatnonzero(prices >= ask_quote - 1e-9) if place_ask else np.empty(0, dtype=np.int64)
        else:
            bid_hits = np.flatnonzero(prices <= bid_quote - TICK_HKD + 1e-9) if place_bid else np.empty(0, dtype=np.int64)
            ask_hits = np.flatnonzero(prices >= ask_quote + TICK_HKD - 1e-9) if place_ask else np.empty(0, dtype=np.int64)
        pending: list[tuple[int, str, float]] = []
        if len(bid_hits):
            pending.append((int(bid_hits[0]), "buy", bid_quote))
        if len(ask_hits):
            pending.append((int(ask_hits[0]), "sell", ask_quote))
        pending.sort(key=lambda item: item[0])
        for local_trade, side, execution_price in pending:
            if side == "buy":
                cash -= execution_price * LOT_SIZE
                inventory += 1
                buy_fills += 1
            else:
                cash += execution_price * LOT_SIZE
                inventory -= 1
                sell_fills += 1
            notional = execution_price * LOT_SIZE
            maker_turnover += notional
            fees += notional * ALL_IN_FEE_RATE
            maker_fills += 1
            max_abs_inventory = max(max_abs_inventory, abs(inventory))
            if record_events:
                events.append(
                    {
                        "strategy": strategy_name,
                        "fill_mode": fill_mode,
                        "date": day.date,
                        "quote_number": quote_number,
                        "snapshot_index": index,
                        "signal_time": int(day.send_times[index]),
                        "trade_time_ms": int(trades.trade_time_ms[left + local_trade]),
                        "side": side,
                        "price_hkd": execution_price,
                        "inventory_after_lots": inventory,
                        "alpha_ticks": float(alpha_value),
                        "center_shift_ticks": center_shift,
                        "fee_hkd": notional * ALL_IN_FEE_RATE,
                    }
                )
        mid = (best_bid + best_ask) / 2.0
        equity_path.append(cash + inventory * LOT_SIZE * mid - fees)

    final_index = int(quote_indices[-1] + parameters.quote_horizon_snapshots)
    final_bid = float(day.book[final_index, 2])
    final_ask = float(day.book[final_index, 0])
    flatten_side = "none"
    flatten_lots = abs(inventory)
    if inventory > 0:
        flatten_side = "sell"
        notional = inventory * LOT_SIZE * final_bid
        flatten_turnover = notional
        cash += notional
        fees += notional * ALL_IN_FEE_RATE
    elif inventory < 0:
        flatten_side = "buy"
        notional = -inventory * LOT_SIZE * final_ask
        flatten_turnover = notional
        cash -= notional
        fees += notional * ALL_IN_FEE_RATE
    inventory = 0
    gross_pnl = cash
    net_pnl = gross_pnl - fees
    equity_path.append(net_pnl)
    equity = np.asarray(equity_path, dtype=np.float64)
    running_max = np.maximum.accumulate(equity)
    max_drawdown_hkd = float(np.max(running_max - equity))
    feeable_turnover = maker_turnover + flatten_turnover
    official_only_fees = feeable_turnover * OFFICIAL_FIXED_FEE_RATE
    all_in_fees = feeable_turnover * ALL_IN_FEE_RATE
    if not np.isclose(fees, all_in_fees, atol=1e-7):
        raise AssertionError("fee accounting mismatch")
    result: dict[str, object] = {
        "strategy": strategy_name,
        "fill_mode": fill_mode,
        "date": day.date,
        "quotes": int(len(quote_indices)),
        "quoted_bid": quoted_bid,
        "quoted_ask": quoted_ask,
        "maker_fills": maker_fills,
        "buy_fills": buy_fills,
        "sell_fills": sell_fills,
        "fill_rate_per_side_quote": float(maker_fills / max(quoted_bid + quoted_ask, 1)),
        "max_abs_inventory_lots": max_abs_inventory,
        "gross_pnl_hkd": gross_pnl,
        "official_only_fees_hkd": official_only_fees,
        "fees_hkd": all_in_fees,
        "net_pnl_official_only_hkd": gross_pnl - official_only_fees,
        "net_pnl_hkd": net_pnl,
        "maker_turnover_hkd": maker_turnover,
        "feeable_turnover_hkd": feeable_turnover,
        "gross_bps_of_feeable_turnover": float(gross_pnl / feeable_turnover * 10_000) if feeable_turnover else None,
        "net_bps_of_maker_turnover": float(net_pnl / maker_turnover * 10_000) if maker_turnover else None,
        "break_even_fee_bps_per_execution_side": float(gross_pnl / feeable_turnover * 10_000) if feeable_turnover else None,
        "max_drawdown_hkd": max_drawdown_hkd,
        "end_flatten_side": flatten_side,
        "end_flatten_lots": flatten_lots,
        "parameters": asdict(parameters),
        "execution_cost": {
            "official_fixed_fee_bps_per_side": OFFICIAL_FIXED_FEE_RATE * 10_000,
            "brokerage_assumption_bps_per_side": BROKERAGE_ASSUMPTION_RATE * 10_000,
            "all_in_bps_per_side": ALL_IN_FEE_RATE * 10_000,
        },
    }
    return result, events


def tuning_score(result: dict[str, object], minimum_fills: int = 20) -> float:
    if int(result["maker_fills"]) < minimum_fills:
        return -1e18
    return float(result["net_pnl_hkd"]) - 0.25 * float(result["max_drawdown_hkd"])
