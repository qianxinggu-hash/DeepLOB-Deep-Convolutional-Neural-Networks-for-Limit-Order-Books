#!/usr/bin/env python3
"""Fee-aware single-quote diagnostic for rolling L2 and L3 price forecasts.

At each held-out quote, post one 100-share best-price maker order on the
forecast's favored side.  If the existing touch/through proxy fills during the
20-snapshot horizon, exit at the displayed opposite best quote at horizon end
and pay fees on both executions.  Each quote is independently flattened;
this is a diagnostic, not a deployable inventory-aware strategy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0906"))
sys.path.insert(0, str(ROOT / "0824"))

import gp_qvi_model as gp  # noqa: E402
from directional_market_maker import ALL_IN_FEE_RATE, LOT_SIZE, load_or_extract_trades  # noqa: E402
from run_multiday_strategy import load_day  # noqa: E402


def main() -> None:
    signal_dir = HERE / "output/l3_order_signal"
    result = json.loads((signal_dir / "results.json").read_text())
    dates = result["protocol"]["dates"][1:]
    rows = []
    for date in dates:
        book_path = ROOT / f"0824/data/july_2026/hk07709_{date}_mbo_tolerant_5m_s10.npz"
        trade_path = ROOT / f"0824/data/july_2026/hk07709_{date}_trades.npz"
        raw = Path.home() / f"Downloads/7709_202607/hk07709_{date}.csv"
        day = load_day(date, "tolerant_mbo", book_path)
        trades = load_or_extract_trades(raw, trade_path)
        with np.load(signal_dir / "predictions" / f"{date}.npz") as archive:
            indices = archive["quote_indices"].copy()
            predictions = {name: archive[f"prediction_{name}"].copy()
                           for name in ("l2", "l3_lifecycle", "l2_plus_l3_lifecycle",
                                        "l3_identity", "l2_plus_l3_identity")}
        times = gp.compact_time_to_day_ms(day.send_times)
        tick = gp.infer_tick_hkd(day)
        outcomes = {name: {mode: [] for mode in gp.FILL_MODES} for name in predictions}
        for place, raw_index in enumerate(indices):
            index = int(raw_index)
            expiry = index + 20
            if expiry >= len(day.book) or day.segments[index] != day.segments[expiry]:
                raise AssertionError(f"invalid quote horizon: {date} {index}")
            for mode in gp.FILL_MODES:
                side_results = {}
                for side in ("buy", "sell"):
                    quote_price = float(day.book[index, 2 if side == "buy" else 0])
                    fill_time, _ = gp._first_fill(
                        day, trades, index, expiry, quote_price, side, mode,
                        tick, 5, times,
                    )
                    if fill_time is None:
                        side_results[side] = None
                        continue
                    exit_price = float(day.book[expiry, 2 if side == "buy" else 0])
                    direction = 1 if side == "buy" else -1
                    side_results[side] = (
                        direction * (exit_price - quote_price) * LOT_SIZE
                        - (quote_price + exit_price) * LOT_SIZE * ALL_IN_FEE_RATE
                    )
                for name, prediction in predictions.items():
                    side = "buy" if prediction[place] >= 0 else "sell"
                    outcomes[name][mode].append(side_results[side])
        for name, by_mode in outcomes.items():
            for mode, values in by_mode.items():
                filled = np.asarray([v for v in values if v is not None], dtype=np.float64)
                rows.append({
                    "date": date, "signal": name, "fill_mode": mode,
                    "quote_attempts": len(values), "proxy_fills": len(filled),
                    "fill_rate": float(len(filled) / len(values)),
                    "net_hkd": float(filled.sum()),
                    "mean_net_hkd_per_fill": float(filled.mean()) if len(filled) else None,
                    "mean_net_hkd_per_attempt": float(filled.sum() / len(values)),
                })
        print(f"{date}: {len(indices)} quotes", flush=True)
    summary = []
    for name in ("l2", "l3_lifecycle", "l2_plus_l3_lifecycle",
                 "l3_identity", "l2_plus_l3_identity"):
        for mode in gp.FILL_MODES:
            group = [r for r in rows if r["signal"] == name and r["fill_mode"] == mode]
            n = sum(r["quote_attempts"] for r in group)
            fills = sum(r["proxy_fills"] for r in group)
            pnl = sum(r["net_hkd"] for r in group)
            summary.append({
                "signal": name, "fill_mode": mode, "quote_attempts": n,
                "proxy_fills": fills, "fill_rate": fills / n,
                "net_hkd": pnl, "mean_net_hkd_per_fill": pnl / fills if fills else None,
                "mean_net_hkd_per_attempt": pnl / n,
                "days_net_positive": sum(r["net_hkd"] > 0 for r in group),
            })
    output = {
        "method": "single-lot favored-side best-price maker quote, 5ms entry latency, 20-snapshot expiry, immediate displayed-best exit and two-sided fees",
        "limitations": [
            "Touch/through are proxy fills and do not model our queue position.",
            "Exit at the displayed best price ignores market-exit latency, depth and slippage.",
            "Independent quotes do not model inventory constraints or QVI action selection.",
        ],
        "summary": summary,
        "daily": rows,
    }
    (signal_dir / "fill_diagnostic.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
