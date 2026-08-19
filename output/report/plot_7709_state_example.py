#!/usr/bin/env python3
"""Extract and plot two adjacent reconstructed HK 7709 LOB states."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, FormatStrFormatter


ROOT = Path(__file__).resolve().parents[2]
NPZ_PATH = ROOT / "data/processed/hk07709_2026-07-09_lob.npz"
JSON_PATH = ROOT / "output/results/7709_state_example.json"
PNG_PATH = ROOT / "output/report/7709_two_states_depth.png"
STATE_INDEX = 27_802


def format_send_time(value: int) -> str:
    parsed = datetime.strptime(str(value), "%Y%m%d%H%M%S%f")
    return parsed.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def milliseconds_between(left: int, right: int) -> float:
    fmt = "%Y%m%d%H%M%S%f"
    start = datetime.strptime(str(left), fmt)
    end = datetime.strptime(str(right), fmt)
    return (end - start).total_seconds() * 1_000


archive = np.load(NPZ_PATH, allow_pickle=False)
features = archive["features"].astype(np.float64)
send_times = archive["send_times"].astype(np.int64)
metadata = json.loads(str(archive["metadata"]))
scale = float(metadata["feature_scale"])

indices = (STATE_INDEX, STATE_INDEX + 1)
expected_times = (20260709101008341, 20260709101008372)
actual_times = tuple(int(send_times[index]) for index in indices)
if actual_times != expected_times:
    raise RuntimeError(f"Unexpected source states: {actual_times}")

rows: list[dict[str, object]] = []
states: list[dict[str, object]] = []
for state_order, (index, state_name) in enumerate(zip(indices, ("t", "t+1")), 1):
    values = features[index].reshape(10, 4)
    time_value = int(send_times[index])
    states.append(
        {
            "state": state_name,
            "state_order": state_order,
            "snapshot_index": index,
            "send_time": time_value,
            "send_time_readable": format_send_time(time_value),
            "model_vector": features[index].round(8).tolist(),
        }
    )
    for level_index, (ask_price, ask_size, bid_price, bid_size) in enumerate(values, 1):
        rows.append(
            {
                "row_order": (state_order - 1) * 10 + level_index,
                "state": state_name,
                "send_time": time_value,
                "time": format_send_time(time_value).split(" ", 1)[1],
                "level": f"L{level_index}",
                "ask_price_hkd": round(float(ask_price), 4),
                "ask_size_units": int(round(float(ask_size) * scale)),
                "bid_price_hkd": round(float(bid_price), 4),
                "bid_size_units": int(round(float(bid_size) * scale)),
                "model_input_slice": (
                    f"[{ask_price:.4f}, {ask_size:.4f}, "
                    f"{bid_price:.4f}, {bid_size:.4f}]"
                ),
            }
        )

payload = {
    "description": "Two adjacent reconstructed HK 7709 ten-level snapshots used to explain one DeepLOB state.",
    "source": str(NPZ_PATH),
    "security_id": int(metadata["security_id"]),
    "feature_order": metadata["feature_order"],
    "feature_scale": scale,
    "snapshot_every_book_update_groups": int(
        metadata["snapshot_every_book_update_groups"]
    ),
    "delta_milliseconds": milliseconds_between(*actual_times),
    "states": states,
    "rows": rows,
}
JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#344054",
        "axes.labelcolor": "#344054",
        "xtick.color": "#475467",
        "ytick.color": "#475467",
    }
)
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, sharey=True)
bid_color = "#4472C4"
ask_color = "#E69F00"
bar_width = 0.00015
max_size = max(
    max(row["ask_size_units"], row["bid_size_units"]) for row in rows
)

for axis, state in zip(axes, states):
    state_rows = [row for row in rows if row["state"] == state["state"]]
    bid_prices = [row["bid_price_hkd"] for row in reversed(state_rows)]
    bid_sizes = [row["bid_size_units"] for row in reversed(state_rows)]
    ask_prices = [row["ask_price_hkd"] for row in state_rows]
    ask_sizes = [row["ask_size_units"] for row in state_rows]
    axis.bar(
        bid_prices,
        bid_sizes,
        width=bar_width,
        color=bid_color,
        edgecolor="#234E91",
        linewidth=0.8,
        label="Bid size",
    )
    axis.bar(
        ask_prices,
        ask_sizes,
        width=bar_width,
        color=ask_color,
        edgecolor="#9A6700",
        linewidth=0.8,
        label="Ask size",
    )
    best_bid = state_rows[0]["bid_price_hkd"]
    best_ask = state_rows[0]["ask_price_hkd"]
    mid = (best_bid + best_ask) / 2
    axis.axvline(mid, color="#667085", linestyle="--", linewidth=1.2)
    axis.text(
        mid,
        max_size * 1.05,
        f"mid {mid:.4f}",
        ha="center",
        va="bottom",
        color="#475467",
        fontsize=9,
    )
    axis.set_title(
        f"State {state['state']} — {state['send_time_readable'].split(' ', 1)[1]}",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#101828",
    )
    axis.set_ylabel("Quantity")
    axis.set_ylim(0, max_size * 1.2)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    axis.grid(axis="y", color="#EAECF0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

axes[0].legend(frameon=False, ncols=2, loc="upper right")
axes[-1].set_xlabel("Price (HKD)")
axes[-1].xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
fig.suptitle(
    "HK 7709 reconstructed 10-level order-book states · next sampled snapshot after 31 ms",
    fontsize=16,
    fontweight="bold",
    color="#101828",
    y=0.985,
)
fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
fig.savefig(PNG_PATH, dpi=180, bbox_inches="tight", facecolor="white")
print(f"wrote {JSON_PATH}")
print(f"wrote {PNG_PATH}")
