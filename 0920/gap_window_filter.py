"""Mask quote decisions around recorded tolerant-MBO feed gaps.

The reconstruction records the last message before a same-session gap and
the first message after it.  For each gap, discard the closed time interval
``[previous_send_time - padding, resume_send_time + padding]``.  A quote's
forward label is usable only if its entire snapshot path is outside those
intervals and remains in the quote's reconstruction segment.

The mask is evaluated on the *unfiltered* reconstructed day.  Callers should
apply it to existing quote indices; deleting rows first would change a
20-snapshot target into a different economic horizon.
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _day_ms(compact_times: np.ndarray) -> np.ndarray:
    """YYYYMMDDhhmmssfff -> milliseconds from local midnight."""

    clock = compact_times % 1_000_000_000
    hours = clock // 10_000_000
    minutes = (clock // 100_000) % 100
    seconds = (clock // 1_000) % 100
    millis = clock % 1_000
    if np.any((hours > 23) | (minutes > 59) | (seconds > 59)):
        raise ValueError("invalid compact SendTime clock")
    return ((hours * 60 + minutes) * 60 + seconds) * 1_000 + millis


def _blackout_snapshots(
    day: Any, padding_ms: int
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Locate the union of all closed event windows on an original day."""

    send_times = np.asarray(day.send_times, dtype=np.int64)
    date = str(day.date)
    compact_date = int(date.replace("-", ""))
    if np.any(send_times // 1_000_000_000 != compact_date):
        raise ValueError(f"{date}: snapshot SendTime date differs from day.date")
    snapshot_ms = _day_ms(send_times)
    if np.any(np.diff(snapshot_ms) < 0):
        raise ValueError(f"{date}: snapshot SendTime is not chronological")

    excluded = np.zeros(len(send_times), dtype=bool)
    windows: list[dict[str, int]] = []
    for event in day.metadata.get("gap_events") or []:
        previous = int(event["previous_send_time"])
        resume = int(event["resume_send_time"])
        if (
            previous // 1_000_000_000 != compact_date
            or resume // 1_000_000_000 != compact_date
        ):
            raise ValueError(f"{date}: gap event belongs to a different date")
        previous_ms, resume_ms = _day_ms(np.array([previous, resume]))
        if resume_ms <= previous_ms:
            raise ValueError(f"{date}: gap resume must follow prior message")
        start_ms = int(previous_ms) - padding_ms
        end_ms = int(resume_ms) + padding_ms
        in_window = (snapshot_ms >= start_ms) & (snapshot_ms <= end_ms)
        excluded |= in_window
        windows.append(
            {
                "previous_send_time": previous,
                "resume_send_time": resume,
                "start_day_ms": start_ms,
                "end_day_ms": end_ms,
                "excluded_snapshot_count": int(np.count_nonzero(in_window)),
            }
        )
    return excluded, windows


def gap_window_keep_mask(
    day: Any,
    quote_indices: np.ndarray,
    *,
    horizon_snapshots: int = 20,
    padding_minutes: float = 2.0,
    history_snapshots: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a quote-aligned keep mask and JSON-ready audit for one day.

    ``day`` needs ``date``, ``send_times``, ``segments`` and ``metadata``
    attributes, as supplied by ``0824.run_multiday_strategy.DayData``.
    ``history_snapshots`` is optional for callers whose features require a
    contiguous within-segment lookback.  All time-window endpoints are
    excluded.  Audit reason counts can overlap.
    """

    if horizon_snapshots < 0 or history_snapshots < 0:
        raise ValueError("snapshot horizons must be non-negative")
    if not np.isfinite(padding_minutes) or padding_minutes < 0:
        raise ValueError("padding_minutes must be finite and non-negative")
    padding_ms = int(round(padding_minutes * 60_000))

    send_times = np.asarray(day.send_times, dtype=np.int64)
    segments = np.asarray(day.segments)
    quotes = np.asarray(quote_indices, dtype=np.int64)
    if send_times.ndim != 1 or segments.shape != send_times.shape:
        raise ValueError("send_times and segments must be aligned 1D arrays")
    if quotes.ndim != 1:
        raise ValueError("quote_indices must be a 1D array")
    if not len(send_times):
        raise ValueError("day has no snapshots")
    if np.any((quotes < 0) | (quotes >= len(send_times))):
        raise IndexError("quote index outside reconstructed day")

    date = str(day.date)
    excluded_snapshots, windows = _blackout_snapshots(day, padding_ms)
    for window in windows:
        start_ms, end_ms = window["start_day_ms"], window["end_day_ms"]
        quote_ms = _day_ms(send_times[quotes])
        window["excluded_quote_count"] = int(
            np.count_nonzero((quote_ms >= start_ms) & (quote_ms <= end_ms))
        )

    n_snapshots = len(send_times)
    future = quotes + horizon_snapshots
    future_in_bounds = future < n_snapshots
    safe_future = np.minimum(future, n_snapshots - 1)
    label_crosses_segment = future_in_bounds & (
        segments[quotes] != segments[safe_future]
    )

    # A prefix sum catches a masked intermediate observation even when both
    # label endpoints are outside a gap window.
    excluded_prefix = np.r_[0, np.cumsum(excluded_snapshots, dtype=np.int64)]
    label_touches_gap = (
        excluded_prefix[safe_future + 1] - excluded_prefix[quotes]
    ) > 0

    history_start = quotes - history_snapshots
    history_in_bounds = history_start >= 0
    safe_history = np.maximum(history_start, 0)
    history_crosses_segment = history_in_bounds & (
        segments[quotes] != segments[safe_history]
    )
    history_touches_gap = (
        excluded_prefix[quotes + 1] - excluded_prefix[safe_history]
    ) > 0

    keep = (
        future_in_bounds
        & ~label_crosses_segment
        & ~label_touches_gap
        & history_in_bounds
        & ~history_crosses_segment
        & ~history_touches_gap
    )
    audit: dict[str, Any] = {
        "date": date,
        "padding_minutes_each_side": float(padding_minutes),
        "horizon_snapshots": int(horizon_snapshots),
        "history_snapshots": int(history_snapshots),
        "gap_count": len(windows),
        "gap_windows": windows,
        "snapshot_count": n_snapshots,
        "excluded_snapshot_count": int(np.count_nonzero(excluded_snapshots)),
        "quote_count": int(len(quotes)),
        "retained_quote_count": int(np.count_nonzero(keep)),
        "removed_quote_count": int(np.count_nonzero(~keep)),
        "removed_quote_in_gap_window_count": int(
            np.count_nonzero(excluded_snapshots[quotes])
        ),
        "removed_quote_label_touches_gap_count": int(
            np.count_nonzero(label_touches_gap)
        ),
        "removed_quote_label_out_of_bounds_count": int(
            np.count_nonzero(~future_in_bounds)
        ),
        "removed_quote_label_crosses_segment_count": int(
            np.count_nonzero(label_crosses_segment)
        ),
        "removed_quote_history_out_of_bounds_count": int(
            np.count_nonzero(~history_in_bounds)
        ),
        "removed_quote_history_crosses_segment_count": int(
            np.count_nonzero(history_crosses_segment)
        ),
        "removed_quote_history_touches_gap_count": int(
            np.count_nonzero(history_touches_gap)
        ),
    }
    return keep, audit


def save_gap_filtered_reconstruction(
    source: Path,
    destination: Path,
    *,
    padding_minutes: float = 2.0,
    required_recovery_ms: int = 120_000,
) -> dict[str, Any]:
    """Delete blackout snapshots from a newly rebuilt tolerant MBO cache.

    Preserve each surviving reconstruction segment ID.  If a blackout would
    leave retained rows on both sides within the same segment, fail instead of
    letting a later 20-snapshot label or GP clock transition jump across it.
    """

    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    with np.load(source, allow_pickle=False) as archive:
        features = archive["features"]
        exchange_ms = archive["exchange_ms"]
        send_times = archive["send_times"]
        segments = archive["segments"]
        metadata = json.loads(str(archive["metadata"]))
    if any(len(values) != len(send_times) for values in (features, exchange_ms, segments)):
        raise ValueError("reconstruction cache arrays are not aligned")
    if metadata.get("reconstruction_policy") != "tolerant":
        raise ValueError("expected tolerant MBO reconstruction")
    if metadata.get("recovery_ms") != required_recovery_ms:
        raise ValueError(
            f"expected recovery_ms={required_recovery_ms}; rebuild source first"
        )
    if not len(send_times):
        raise ValueError("reconstruction cache has no snapshots")
    compact_date = int(send_times[0] // 1_000_000_000)
    date = f"{compact_date:08d}"
    formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    day = SimpleNamespace(
        date=formatted_date,
        send_times=send_times,
        segments=segments,
        metadata=metadata,
    )
    padding_ms = int(round(padding_minutes * 60_000))
    if not np.isfinite(padding_minutes) or padding_minutes < 0:
        raise ValueError("padding_minutes must be finite and non-negative")
    excluded, windows = _blackout_snapshots(day, padding_ms)
    keep = ~excluded
    if not np.any(keep):
        raise ValueError(f"{formatted_date}: blackout removes every snapshot")
    retained_original_indices = np.flatnonzero(keep)
    retained_segments = segments[keep]
    if np.any(
        (retained_segments[1:] == retained_segments[:-1])
        & (np.diff(retained_original_indices) != 1)
    ):
        raise ValueError(
            f"{formatted_date}: blackout splits a retained segment; "
            "resegment before using labels or GP clock transitions"
        )
    for segment in np.unique(retained_segments):
        positions = np.flatnonzero(retained_segments == segment)
        if positions[-1] - positions[0] + 1 != len(positions):
            raise ValueError(f"{formatted_date}: segment {segment} is not contiguous")

    retained_features = features[keep]
    retained_times = send_times[keep]
    mids = (
        retained_features[:, 0].astype(np.float64)
        + retained_features[:, 2].astype(np.float64)
    ) / 2
    spreads = (
        retained_features[:, 0].astype(np.float64)
        - retained_features[:, 2].astype(np.float64)
    )
    audit: dict[str, Any] = {
        "date": formatted_date,
        "source": str(source),
        "destination": str(destination),
        "padding_minutes_each_side": float(padding_minutes),
        "recovery_ms": int(required_recovery_ms),
        "gap_count": len(windows),
        "gap_windows": windows,
        "original_snapshot_count": int(len(send_times)),
        "removed_snapshot_count": int(np.count_nonzero(excluded)),
        "retained_snapshot_count": int(np.count_nonzero(keep)),
        "retained_segment_count": int(len(np.unique(retained_segments))),
    }
    metadata = dict(metadata)
    metadata["gap_blackout_filter"] = audit
    metadata["snapshot_count"] = audit["retained_snapshot_count"]
    metadata["segments"] = {
        str(segment): int(np.count_nonzero(retained_segments == segment))
        for segment in np.unique(retained_segments)
    }
    metadata["first_send_time"] = int(retained_times[0])
    metadata["last_send_time"] = int(retained_times[-1])
    metadata["mid_price_hkd"] = {
        "min": float(mids.min()),
        "median": float(np.median(mids)),
        "max": float(mids.max()),
    }
    metadata["spread_hkd"] = {
        "min": float(spreads.min()),
        "median": float(np.median(spreads)),
        "max": float(spreads.max()),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=retained_features,
        exchange_ms=exchange_ms[keep],
        send_times=retained_times,
        segments=retained_segments,
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    return audit


def enforce_safe_day_eligibility(
    day: Any, *, horizon_snapshots: int = 20, history_snapshots: int = 100
) -> tuple[Any, dict[str, Any]]:
    """Trim a loaded DayData's eligible rows for safe labels and features.

    Gap days must first be loaded from a physically filtered cache so GP clock
    calibration cannot use discarded observations.  The 0824 loader admits
    the first segment row with only 99 prior snapshots; this function removes
    it when a 100-lag causal feature is used.
    """

    if day.metadata.get("gap_events") and not day.metadata.get("gap_blackout_filter"):
        raise ValueError("gap day must be loaded from a blackout-filtered cache")
    if not is_dataclass(day):
        raise TypeError("day must be a dataclass instance")
    eligible = np.asarray(day.eligible, dtype=np.int64)
    if len(day.features) != len(eligible):
        raise ValueError("features and eligible quote rows are not aligned")
    keep, audit = gap_window_keep_mask(
        day,
        eligible,
        horizon_snapshots=horizon_snapshots,
        history_snapshots=history_snapshots,
    )
    cleaned = replace(day, eligible=eligible[keep], features=day.features[keep])
    return cleaned, audit
