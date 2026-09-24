from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gap_window_filter import (  # noqa: E402
    enforce_safe_day_eligibility,
    gap_window_keep_mask,
    save_gap_filtered_reconstruction,
)


@dataclass
class SyntheticDay:
    date: str
    send_times: np.ndarray
    segments: np.ndarray
    metadata: dict
    eligible: np.ndarray | None = None
    features: np.ndarray | None = None


def minute_clock(minutes: int) -> int:
    hour, minute = divmod(9 * 60 + 30 + minutes, 60)
    return 20260703 * 1_000_000_000 + hour * 10_000_000 + minute * 100_000


def gap_day() -> SyntheticDay:
    return SyntheticDay(
        date="2026-07-03",
        send_times=np.array([minute_clock(i) for i in range(13)], dtype=np.int64),
        segments=np.array([1] * 7 + [2] * 6, dtype=np.int16),
        metadata={
            "gap_events": [
                {
                    "previous_send_time": minute_clock(5),
                    "resume_send_time": minute_clock(7),
                    "gap_ms": 120_000,
                    "new_segment": 2,
                }
            ]
        },
    )


class GapWindowFilterTests(unittest.TestCase):
    def test_closed_window_label_and_history_are_all_excluded(self) -> None:
        day = gap_day()
        quotes = np.arange(13, dtype=np.int64)
        keep, audit = gap_window_keep_mask(
            day, quotes, horizon_snapshots=1, history_snapshots=1
        )
        self.assertEqual(np.flatnonzero(keep).tolist(), [1, 11])
        self.assertEqual(audit["gap_count"], 1)
        self.assertEqual(audit["excluded_snapshot_count"], 7)
        self.assertEqual(audit["gap_windows"][0]["start_day_ms"], (9 * 60 + 33) * 60_000)
        self.assertEqual(audit["gap_windows"][0]["end_day_ms"], (9 * 60 + 39) * 60_000)
        self.assertEqual(audit["removed_quote_count"], 11)

    def test_label_cannot_cross_segment_without_a_recorded_gap(self) -> None:
        day = SyntheticDay(
            date="2026-07-03",
            send_times=np.array([minute_clock(i) for i in range(6)]),
            segments=np.array([1, 1, 1, 2, 2, 2]),
            metadata={"gap_events": []},
        )
        keep, audit = gap_window_keep_mask(
            day, np.arange(6), horizon_snapshots=1, history_snapshots=1
        )
        self.assertEqual(np.flatnonzero(keep).tolist(), [1, 4])
        self.assertEqual(audit["removed_quote_label_crosses_segment_count"], 1)
        self.assertEqual(audit["removed_quote_history_crosses_segment_count"], 1)

    def test_loaded_day_drops_first_100_lag_crossing(self) -> None:
        dates = np.array(
            [minute_clock(i // 10) + (i % 10) * 1_000 for i in range(150)],
            dtype=np.int64,
        )
        day = SyntheticDay(
            date="2026-07-03",
            send_times=dates,
            segments=np.ones(150, dtype=np.int16),
            metadata={"gap_events": []},
            eligible=np.array([99, 100, 120], dtype=np.int64),
            features=np.array([[99.0], [100.0], [120.0]]),
        )
        safe, audit = enforce_safe_day_eligibility(day)
        self.assertEqual(safe.eligible.tolist(), [100, 120])
        self.assertEqual(safe.features[:, 0].tolist(), [100.0, 120.0])
        self.assertEqual(audit["removed_quote_history_out_of_bounds_count"], 1)
        self.assertEqual(day.eligible.tolist(), [99, 100, 120])

    def test_archive_filter_preserves_contiguous_segments(self) -> None:
        day = gap_day()
        metadata = dict(day.metadata)
        metadata.update(reconstruction_policy="tolerant", recovery_ms=120_000)
        features = np.tile([10.0, 100.0, 9.0, 100.0], (13, 1))
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.npz"
            filtered = Path(temporary) / "filtered.npz"
            np.savez_compressed(
                source,
                features=features,
                exchange_ms=np.arange(13, dtype=np.int64),
                send_times=day.send_times,
                segments=day.segments,
                metadata=np.array(json.dumps(metadata)),
            )
            audit = save_gap_filtered_reconstruction(source, filtered)
            with np.load(filtered, allow_pickle=False) as archive:
                kept_times = archive["send_times"].tolist()
                kept_segments = archive["segments"].tolist()
                saved_metadata = json.loads(str(archive["metadata"]))
        self.assertEqual(kept_times, [minute_clock(i) for i in (0, 1, 2, 10, 11, 12)])
        self.assertEqual(kept_segments, [1, 1, 1, 2, 2, 2])
        self.assertEqual(audit["removed_snapshot_count"], 7)
        self.assertEqual(saved_metadata["snapshot_count"], 6)
        self.assertEqual(saved_metadata["gap_blackout_filter"]["gap_count"], 1)

    def test_archive_filter_rejects_unsegmented_hole(self) -> None:
        day = gap_day()
        metadata = dict(day.metadata)
        metadata.update(reconstruction_policy="tolerant", recovery_ms=120_000)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.npz"
            np.savez_compressed(
                source,
                features=np.tile([10.0, 100.0, 9.0, 100.0], (13, 1)),
                exchange_ms=np.arange(13),
                send_times=day.send_times,
                segments=np.ones(13, dtype=np.int16),
                metadata=np.array(json.dumps(metadata)),
            )
            with self.assertRaisesRegex(ValueError, "splits a retained segment"):
                save_gap_filtered_reconstruction(source, Path(temporary) / "filtered.npz")


if __name__ == "__main__":
    unittest.main()
