#!/usr/bin/env python3
"""Export July GP test-day reconstruction quality for sensitivity analyses.

The source audit's ``clean_trading_dates`` means a day was accepted for the
existing replay. It does not imply that no order-state errors or feed gaps
were recorded. This script keeps that distinction explicit.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "0906/output/july_2026_tolerant_as_gp_drift_results.json"
OUTPUT = HERE / "output/combination_study/source_quality.json"
SESSION_MS = (2.5 + 3.0) * 60 * 60 * 1000


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    audit = source["source_audit"]
    test_dates = audit["clean_trading_dates"][1:]
    reconstruction = {row["date"]: row for row in audit["reconstruction"]}
    def has_recorded_issue(record: dict) -> bool:
        return bool(record.get("tainted") or record.get("order_errors") or record.get("gap_events"))

    rows = []
    for position, date in enumerate(test_dates, start=1):
        record = reconstruction[date]
        errors = record.get("order_errors") or {}
        gaps = record.get("gap_events") or []
        delete = int(errors.get("delete_missing_order", 0))
        modify = int(errors.get("modify_missing_order", 0))
        gap_ms = sum(int(gap["gap_ms"]) for gap in gaps)
        flagged = has_recorded_issue(record)
        prior = audit["clean_trading_dates"][max(0, position - 5):position]
        prior_with_issue = [day for day in prior if has_recorded_issue(reconstruction[day])]
        rows.append({
            "date": date,
            "phase": "exploration_2026_07_03_to_17" if date <= "2026-07-17"
            else "late_check_2026_07_20_to_30",
            "quality_group": "recorded_reconstruction_issue" if flagged
            else "no_recorded_reconstruction_issue",
            "rolling_calibration_dates": prior,
            "rolling_calibration_dates_with_issue": prior_with_issue,
            "rolling_calibration_issue_count": len(prior_with_issue),
            "replay_eligible": True,
            "full_session_coverage": bool(record["full_session_coverage"]),
            "snapshots": int(record["snapshots"]),
            "missing_order_deletes": delete,
            "missing_order_modifies": modify,
            "missing_order_errors_total": delete + modify,
            "feed_gap_count": len(gaps),
            "feed_gap_total_ms": gap_ms,
            "feed_gap_max_ms": max((int(gap["gap_ms"]) for gap in gaps), default=0),
            "feed_gap_scheduled_session_share": gap_ms / SESSION_MS,
            "recovery_skipped_changed_groups": int(
                record.get("recovery_skipped_changed_groups") or 0
            ),
            "final_state_tainted": bool(record.get("tainted")),
        })

    assert len(rows) == 19
    assert len({row["date"] for row in rows}) == len(rows)
    assert all(row["full_session_coverage"] and not row["final_state_tainted"] for row in rows)
    summary = {}
    for phase in ("all", "exploration_2026_07_03_to_17", "late_check_2026_07_20_to_30"):
        selected = rows if phase == "all" else [row for row in rows if row["phase"] == phase]
        count = sum(row["quality_group"] == "recorded_reconstruction_issue" for row in selected)
        summary[phase] = {
            "test_days": len(selected),
            "days_with_recorded_issue": count,
            "share_with_recorded_issue": count / len(selected),
            "days_with_issue_in_rolling_calibration": sum(
                row["rolling_calibration_issue_count"] > 0 for row in selected
            ),
            "missing_order_errors_total": sum(row["missing_order_errors_total"] for row in selected),
            "feed_gap_count": sum(row["feed_gap_count"] for row in selected),
            "feed_gap_total_ms": sum(row["feed_gap_total_ms"] for row in selected),
            "recovery_skipped_changed_groups": sum(
                row["recovery_skipped_changed_groups"] for row in selected
            ),
        }
    assert summary["all"]["days_with_recorded_issue"] == 9
    assert summary["all"]["days_with_issue_in_rolling_calibration"] == 18
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "grain": "one row per included GP replay test date",
        "quality_group_rule": (
            "recorded_reconstruction_issue iff final_state_tainted or at least one "
            "missing-order error or feed gap in source reconstruction audit"
        ),
        "session_denominator_ms": int(SESSION_MS),
        "caveats": [
            "No recorded issue does not prove complete or correct exchange reconstruction.",
            "Missing-order error counts lack a common event-count denominator and are not error rates.",
            "Gap share uses the scheduled 5.5-hour session and is an approximate coverage measure.",
            "This grouping describes test-day source quality only; rolling calibration may use issue-flagged prior days.",
            "Both chronological phases have already been examined in prior gamma studies and are not pristine holdouts.",
        ],
        "summary": summary,
        "days": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
