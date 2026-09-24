#!/usr/bin/env python3
"""Check the completed new-hybrid GP gamma sweep and its source anchors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "output/new_hybrid_gamma_sweep")
    args = parser.parse_args()
    result = json.loads((args.output_dir / "results.json").read_text())
    rows = result["daily"]
    keys = [(r["date"], r["gamma"], r["fill_mode"], r["gap_inventory"]) for r in rows]
    expected = {(d, g, mode, treatment)
                for d in result["test_dates"] for g in result["gammas"]
                for mode in ("through", "touch")
                for treatment in ("carry", "forced_flat")}
    if set(keys) != expected or len(keys) != len(expected):
        raise AssertionError("missing or duplicate daily gamma rows")
    prior_month = {t: json.loads(Path(path).read_text())
                   for t, path in result["anchor_results"].items()}
    anchor_checks = fee_checks = summary_checks = 0
    for r in rows:
        if r["gamma"] in (5.0, 3.125):
            original = next(a for a in prior_month[r["gap_inventory"]]["daily"]
                            if a["date"] == r["date"] and a["family"] == "GP"
                            and a["gamma"] == r["gamma"]
                            and a["fill_mode"] == r["fill_mode"])
            for variant in ("baseline", "new_hybrid"):
                if not close(r[f"{variant}_net_pnl_hkd"],
                             original[f"{variant}_net_hkd"]):
                    raise AssertionError("anchor net differs")
                if not close(r[f"{variant}_all_in_fees_hkd"],
                             original[f"{variant}_all_in_fees_hkd"]):
                    raise AssertionError("anchor fees differ")
            anchor_checks += 1
        for variant in ("baseline", "new_hybrid"):
            if not close(r[f"{variant}_gross_pnl_hkd"] -
                         r[f"{variant}_all_in_fees_hkd"],
                         r[f"{variant}_net_pnl_hkd"]):
                raise AssertionError("net/gross/fees mismatch")
            fee_checks += 1
        if not close(r["new_minus_baseline_hkd"],
                     r["new_hybrid_net_pnl_hkd"] - r["baseline_net_pnl_hkd"]):
            raise AssertionError("delta mismatch")
        if r["quote_decisions"] <= 0:
            raise AssertionError("zero quote decisions")
    for s in result["summary"]:
        chosen = [r for r in rows if r["gamma"] == s["gamma"]
                  and r["fill_mode"] == s["fill_mode"]
                  and r["gap_inventory"] == s["gap_inventory"]
                  and (s["subset"] == "all" or
                       r["l3_available"] == (s["subset"] == "l3_available"))]
        if len(chosen) != s["days"]:
            raise AssertionError("summary day count mismatch")
        for field in ("baseline_net_pnl_hkd", "new_hybrid_net_pnl_hkd",
                      "baseline_gross_pnl_hkd", "new_hybrid_gross_pnl_hkd",
                      "baseline_all_in_fees_hkd", "new_hybrid_all_in_fees_hkd",
                      "new_minus_baseline_hkd"):
            if not close(s[field], sum(r[field] for r in chosen)):
                raise AssertionError(f"summary mismatch: {field}")
            summary_checks += 1
    if len(result["test_dates"]) == 19:
        l3_days = {r["date"] for r in rows if r["l3_available"]}
        if len(l3_days) != 10:
            raise AssertionError("expected ten complete-L3 test days")
    verification = {
        "complete_grid": True,
        "daily_rows": len(rows),
        "anchor_reconciliation_checks": anchor_checks,
        "net_gross_fee_checks": fee_checks,
        "summary_total_checks": summary_checks,
        "test_days": len(result["test_dates"]),
        "gamma_count": len(result["gammas"]),
        "l3_available_test_days": len({r["date"] for r in rows if r["l3_available"]}),
        "caution": "Gamma selection on these July dates is exploratory; test on new untouched dates.",
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
