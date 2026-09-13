import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_fee_break_even_experiment.py"
SPEC = importlib.util.spec_from_file_location("fee_break_even_experiment", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FeeBreakEvenExperimentTests(unittest.TestCase):
    def test_break_even_rate_zeros_net_pnl(self) -> None:
        rate = MODULE.break_even_fee_bps(100.0, 1_000_000.0)
        self.assertAlmostEqual(rate, 1.0)
        self.assertAlmostEqual(MODULE.net_pnl_at_fee(100.0, 1_000_000.0, rate), 0.0)

    def test_fee_repricing_is_monotonic(self) -> None:
        low_fee = MODULE.net_pnl_at_fee(100.0, 1_000_000.0, 0.5)
        high_fee = MODULE.net_pnl_at_fee(100.0, 1_000_000.0, 1.5)
        self.assertGreater(low_fee, high_fee)

    def test_nonpositive_turnover_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.break_even_fee_bps(100.0, 0.0)

    def test_experiment_reproduces_source_results(self) -> None:
        result = MODULE.build_experiment()
        self.assertTrue(result["validation"]["all_passed"])
        classic_through = next(
            row
            for row in result["summary"]
            if row["model"] == "classic_as" and row["fill_mode"] == "through"
        )
        self.assertAlmostEqual(
            classic_through["break_even_all_in_fee_bps_per_side"],
            1.8990749771412343,
        )


if __name__ == "__main__":
    unittest.main()
