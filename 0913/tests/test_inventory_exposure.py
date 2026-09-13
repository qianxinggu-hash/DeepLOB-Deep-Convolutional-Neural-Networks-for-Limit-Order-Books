import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for directory in ("0913", "0906", "0824"):
    sys.path.insert(0, str(ROOT / directory))
from paper_gp_replay import InventoryExposure


class InventoryExposureTests(unittest.TestCase):
    def test_time_integral_includes_flat_long_short_and_gap(self):
        clock = InventoryExposure(0)
        clock.advance(2000, 0)
        clock.advance(5000, 2)
        clock.advance(8000, -1)
        clock.advance(10000, 0)
        m = clock.metrics()
        self.assertEqual(m["squared_inventory_lot2_seconds"], 15.0)
        self.assertEqual(m["absolute_inventory_lot_seconds"], 9.0)
        self.assertEqual(m["exposure_duration_seconds"], 10.0)
        self.assertAlmostEqual(m["nonzero_inventory_time_share"], 0.6)
        self.assertEqual(m["time_at_inventory_lots_seconds"], {"-1": 3.0, "0": 4.0, "2": 3.0})

    def test_simultaneous_events_have_no_duration_and_reverse_time_is_rejected(self):
        clock = InventoryExposure(0)
        clock.advance(1000, 0)
        clock.advance(1000, 1)
        clock.advance(2000, 0)
        self.assertEqual(clock.metrics()["squared_inventory_lot2_seconds"], 0)
        with self.assertRaises(ValueError):
            clock.advance(1999, 0)


if __name__ == "__main__":
    unittest.main()
