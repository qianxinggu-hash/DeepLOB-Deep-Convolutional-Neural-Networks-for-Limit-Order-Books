"""Order identity extraction must use whole timestamp groups and real IDs."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l3_order_signal import extract_l3_features  # noqa: E402


class L3OrderSignalTests(unittest.TestCase):
    def test_atomic_replace_preserves_l2_but_changes_l3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orders.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("SendTime", "MsgType", "SecurityId", "Price", "Quantity", "Side", "OrderID"))
                writer.writerows((
                    (20260721093000000, 30, 7709, 100000, 100, 0, 1),
                    (20260721093000000, 30, 7709, 100000, 100, 0, 2),
                    (20260721093000000, 30, 7709, 101000, 100, 1, 3),
                    (20260721093012000, 32, 7709, "", "", 0, 1),
                    (20260721093012000, 30, 7709, 100000, 100, 0, 4),
                ))
            book = np.zeros((2, 40), dtype=np.float32)
            book[:, :4] = (101.0, 100.0, 100.0, 200.0)
            features, diagnostics = extract_l3_features(
                path,
                np.asarray((20260721093000000, 20260721093012000), dtype=np.int64),
                book,
            )
            self.assertEqual(diagnostics["quotes_checked"], 2)
            self.assertAlmostEqual(float(features[0, 0]), 1 / 3)
            self.assertAlmostEqual(float(features[1, 0]), 1 / 3)
            self.assertAlmostEqual(float(features[0, 1]), 0.0)
            self.assertLess(float(features[1, 1]), 0.0)
            self.assertAlmostEqual(float(features[1, 4]), -0.5)

    def test_rejects_best_book_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orders.csv"
            path.write_text(
                "SendTime,MsgType,SecurityId,Price,Quantity,Side,OrderID\n"
                "20260721093000000,30,7709,100000,100,0,1\n"
                "20260721093000000,30,7709,101000,100,1,2\n"
            )
            wrong = np.zeros((1, 40), dtype=np.float32)
            wrong[0, :4] = (101.0, 100.0, 100.0, 200.0)
            with self.assertRaisesRegex(ValueError, "MBO/L2 best-book mismatch"):
                extract_l3_features(path, np.asarray((20260721093000000,)), wrong)


if __name__ == "__main__":
    unittest.main()
