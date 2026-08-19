import unittest

import numpy as np

from train_7709_hybrid_deeplob import transform_window


class AnchoredTransformTests(unittest.TestCase):
    def test_history_is_anchored_to_prediction_time_mid(self) -> None:
        window = np.ones((2, 40), dtype=np.float32)
        window[0, np.arange(0, 40, 2)] = 99.0
        window[1, np.arange(0, 40, 2)] = 101.0
        window[:, 0] += 0.05
        window[:, 2] -= 0.05
        transformed = transform_window(
            window, np.zeros(40, dtype=np.float32), np.ones(40, dtype=np.float32)
        )
        self.assertLess(float(transformed[0, 0]), 0.0)
        self.assertGreater(float(transformed[1, 0]), 0.0)
        self.assertAlmostEqual(float(transformed[1, 0]), 0.05 / 101.0 * 10_000, places=2)


if __name__ == "__main__":
    unittest.main()
