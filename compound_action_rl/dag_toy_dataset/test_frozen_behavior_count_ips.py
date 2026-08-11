from __future__ import annotations

import unittest

import numpy as np

from config import TrainConfig
from frozen_behavior_count_ips import (
    FrozenBehaviorCountIPSConfig,
    FrozenBehaviorCountIPSTrainer,
    frozen_behavior_count_ips_advantages,
)


class FrozenBehaviorCountIPSTest(unittest.TestCase):
    def test_advantages_use_external_frozen_pool_counts(self) -> None:
        advantages, metrics = frozen_behavior_count_ips_advantages(
            [1.0, 0.8, 0.2],
            ["a", "b", "c"],
            {"a": 8, "b": 4, "c": 4},
            estimation_size=16,
        )
        scaled = np.array([2.0, 3.2, 0.8])
        expected = (scaled - scaled.mean()) / (scaled.std() + 1e-8)
        np.testing.assert_allclose(advantages, expected)
        self.assertEqual(metrics["estimator_unique_outcomes"], 3.0)
        self.assertEqual(metrics["estimation_size"], 16.0)
        self.assertAlmostEqual(metrics["estimator_probability_mass"], 1.0)

    def test_training_and_checkpoint_smoke(self) -> None:
        config = TrainConfig(
            budget=3,
            hidden_size=8,
            num_layers=1,
            group_size=8,
            num_updates=1,
            log_every=10,
        )
        frozen_ips = FrozenBehaviorCountIPSConfig(
            estimation_size=32,
            optimization_batch_size=8,
        )
        trainer = FrozenBehaviorCountIPSTrainer(
            config,
            frozen_ips=frozen_ips,
        )
        row = trainer.train()[0]
        self.assertTrue(np.isfinite(row["loss"]))
        self.assertEqual(row["estimation_size"], 32.0)
        self.assertEqual(row["optimization_batch_size"], 8.0)
        self.assertAlmostEqual(row["estimator_probability_mass"], 1.0)


if __name__ == "__main__":
    unittest.main()
