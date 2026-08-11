from __future__ import annotations

import math
import unittest

import numpy as np

from config import TrainConfig
from exact_probability_ips import (
    ExactProbabilityIPSTrainer,
    exact_probability_ips_advantages,
)


class ExactProbabilityIPSTest(unittest.TestCase):
    def test_advantages_use_exact_forward_path_probability(self) -> None:
        probabilities = np.array([0.5, 0.25, 0.125], dtype=np.float64)
        rewards = np.array([1.0, 0.8, 0.2], dtype=np.float64)
        advantages, metrics = exact_probability_ips_advantages(
            rewards,
            ["a", "b", "b"],
            ["tau-a", "tau-b", "tau-c"],
            np.log(probabilities),
        )

        scaled = rewards / probabilities
        expected = (scaled - scaled.mean()) / (scaled.std() + 1e-8)
        np.testing.assert_allclose(advantages, expected)
        self.assertAlmostEqual(metrics["ips_prob_mean"], probabilities.mean())
        self.assertEqual(metrics["ips_unique_outcomes"], 2.0)
        self.assertEqual(metrics["ips_unique_trajectories"], 3.0)

    def test_group_uses_sum_of_recorded_action_log_probabilities(self) -> None:
        trainer = ExactProbabilityIPSTrainer(
            TrainConfig(
                budget=3,
                max_step=3,
                hidden_size=8,
                num_layers=1,
                group_size=16,
                num_groups=1,
                num_updates=1,
            )
        )
        episodes = trainer.rollout_batch(16, explore=True)
        expected_probabilities = np.array(
            [
                math.exp(sum(step.log_prob_joint for step in episode.steps))
                for episode in episodes
            ]
        )

        trainer._group_advantages(episodes)

        self.assertAlmostEqual(
            trainer._last_ips_metrics["ips_prob_mean"],
            float(expected_probabilities.mean()),
        )
        for episode in episodes:
            self.assertTrue(all(np.isfinite(step.advantage) for step in episode.steps))

    def test_rejects_invalid_forward_log_probability(self) -> None:
        with self.assertRaises(ValueError):
            exact_probability_ips_advantages(
                [1.0], ["x"], ["tau"], [0.1]
            )


if __name__ == "__main__":
    unittest.main()
