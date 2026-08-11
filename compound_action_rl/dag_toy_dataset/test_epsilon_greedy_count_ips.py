from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.distributions import Categorical

from config import TrainConfig
from epsilon_greedy_count_ips import (
    EpsilonGreedyCountIPSTrainer,
    ExplorationConfig,
    annealed_value,
    epsilon_temperature_distribution,
)


class EpsilonGreedyCountIPSTest(unittest.TestCase):
    def test_distribution_mixes_with_uniform_over_valid_actions(self) -> None:
        base = Categorical(logits=torch.tensor([[4.0, 1.0, -2.0]]))
        mask = torch.tensor([[True, True, False]])
        mixed = epsilon_temperature_distribution(
            base, mask, epsilon=0.25, temperature=2.0
        )
        tempered = torch.softmax(torch.tensor([4.0, 1.0]) / 2.0, dim=0)
        expected = 0.75 * tempered + 0.25 * torch.tensor([0.5, 0.5])
        torch.testing.assert_close(mixed.probs[0, :2], expected)
        self.assertEqual(float(mixed.probs[0, 2]), 0.0)

    def test_full_epsilon_is_uniform_and_evaluation_bypasses_it(self) -> None:
        trainer = EpsilonGreedyCountIPSTrainer(
            TrainConfig(
                budget=3,
                hidden_size=8,
                num_layers=1,
                group_size=8,
                num_groups=1,
                num_updates=1,
            ),
            exploration=ExplorationConfig(
                epsilon_start=1.0,
                epsilon_end=1.0,
                temperature_start=10.0,
                temperature_end=10.0,
            ),
        )
        base = Categorical(logits=torch.tensor([[8.0, -8.0]]))
        mask = torch.tensor([[True, True]])
        training_dist = trainer._action_distribution(base, mask, explore=True)
        evaluation_dist = trainer._action_distribution(base, mask, explore=False)
        torch.testing.assert_close(training_dist.probs, torch.tensor([[0.5, 0.5]]))
        self.assertIs(evaluation_dist, base)

    def test_linear_and_cosine_schedules_reach_endpoints(self) -> None:
        for schedule in ("linear", "cosine"):
            self.assertEqual(
                annealed_value(
                    0.4,
                    0.05,
                    update_step=1,
                    anneal_updates=5,
                    schedule=schedule,
                ),
                0.4,
            )
            self.assertAlmostEqual(
                annealed_value(
                    0.4,
                    0.05,
                    update_step=5,
                    anneal_updates=5,
                    schedule=schedule,
                ),
                0.05,
            )
            self.assertAlmostEqual(
                annealed_value(
                    0.4,
                    0.05,
                    update_step=50,
                    anneal_updates=5,
                    schedule=schedule,
                ),
                0.05,
            )

    def test_rollout_and_ppo_use_the_same_exploration_policy(self) -> None:
        trainer = EpsilonGreedyCountIPSTrainer(
            TrainConfig(
                budget=4,
                max_step=3,
                hidden_size=16,
                num_layers=1,
                group_size=32,
                num_groups=1,
                num_updates=2,
            ),
            exploration=ExplorationConfig(
                epsilon_start=0.4,
                epsilon_end=0.1,
                temperature_start=2.0,
                temperature_end=1.0,
                schedule="linear",
            ),
        )
        trainer._on_update_start(1)
        episodes = trainer.rollout_batch(32, explore=True)
        trainer._group_advantages(episodes)
        loss, metrics = trainer._joint_policy_loss(episodes)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(metrics["mean_importance_ratio"], 1.0, places=5)
        self.assertEqual(trainer._last_ips_metrics["exploration_epsilon"], 0.4)
        self.assertEqual(trainer._last_ips_metrics["exploration_temperature"], 2.0)

    def test_training_and_checkpoint_smoke(self) -> None:
        config = TrainConfig(
            budget=3,
            hidden_size=16,
            num_layers=1,
            group_size=32,
            num_groups=1,
            num_updates=2,
            log_every=10,
        )
        exploration = ExplorationConfig(
            epsilon_start=0.5,
            epsilon_end=0.1,
            temperature_start=2.0,
            temperature_end=1.0,
            anneal_updates=2,
            schedule="linear",
        )
        trainer = EpsilonGreedyCountIPSTrainer(
            config, exploration=exploration
        )
        history = trainer.train()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["exploration_epsilon"], 0.5)
        self.assertAlmostEqual(history[1]["exploration_epsilon"], 0.1)
        self.assertEqual(history[0]["exploration_temperature"], 2.0)
        self.assertEqual(history[1]["exploration_temperature"], 1.0)
        self.assertTrue(np.isfinite(history[-1]["loss"]))

        with TemporaryDirectory() as directory:
            checkpoint = trainer.save(Path(directory) / "checkpoint.pt", update_step=2)
            loaded = EpsilonGreedyCountIPSTrainer.load(checkpoint)
        self.assertEqual(loaded.exploration, exploration)
        self.assertAlmostEqual(loaded.current_epsilon, 0.1)
        self.assertAlmostEqual(loaded.current_temperature, 1.0)


if __name__ == "__main__":
    unittest.main()
