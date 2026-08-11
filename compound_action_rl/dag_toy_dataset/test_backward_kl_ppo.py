from __future__ import annotations

import math
import unittest
from collections import defaultdict

import numpy as np

from backward_kl_ppo import (
    BackwardKLPPOTrainer,
    backward_kl_advantages,
    uniform_backward_log_probability,
)
from config import TrainConfig
from dag_env import RIGHT, UP, State


def _enumerate_trajectories(
    budget: int,
    max_step: int,
) -> list[tuple[tuple[int, int], ...]]:
    trajectories: list[tuple[tuple[int, int], ...]] = []

    def visit(depth: int, path: tuple[tuple[int, int], ...]) -> None:
        if depth == budget:
            trajectories.append(path)
            return
        for direction in (RIGHT, UP):
            for length in range(1, min(max_step, budget - depth) + 1):
                visit(depth + length, path + ((direction, length),))

    visit(0, ())
    return trajectories


def _terminal(trajectory: tuple[tuple[int, int], ...]) -> State:
    return State(
        sum(length for direction, length in trajectory if direction == RIGHT),
        sum(length for direction, length in trajectory if direction == UP),
    )


class BackwardKLPPOTest(unittest.TestCase):
    def test_uniform_backward_policy_normalizes_for_every_terminal(self) -> None:
        probability_by_terminal: dict[State, float] = defaultdict(float)
        for trajectory in _enumerate_trajectories(budget=5, max_step=3):
            probability_by_terminal[_terminal(trajectory)] += math.exp(
                uniform_backward_log_probability(trajectory, max_step=3)
            )

        self.assertEqual(len(probability_by_terminal), 6)
        for probability in probability_by_terminal.values():
            self.assertAlmostEqual(probability, 1.0, places=12)

    def test_backward_kl_score_matches_log_target_to_policy_ratio(self) -> None:
        rewards = [1.0, 0.5, 0.25]
        forward = np.log([0.1, 0.2, 0.05])
        backward = np.log([0.25, 0.5, 0.125])
        advantages, metrics = backward_kl_advantages(
            rewards,
            ["x", "y", "z"],
            forward,
            backward,
            reward_beta=1.0,
        )
        scores = np.log(rewards) + backward - forward
        expected = (scores - scores.mean()) / (scores.std() + 1e-8)
        np.testing.assert_allclose(advantages, expected)
        self.assertAlmostEqual(metrics["trajectory_kl_score_mean"], scores.mean())
        self.assertEqual(metrics["ips_unique_outcomes"], 3.0)

    def test_beta_schedule_and_training_smoke(self) -> None:
        config = TrainConfig(
            budget=3,
            hidden_size=16,
            num_layers=1,
            group_size=16,
            num_groups=1,
            num_updates=2,
            entropy_coef=0.0,
            log_every=10,
        )
        trainer = BackwardKLPPOTrainer(
            config,
            reward_beta_start=0.25,
            reward_beta_end=1.0,
            beta_anneal_updates=2,
        )
        history = trainer.train()
        self.assertEqual([row["reward_beta"] for row in history], [0.25, 1.0])
        self.assertTrue(np.isfinite(history[-1]["loss"]))
        self.assertGreaterEqual(history[-1]["global_unique_outcomes"], 1.0)


if __name__ == "__main__":
    unittest.main()
