from __future__ import annotations

import math
import unittest
from collections import defaultdict

import numpy as np

from backward_corrected_ips import (
    BackwardCorrectedIPSTrainer,
    backward_corrected_ips_advantages,
)
from config import TrainConfig
from dag_env import RIGHT, UP, State, uniform_backward_log_probability


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


class BackwardCorrectedIPSTest(unittest.TestCase):
    def test_advantages_use_raw_reward_backward_over_forward_weight(self) -> None:
        rewards = np.array([1.0, 0.8, 0.2], dtype=np.float64)
        p_f = np.array([0.5, 0.25, 0.125], dtype=np.float64)
        p_b = np.array([0.25, 0.5, 0.125], dtype=np.float64)
        advantages, metrics = backward_corrected_ips_advantages(
            rewards,
            ["a", "b", "b"],
            ["tau-a", "tau-b", "tau-c"],
            np.log(p_f),
            np.log(p_b),
        )

        weights = rewards * p_b / p_f
        expected = (weights - weights.mean()) / (weights.std() + 1e-8)
        np.testing.assert_allclose(advantages, expected)
        self.assertAlmostEqual(metrics["ips_scaled_reward_mean"], weights.mean())
        self.assertAlmostEqual(metrics["backward_probability_mean"], p_b.mean())

    def test_backward_correction_allocates_one_unit_per_terminal(self) -> None:
        mass_by_terminal: dict[State, float] = defaultdict(float)
        reward_by_terminal: dict[State, float] = {}
        trajectories = _enumerate_trajectories(budget=5, max_step=3)
        p_f = 1.0 / len(trajectories)
        for trajectory in trajectories:
            terminal = _terminal(trajectory)
            reward = 1.0 / (1.0 + terminal.x)
            p_b = math.exp(
                uniform_backward_log_probability(trajectory, max_step=3)
            )
            # Expected sampled contribution: P_F * R * P_B / P_F.
            mass_by_terminal[terminal] += p_f * reward * p_b / p_f
            reward_by_terminal[terminal] = reward

        self.assertEqual(mass_by_terminal.keys(), reward_by_terminal.keys())
        for terminal, mass in mass_by_terminal.items():
            self.assertAlmostEqual(mass, reward_by_terminal[terminal], places=12)

    def test_training_uses_finite_raw_ips_weights(self) -> None:
        trainer = BackwardCorrectedIPSTrainer(
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
        history = trainer.train()
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(history[0]["ips_scaled_reward_mean"]))
        self.assertTrue(np.isfinite(history[0]["loss"]))
        self.assertIn("backward_probability_mean", history[0])


if __name__ == "__main__":
    unittest.main()
