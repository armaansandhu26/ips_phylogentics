from __future__ import annotations

import math
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from config import TrainConfig
from count_ips import Episode
from dag_env import RIGHT, UP, State
from learned_reverse_ips import (
    LearnedReverseConfig,
    LearnedReverseIPSTrainer,
    RunningLogWeightNormalizer,
    learned_reverse_ips_advantages,
)


def _enumerate_trajectories(
    budget: int, max_step: int
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


class LearnedReverseIPSTest(unittest.TestCase):
    def test_running_scale_preserves_cross_batch_weight_severity(self) -> None:
        mild = RunningLogWeightNormalizer(
            decay=0.99, advantage_clip=10.0, log_ratio_clip=20.0
        )
        severe = RunningLogWeightNormalizer(
            decay=0.99, advantage_clip=10.0, log_ratio_clip=20.0
        )
        baseline = np.zeros(16, dtype=np.float64)
        mild.normalize(baseline)
        severe.normalize(baseline)

        mild_log_weights = np.zeros(16, dtype=np.float64)
        mild_log_weights[0] = np.log(2.0)
        severe_log_weights = np.zeros(16, dtype=np.float64)
        severe_log_weights[0] = np.log(1000.0)
        mild_advantages, mild_metrics = mild.normalize(mild_log_weights)
        severe_advantages, severe_metrics = severe.normalize(severe_log_weights)

        self.assertAlmostEqual(mild_advantages[0], 1.0)
        self.assertEqual(severe_advantages[0], 10.0)
        self.assertGreater(severe_advantages[0], 5.0 * mild_advantages[0])
        self.assertEqual(mild_metrics["running_advantage_clip_fraction"], 0.0)
        self.assertEqual(
            severe_metrics["running_advantage_clip_fraction"], 1.0 / 16.0
        )

    def test_reverse_policy_normalizes_over_paths_for_each_terminal(self) -> None:
        trainer = LearnedReverseIPSTrainer(
            TrainConfig(
                budget=5,
                max_step=3,
                hidden_size=8,
                num_layers=1,
                group_size=4,
                num_updates=1,
            ),
            reverse_config=LearnedReverseConfig(
                hidden_size=8,
                num_layers=1,
                train_epochs=1,
            ),
        )
        # Check normalization for a non-uniform learned policy, not just the
        # exactly uniform zero initialization.
        torch.manual_seed(17)
        with torch.no_grad():
            trainer.reverse_policy.head.weight.normal_(0.0, 0.4)
            trainer.reverse_policy.head.bias.normal_(0.0, 0.4)

        trajectories = _enumerate_trajectories(5, 3)
        episodes = [
            Episode(terminal=_terminal(trajectory), trajectory=trajectory)
            for trajectory in trajectories
        ]
        log_probabilities = trainer.reverse_path_log_probabilities(episodes)
        probability_by_terminal: dict[State, float] = defaultdict(float)
        for episode, log_probability in zip(episodes, log_probabilities):
            probability_by_terminal[episode.terminal] += math.exp(log_probability)

        self.assertEqual(len(probability_by_terminal), 6)
        for probability in probability_by_terminal.values():
            self.assertAlmostEqual(probability, 1.0, places=5)

    def test_advantages_match_reward_reverse_over_forward_weights(self) -> None:
        rewards = np.asarray([1.0, 0.6, 0.2], dtype=np.float64)
        p_f = np.asarray([0.1, 0.2, 0.05], dtype=np.float64)
        q = np.asarray([0.25, 0.5, 0.125], dtype=np.float64)
        advantages, metrics = learned_reverse_ips_advantages(
            rewards,
            ["a", "b", "b"],
            ["tau-a", "tau-b", "tau-c"],
            np.log(p_f),
            np.log(q),
        )

        weights = rewards * q / p_f
        expected = (weights - weights.mean()) / (weights.std() + 1e-8)
        np.testing.assert_allclose(advantages, expected, atol=1e-7)
        np.testing.assert_allclose(
            metrics["ips_prob_mean"], np.mean(p_f / q), atol=1e-12
        )

    def test_training_updates_reverse_policy_and_checkpoint_round_trips(self) -> None:
        trainer = LearnedReverseIPSTrainer(
            TrainConfig(
                budget=3,
                max_step=3,
                hidden_size=16,
                num_layers=1,
                group_size=16,
                num_groups=1,
                num_updates=2,
                log_every=10,
            ),
            reverse_config=LearnedReverseConfig(
                hidden_size=16,
                num_layers=1,
                lr=1e-2,
                train_epochs=2,
            ),
            forward_lr_decay_after=1,
            forward_lr_after_decay=1e-4,
            advantage_normalization="running",
            running_scale_decay=0.9,
            running_advantage_clip=10.0,
        )
        initial = {
            name: parameter.detach().clone()
            for name, parameter in trainer.reverse_policy.named_parameters()
        }
        history = trainer.train(eval_every=1, eval_episodes=64)

        self.assertEqual(len(history), 2)
        self.assertEqual(
            [row["forward_lr"] for row in history], [3e-4, 1e-4]
        )
        self.assertIn("eval_boundary_outcome_probs", history[-1])
        self.assertEqual(len(history[-1]["eval_boundary_outcome_probs"]), 4)
        self.assertIn("running_log_weight_rms", history[-1])
        self.assertTrue(np.isfinite(history[-1]["loss"]))
        self.assertTrue(np.isfinite(history[-1]["reverse_loss"]))
        self.assertGreater(history[-1]["ips_ess_fraction"], 0.0)
        self.assertTrue(
            any(
                not torch.equal(initial[name], parameter.detach())
                for name, parameter in trainer.reverse_policy.named_parameters()
            )
        )

        episodes = trainer.rollout_batch(8)
        before = trainer.reverse_path_log_probabilities(episodes)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save(Path(directory) / "checkpoint.pt", update_step=2)
            loaded = LearnedReverseIPSTrainer.load(checkpoint)
        self.assertEqual(loaded.forward_lr_decay_after, 1)
        self.assertEqual(loaded.forward_lr_after_decay, 1e-4)
        self.assertEqual(loaded.advantage_normalization, "running")
        self.assertIsNotNone(loaded.running_weight_normalizer)
        assert loaded.running_weight_normalizer is not None
        self.assertEqual(loaded.running_weight_normalizer.updates, 2)
        after = loaded.reverse_path_log_probabilities(episodes)
        np.testing.assert_allclose(after, before, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
