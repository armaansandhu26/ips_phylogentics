from __future__ import annotations

import math
import tempfile
import unittest
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import TrainConfig
from count_ips import Episode, count_ips_advantages
from dag_env import RIGHT, UP, State
from single_trajectory_ips import (
    BackwardPolicyConfig,
    SingleTrajectoryIPSTrainer,
    single_trajectory_ips_advantages,
)


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


class SingleTrajectoryIPSTest(unittest.TestCase):
    def test_unique_outcomes_do_not_reduce_estimator_to_plain_grpo(self) -> None:
        rewards = [1.0, 1.0, 1.0, 1.0]
        outcomes = ["a", "b", "c", "d"]
        count_advantages, _ = count_ips_advantages(rewards, outcomes)
        path_advantages, metrics = single_trajectory_ips_advantages(
            rewards,
            outcomes,
            ["ta", "tb", "tc", "td"],
            np.log([0.20, 0.10, 0.05, 0.025]),
            np.log([0.25, 0.25, 0.25, 0.25]),
        )

        np.testing.assert_allclose(count_advantages, np.zeros(4))
        self.assertGreater(float(path_advantages.std()), 0.99)
        self.assertEqual(metrics["ips_unique_outcomes"], 4.0)
        self.assertEqual(metrics["propensity_uses_counts"], 0.0)

    def test_ratio_is_conditionally_unbiased_for_inverse_propensity(self) -> None:
        # Two paths reach x. P_F(x) = 0.3 and Q_B(. | x) is any normalized
        # proposal; it need not equal the forward conditional distribution.
        p_f = np.asarray([0.1, 0.2], dtype=np.float64)
        q_b = np.asarray([0.8, 0.2], dtype=np.float64)
        conditional_forward = p_f / p_f.sum()
        expectation = float(np.sum(conditional_forward * q_b / p_f))
        self.assertAlmostEqual(expectation, 1.0 / p_f.sum())

    def test_backward_policy_normalizes_over_paths_for_each_outcome(self) -> None:
        trainer = SingleTrajectoryIPSTrainer(
            TrainConfig(
                budget=4,
                max_step=3,
                hidden_size=8,
                num_layers=1,
                group_size=4,
                num_updates=1,
            ),
            backward_config=BackwardPolicyConfig(
                hidden_size=8,
                num_layers=1,
                train_epochs=1,
            ),
        )
        torch.manual_seed(17)
        with torch.no_grad():
            trainer.backward_policy.head.weight.normal_(0.0, 0.4)
            trainer.backward_policy.head.bias.normal_(0.0, 0.4)

        trajectories = _enumerate_trajectories(4, 3)
        episodes = [
            Episode(terminal=_terminal(trajectory), trajectory=trajectory)
            for trajectory in trajectories
        ]
        log_probabilities = trainer.backward_path_log_probabilities(episodes)
        probability_by_terminal: dict[State, float] = defaultdict(float)
        for episode, log_probability in zip(episodes, log_probabilities):
            probability_by_terminal[episode.terminal] += math.exp(log_probability)

        self.assertEqual(len(probability_by_terminal), 5)
        for probability in probability_by_terminal.values():
            self.assertAlmostEqual(probability, 1.0, places=5)

    def test_training_updates_both_policies_and_checkpoint_round_trips(self) -> None:
        trainer = SingleTrajectoryIPSTrainer(
            TrainConfig(
                budget=4,
                max_step=3,
                hidden_size=16,
                num_layers=1,
                group_size=8,
                num_groups=1,
                num_updates=2,
                log_every=10,
            ),
            backward_config=BackwardPolicyConfig(
                hidden_size=8,
                num_layers=1,
                lr=1e-2,
                train_epochs=2,
            ),
        )
        initial_backward = {
            name: parameter.detach().clone()
            for name, parameter in trainer.backward_policy.named_parameters()
        }
        history = trainer.train(eval_every=1, eval_episodes=32)

        self.assertEqual(len(history), 2)
        self.assertIn("backward_loss", history[-1])
        self.assertTrue(np.isfinite(history[-1]["loss"]))
        self.assertTrue(np.isfinite(history[-1]["backward_loss"]))
        self.assertTrue(
            any(
                not torch.equal(initial_backward[name], parameter.detach())
                for name, parameter in trainer.backward_policy.named_parameters()
            )
        )

        episodes = trainer.rollout_batch(8)
        before = trainer.backward_path_log_probabilities(episodes)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save(
                Path(directory) / "checkpoint.pt",
                update_step=2,
            )
            loaded = SingleTrajectoryIPSTrainer.load(checkpoint)
        after = loaded.backward_path_log_probabilities(episodes)
        np.testing.assert_allclose(after, before, atol=1e-7)

    def test_resumed_training_matches_uninterrupted_training_exactly(self) -> None:
        config = TrainConfig(
            budget=4,
            max_step=3,
            hidden_size=16,
            num_layers=1,
            group_size=8,
            num_groups=1,
            num_updates=2,
            log_every=10,
            seed=23,
        )
        backward_config = BackwardPolicyConfig(
            hidden_size=8,
            num_layers=1,
            lr=1e-2,
            train_epochs=2,
        )

        uninterrupted = SingleTrajectoryIPSTrainer(
            config,
            backward_config=backward_config,
        )
        uninterrupted.train()

        interrupted = SingleTrajectoryIPSTrainer(
            replace(config, num_updates=1),
            backward_config=backward_config,
        )
        interrupted.train()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = interrupted.save(
                Path(directory) / "checkpoint.pt",
                update_step=1,
            )
            resumed = SingleTrajectoryIPSTrainer.load(checkpoint)
        self.assertTrue(resumed.checkpoint_resumable)
        resumed.config = replace(resumed.config, num_updates=2)
        resumed.train()

        self.assertEqual([row["step"] for row in resumed.training_history], [1, 2])
        self.assertEqual(
            resumed.training_history[-1]["cumulative_rollouts"],
            uninterrupted.training_history[-1]["cumulative_rollouts"],
        )
        for uninterrupted_policy, resumed_policy in (
            (uninterrupted.direction_policy, resumed.direction_policy),
            (uninterrupted.step_policy, resumed.step_policy),
            (uninterrupted.backward_policy, resumed.backward_policy),
        ):
            for name, expected in uninterrupted_policy.state_dict().items():
                torch.testing.assert_close(
                    resumed_policy.state_dict()[name],
                    expected,
                    rtol=0.0,
                    atol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
