from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from config import TrainConfig, default_terminal_rewards
from count_ips import (
    CountIPSTrainer,
    _r2_against,
    count_ips_advantages,
    ppo_token_loss,
)
from dag_env import (
    RIGHT,
    UP,
    State,
    find_default_terminal_states,
    reward_per_terminal_state,
)
from trajectory_ips import (
    KnownMultiplicityTrajectoryIPSTrainer,
    terminal_multiplicities,
    trajectory_ips_advantages,
)
from unknown_m_trajectory_ips import (
    UnknownMultiplicityTrajectoryIPSTrainer,
    unknown_m_trajectory_advantages,
)


class CountIPSTest(unittest.TestCase):
    def test_r2_against_measures_linear_fit_to_target(self) -> None:
        target = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(_r2_against(target, 2.0 * target + 0.1), 1.0)
        self.assertLess(
            _r2_against(target, np.array([0.4, 0.1, 0.3, 0.2])),
            0.1,
        )

    def test_budget_scaled_terminals_and_rewards(self) -> None:
        self.assertEqual(
            find_default_terminal_states(5),
            tuple(State(x, 5 - x) for x in range(6)),
        )
        self.assertEqual(
            default_terminal_rewards(3), (1.0, 0.8, 0.2, 0.05)
        )
        reward_map = reward_per_terminal_state(5)
        self.assertEqual(len(reward_map), 6)
        self.assertEqual(reward_map[State(0, 5)], 1.0)
        self.assertEqual(reward_map[State(5, 0)], 0.05)

        config = TrainConfig(budget=5, max_step=3)
        config.validate()
        self.assertEqual(len(config.terminal_rewards), 6)
        trainer = CountIPSTrainer(config)
        self.assertEqual(trainer.terminals, list(find_default_terminal_states(5)))

    def test_count_scaled_advantages(self) -> None:
        advantages, metrics = count_ips_advantages(
            [1.0, 1.0, 0.8, 0.2], ["a", "a", "b", "c"]
        )
        scaled = np.array([2.0, 2.0, 3.2, 0.8])
        expected = (scaled - scaled.mean()) / (scaled.std() + 1e-8)
        np.testing.assert_allclose(advantages, expected)
        self.assertEqual(metrics["ips_unique_outcomes"], 3.0)
        self.assertEqual(metrics["ips_max_outcome_count"], 2.0)

    def test_vectorized_rollout_batch_invariants(self) -> None:
        config = TrainConfig(
            budget=8,
            max_step=3,
            hidden_size=16,
            num_layers=1,
            group_size=257,
            num_updates=1,
        )
        trainer = CountIPSTrainer(config)
        episodes = trainer.rollout_batch(config.group_size)
        self.assertEqual(len(episodes), config.group_size)

        obs_width = config.budget + 1
        for episode in episodes:
            x = 0
            y = 0
            for step in episode.steps:
                remaining = config.budget - x - y
                self.assertEqual(int(np.argmax(step.obs[:obs_width])), x)
                self.assertEqual(
                    int(np.argmax(step.obs[obs_width : 2 * obs_width])), y
                )
                self.assertEqual(
                    int(np.argmax(step.obs[2 * obs_width :])), remaining
                )
                np.testing.assert_array_equal(step.direction_mask, [True, True])
                np.testing.assert_array_equal(
                    step.step_mask,
                    np.arange(config.max_step) < min(config.max_step, remaining),
                )
                self.assertTrue(np.isfinite(step.log_prob_joint))
                physical_step = step.step_index + 1
                if step.direction == RIGHT:
                    x += physical_step
                elif step.direction == UP:
                    y += physical_step
                else:
                    self.fail(f"invalid direction: {step.direction}")

            self.assertEqual(State(x, y), episode.terminal)
            self.assertEqual(x + y, config.budget)
            self.assertEqual(episode.signature, episode.terminal.signature)
            self.assertEqual(
                episode.reward, trainer.reward_by_terminal[episode.terminal]
            )

        evaluation = trainer.evaluate(257, batch_size=64)
        self.assertEqual(sum(evaluation["eval_outcome_counts"].values()), 257)

    def test_ppo_loss_matches_on_policy_ratio_one(self) -> None:
        new = torch.tensor([[-0.5, -0.2], [-0.7, 0.0]], requires_grad=True)
        old = new.detach().clone()
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.tensor([[True, True], [True, False]])
        loss, metrics = ppo_token_loss(
            new,
            advantages,
            log_paths_pf_old=old,
            mask=mask,
            clip_eps=0.2,
        )
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertAlmostEqual(metrics["mean_importance_ratio"], 1.0)
        loss.backward()
        self.assertIsNotNone(new.grad)

    def test_known_multiplicity_trajectory_advantages(self) -> None:
        advantages, metrics = trajectory_ips_advantages(
            [1.0, 1.0, 0.8, 0.2],
            ["x", "x", "y", "z"],
            ["a", "a", "b", "c"],
            {"x": 2, "y": 1, "z": 1},
        )
        scaled = np.array([1.0, 1.0, 3.2, 0.8])
        expected = (scaled - scaled.mean()) / (scaled.std() + 1e-8)
        np.testing.assert_allclose(advantages, expected)
        self.assertEqual(metrics["ips_unique_trajectories"], 3.0)

    def test_known_multiplicity_training_smoke(self) -> None:
        config = TrainConfig(
            budget=3,
            hidden_size=16,
            num_layers=1,
            group_size=32,
            num_groups=1,
            num_updates=1,
            log_every=1,
        )
        trainer = KnownMultiplicityTrajectoryIPSTrainer(config)
        history = trainer.train()
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(history[0]["loss"]))
        self.assertGreaterEqual(history[0]["ips_unique_trajectories"], 1.0)

    def test_unknown_m_path_advantage_rewards_rare_path(self) -> None:
        baseline, _ = unknown_m_trajectory_advantages(
            [1.0, 1.0, 1.0, 0.8],
            ["x", "x", "x", "y"],
            ["a", "a", "b", "c"],
            path_coefficient=0.0,
        )
        combined, metrics = unknown_m_trajectory_advantages(
            [1.0, 1.0, 1.0, 0.8],
            ["x", "x", "x", "y"],
            ["a", "a", "b", "c"],
            path_coefficient=1.0,
        )
        self.assertAlmostEqual(baseline[0], baseline[2])
        self.assertGreater(combined[2], combined[0])
        self.assertEqual(metrics["path_unique_trajectories"], 3.0)
        self.assertAlmostEqual(metrics["path_advantage_mean"], 0.0, places=7)

    def test_unknown_m_schedule_and_training_smoke(self) -> None:
        config = TrainConfig(
            budget=3,
            hidden_size=16,
            num_layers=1,
            group_size=32,
            num_groups=1,
            num_updates=1,
            log_every=1,
        )
        trainer = UnknownMultiplicityTrajectoryIPSTrainer(
            config,
            path_coefficient=1.0,
            path_warmup_updates=100,
            path_ramp_updates=100,
        )
        trainer._on_update_start(100)
        self.assertEqual(trainer.current_path_coefficient, 0.0)
        trainer._on_update_start(150)
        self.assertEqual(trainer.current_path_coefficient, 0.5)
        trainer._on_update_start(200)
        self.assertEqual(trainer.current_path_coefficient, 1.0)
        history = trainer.train()
        self.assertEqual(history[0]["path_coefficient"], 0.0)
        self.assertEqual(history[0]["path_count_decay"], 0.95)
        self.assertTrue(np.isfinite(history[0]["loss"]))

    def test_training_smoke(self) -> None:
        config = TrainConfig(
            budget=3,
            hidden_size=16,
            num_layers=1,
            group_size=16,
            num_groups=1,
            num_updates=1,
            entropy_coef=0.0,
            log_every=1,
        )
        trainer = CountIPSTrainer(config)
        history = trainer.train()
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(history[0]["loss"]))
        self.assertGreaterEqual(history[0]["ips_unique_outcomes"], 1.0)
        self.assertGreaterEqual(history[0]["global_unique_outcomes"], 1.0)
        self.assertGreaterEqual(
            history[0]["global_unique_outcomes"],
            history[0]["ips_unique_outcomes"],
        )

        evaluation = trainer.evaluate(128)
        self.assertEqual(sum(terminal_multiplicities(3, 3).values()), 18)
        self.assertEqual(sum(evaluation["eval_trajectory_counts"].values()), 128)
        self.assertLessEqual(evaluation["eval_unique_trajectories"], 18)
        for state in trainer.terminals:
            signature = state.signature
            terminal_count = evaluation["eval_outcome_counts"][signature]
            conditional = evaluation["eval_conditional_trajectory_probs"][signature]
            self.assertAlmostEqual(
                sum(conditional.values()), 1.0 if terminal_count else 0.0
            )
            self.assertGreaterEqual(
                evaluation["eval_normalized_trajectory_entropy"][signature], 0.0
            )
            self.assertLessEqual(
                evaluation["eval_normalized_trajectory_entropy"][signature], 1.0 + 1e-12
            )

    def test_periodic_checkpoints(self) -> None:
        config = TrainConfig(
            budget=3,
            hidden_size=8,
            num_layers=1,
            group_size=8,
            num_groups=1,
            num_updates=2,
            log_every=10,
        )
        trainer = CountIPSTrainer(config)
        with TemporaryDirectory() as checkpoint_dir:
            trainer.train(
                checkpoint_every=1,
                checkpoint_dir=checkpoint_dir,
            )
            self.assertEqual(
                sorted(path.name for path in Path(checkpoint_dir).iterdir()),
                [
                    "checkpoint_update_000001.pt",
                    "checkpoint_update_000002.pt",
                ],
            )


if __name__ == "__main__":
    unittest.main()
