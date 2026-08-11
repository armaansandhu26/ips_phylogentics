from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from config import TrainConfig
from count_ips import CountIPSTrainer, Episode
from dag_env import State
from learned_outcome_ips import (
    LearnedOutcomeConfig,
    LearnedOutcomeIPSTrainer,
    learned_outcome_ips_advantages,
)


class LearnedOutcomeIPSTest(unittest.TestCase):
    def test_uses_the_unchanged_count_ips_token_ppo_trainer(self) -> None:
        self.assertTrue(issubclass(LearnedOutcomeIPSTrainer, CountIPSTrainer))

    def test_outcome_model_starts_uniform_and_normalized(self) -> None:
        trainer = LearnedOutcomeIPSTrainer(
            TrainConfig(
                budget=4,
                hidden_size=8,
                num_layers=1,
                group_size=4,
                num_updates=1,
            ),
            outcome_config=LearnedOutcomeConfig(
                hidden_size=8,
                num_layers=1,
                train_epochs=1,
            ),
        )
        probabilities = trainer.outcome_probabilities()
        np.testing.assert_allclose(probabilities, np.full(5, 0.2), atol=1e-7)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)

    def test_advantages_are_reward_over_learned_outcome_probability(self) -> None:
        rewards = np.asarray([1.0, 0.6, 0.2], dtype=np.float64)
        probabilities = np.asarray([0.5, 0.25, 0.25], dtype=np.float64)
        advantages, metrics = learned_outcome_ips_advantages(
            rewards,
            ["a", "b", "b"],
            np.log(probabilities),
        )

        weights = rewards / probabilities
        expected = (weights - weights.mean()) / (weights.std() + 1e-8)
        np.testing.assert_allclose(advantages, expected, atol=1e-7)
        self.assertAlmostEqual(metrics["ips_prob_mean"], probabilities.mean())

    def test_uniform_mixture_lower_bounds_every_propensity(self) -> None:
        trainer = LearnedOutcomeIPSTrainer(
            TrainConfig(
                budget=3,
                hidden_size=8,
                num_layers=1,
                group_size=4,
                num_updates=1,
            ),
            outcome_config=LearnedOutcomeConfig(
                hidden_size=8,
                num_layers=1,
                train_epochs=1,
                uniform_mix=0.2,
            ),
        )
        raw = torch.log(torch.tensor([0.999997, 1e-6, 1e-6, 1e-6]))
        safe = trainer._safe_log_probabilities(raw).exp().numpy()

        self.assertTrue(np.all(safe >= 0.2 / 4.0))
        self.assertAlmostEqual(float(safe.sum()), 1.0, places=6)

    def test_density_fit_uses_outcomes_without_episode_trajectories(self) -> None:
        trainer = LearnedOutcomeIPSTrainer(
            TrainConfig(
                budget=3,
                hidden_size=8,
                num_layers=1,
                group_size=4,
                num_updates=1,
            ),
            outcome_config=LearnedOutcomeConfig(
                hidden_size=8,
                num_layers=1,
                lr=5e-2,
                train_epochs=20,
            ),
        )
        episodes = [
            Episode(terminal=State(0, 3)),
            Episode(terminal=State(0, 3)),
            Episode(terminal=State(0, 3)),
            Episode(terminal=State(1, 2)),
        ]
        before = trainer.outcome_probabilities()
        metrics = trainer._update_outcome_model(episodes)
        after = trainer.outcome_probabilities()

        self.assertGreater(after[0], before[0])
        self.assertAlmostEqual(float(after.sum()), 1.0, places=6)
        self.assertAlmostEqual(
            metrics["outcome_model_probability_mass"], 1.0, places=6
        )

    def test_training_updates_model_and_checkpoint_round_trips(self) -> None:
        trainer = LearnedOutcomeIPSTrainer(
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
            outcome_config=LearnedOutcomeConfig(
                hidden_size=16,
                num_layers=1,
                lr=1e-2,
                train_epochs=2,
            ),
            forward_lr_decay_after=1,
            forward_lr_after_decay=1e-4,
        )
        initial = {
            name: parameter.detach().clone()
            for name, parameter in trainer.outcome_model.named_parameters()
        }
        history = trainer.train(eval_every=1, eval_episodes=64)

        self.assertEqual(len(history), 2)
        self.assertEqual(
            [row["forward_lr"] for row in history], [3e-4, 1e-4]
        )
        self.assertTrue(np.isfinite(history[-1]["loss"]))
        self.assertTrue(np.isfinite(history[-1]["outcome_model_loss"]))
        self.assertGreater(history[-1]["ips_ess_fraction"], 0.0)
        self.assertEqual(history[-1]["outcome_fit_samples"], 16.0)
        self.assertEqual(history[-1]["total_rollouts_per_update"], 16.0)
        self.assertEqual(history[-1]["rollouts_per_update"], 16)
        self.assertAlmostEqual(
            history[-1]["outcome_model_probability_mass"], 1.0, places=6
        )
        self.assertTrue(
            any(
                not torch.equal(initial[name], parameter.detach())
                for name, parameter in trainer.outcome_model.named_parameters()
            )
        )

        before = trainer.outcome_probabilities()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save(Path(directory) / "checkpoint.pt", update_step=2)
            loaded = LearnedOutcomeIPSTrainer.load(checkpoint)
        self.assertEqual(loaded.forward_lr_decay_after, 1)
        self.assertEqual(loaded.forward_lr_after_decay, 1e-4)
        np.testing.assert_allclose(
            loaded.outcome_probabilities(), before, atol=1e-7
        )

    def test_outcome_fit_happens_after_forward_update_on_same_batch(self) -> None:
        trainer = LearnedOutcomeIPSTrainer(
            TrainConfig(
                budget=3,
                max_step=3,
                hidden_size=8,
                num_layers=1,
                group_size=8,
                num_groups=1,
                num_updates=1,
            ),
            outcome_config=LearnedOutcomeConfig(
                hidden_size=8,
                num_layers=1,
                train_epochs=1,
            ),
        )
        events: list[tuple[str, list[int]]] = []
        original_forward_update = trainer.update
        original_outcome_update = trainer._update_outcome_model

        def record_forward(episodes: list[Episode]) -> dict[str, float]:
            events.append(("forward", [id(episode) for episode in episodes]))
            return original_forward_update(episodes)

        def record_outcome(episodes: list[Episode]) -> dict[str, float]:
            events.append(("outcome", [id(episode) for episode in episodes]))
            return original_outcome_update(episodes)

        trainer.update = record_forward  # type: ignore[method-assign]
        trainer._update_outcome_model = record_outcome  # type: ignore[method-assign]

        groups, _ = trainer._collect_training_groups()
        self.assertEqual(events, [])
        trainer._update_training_groups(groups)

        self.assertEqual([event for event, _ in events], ["forward", "outcome"])
        self.assertEqual(events[0][1], events[1][1])


if __name__ == "__main__":
    unittest.main()
