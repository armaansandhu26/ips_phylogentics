from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import TrainConfig
from count_ips import Episode, StepRecord
from dag_env import State
from ema_count_ips import ema_count_ips_advantages
from epsilon_greedy_count_ips import ExplorationConfig
from scheduled_ema_count_ips import (
    ExplorationHistoryIPSConfig,
    ScheduledEMACountIPSTrainer,
    running_histogram_ips_advantages,
    update_ema_from_groups,
)
from scheduled_rollout_count_ips import RolloutScheduleConfig


def _minimal_episode(terminal: State, reward: float) -> Episode:
    step = StepRecord(
        obs=np.zeros(12, dtype=np.float32),
        direction_mask=np.ones(2, dtype=bool),
        step_mask=np.ones(1, dtype=bool),
        direction=0,
        step_index=0,
        log_prob_direction=0.0,
        log_prob_step=0.0,
    )
    return Episode(
        steps=[step],
        terminal=terminal,
        signature=terminal.signature,
        trajectory=((0, 1),),
        reward=reward,
    )


class ScheduledEMACountIPSTest(unittest.TestCase):
    def test_group_ema_update_matches_sequential_batches(self) -> None:
        frequencies: dict[object, float] = {}
        last = update_ema_from_groups(
            [["a", "a", "b", "b"], ["b", "c", "c", "c"]],
            frequencies,
            alpha=0.25,
            update_unit="group",
        )
        self.assertAlmostEqual(frequencies["a"], 0.375)
        self.assertAlmostEqual(frequencies["b"], 0.4375)
        self.assertAlmostEqual(frequencies["c"], 0.1875)
        self.assertAlmostEqual(sum(frequencies.values()), 1.0)
        self.assertAlmostEqual(last["b"], 0.25)
        self.assertAlmostEqual(last["c"], 0.75)

    def test_pool_ema_update_uses_full_pool_frequencies(self) -> None:
        frequencies: dict[object, float] = {}
        update_ema_from_groups(
            [["a", "a"], ["b", "c"]],
            frequencies,
            alpha=0.5,
            update_unit="pool",
        )
        self.assertAlmostEqual(frequencies["a"], 0.5)
        self.assertAlmostEqual(frequencies["b"], 0.25)
        self.assertAlmostEqual(frequencies["c"], 0.25)

    def test_running_histogram_advantages_use_lifetime_counts(self) -> None:
        advantages, metrics = running_histogram_ips_advantages(
            [1.0, 0.5],
            ["a", "b"],
            {"a": 3, "b": 1},
        )
        scaled = np.array([1.0 / 0.75, 0.5 / 0.25])
        expected = (scaled - scaled.mean()) / (scaled.std() + 1e-8)
        np.testing.assert_allclose(advantages, expected)
        self.assertAlmostEqual(metrics["histogram_total_visits"], 4.0)

    def test_collect_updates_history_then_scales_with_ema(self) -> None:
        trainer = ScheduledEMACountIPSTrainer(
            TrainConfig(
                budget=3,
                max_step=1,
                hidden_size=8,
                num_layers=1,
                group_size=2,
                num_groups=2,
                num_updates=1,
            ),
            exploration=ExplorationConfig(
                epsilon_start=0.0,
                epsilon_end=0.0,
                temperature_start=1.0,
                temperature_end=1.0,
            ),
            rollout_schedule=RolloutScheduleConfig(groups_start=2, groups_end=2),
            history_ips=ExplorationHistoryIPSConfig(
                propensity_mode="ema",
                alpha=1.0,
                initialization="first_batch",
                ema_update_unit="pool",
            ),
        )
        state_a = State(0, 3)
        state_b = State(1, 2)
        state_c = State(2, 1)
        pool = [
            _minimal_episode(state_a, 1.0),
            _minimal_episode(state_b, 0.8),
            _minimal_episode(state_a, 1.0),
            _minimal_episode(state_c, 0.2),
        ]
        trainer.rollout_batch = (  # type: ignore[method-assign]
            lambda batch_size, explore=False: pool
        )
        trainer._on_update_start(1)
        groups, metrics = trainer._collect_training_groups()

        self.assertEqual(len(groups), 2)
        self.assertEqual(trainer.lifetime_terminal_counts[state_a], 2)
        self.assertEqual(trainer.lifetime_terminal_counts[state_b], 1)
        self.assertEqual(trainer.lifetime_terminal_counts[state_c], 1)
        self.assertAlmostEqual(trainer.ema_terminal_frequencies[state_a], 0.5)
        self.assertAlmostEqual(trainer.ema_terminal_frequencies[state_b], 0.25)
        self.assertAlmostEqual(trainer.ema_terminal_frequencies[state_c], 0.25)

        first_expected, _ = ema_count_ips_advantages(
            [1.0, 0.8],
            [state_a, state_b],
            trainer.ema_terminal_frequencies,
        )
        second_expected, _ = ema_count_ips_advantages(
            [1.0, 0.2],
            [state_a, state_c],
            trainer.ema_terminal_frequencies,
        )
        np.testing.assert_allclose(
            [step.advantage for step in groups[0][0].steps]
            + [step.advantage for step in groups[0][1].steps],
            first_expected,
        )
        np.testing.assert_allclose(
            [step.advantage for step in groups[1][0].steps]
            + [step.advantage for step in groups[1][1].steps],
            second_expected,
        )
        self.assertEqual(metrics[0]["advantage_group_size"], 2.0)
        self.assertEqual(metrics[0]["propensity_pool_size"], 4.0)

    def test_schedule_anneals_groups_with_small_g(self) -> None:
        trainer = ScheduledEMACountIPSTrainer(
            TrainConfig(
                budget=3,
                max_step=1,
                hidden_size=8,
                num_layers=1,
                group_size=4,
                num_groups=3,
                num_updates=3,
                log_every=10,
            ),
            exploration=ExplorationConfig(
                epsilon_start=0.0,
                epsilon_end=0.0,
                temperature_start=1.0,
                temperature_end=1.0,
            ),
            rollout_schedule=RolloutScheduleConfig(
                groups_start=3,
                groups_end=1,
                anneal_updates=3,
                schedule="linear",
            ),
            history_ips=ExplorationHistoryIPSConfig(
                propensity_mode="ema",
                alpha=0.5,
                initialization="uniform",
            ),
        )
        history = trainer.train()
        self.assertEqual([row["rollout_groups"] for row in history], [3, 2, 1])
        self.assertEqual(
            [row["rollouts_per_update"] for row in history], [12, 8, 4]
        )
        self.assertEqual(history[-1]["cumulative_rollouts"], 24)
        self.assertGreater(history[-1]["histogram_total_visits"], 0.0)
        self.assertTrue(np.isfinite(history[-1]["loss"]))

    def test_running_mode_training_and_checkpoint(self) -> None:
        config = TrainConfig(
            budget=3,
            max_step=1,
            hidden_size=8,
            num_layers=1,
            group_size=4,
            num_groups=2,
            num_updates=2,
            log_every=10,
        )
        trainer = ScheduledEMACountIPSTrainer(
            config,
            exploration=ExplorationConfig(
                epsilon_start=0.0,
                epsilon_end=0.0,
                temperature_start=1.0,
                temperature_end=1.0,
            ),
            rollout_schedule=RolloutScheduleConfig(groups_start=2, groups_end=1),
            history_ips=ExplorationHistoryIPSConfig(propensity_mode="running"),
        )
        with TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            history = trainer.train(
                checkpoint_every=2,
                checkpoint_dir=checkpoint_dir,
            )
            path = trainer.save(Path(tmp) / "checkpoint.pt", update_step=2)
            loaded = ScheduledEMACountIPSTrainer.load(path)
            self.assertEqual(
                dict(loaded.lifetime_terminal_counts),
                dict(trainer.lifetime_terminal_counts),
            )
            self.assertEqual(history[-1]["rollout_groups"], 1)
            self.assertTrue((checkpoint_dir / "checkpoint_update_000002.pt").exists())


if __name__ == "__main__":
    unittest.main()
