from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import TrainConfig
from count_ips import Episode, StepRecord
from dag_env import State
from epsilon_greedy_count_ips import ExplorationConfig
from frozen_behavior_count_ips import frozen_behavior_count_ips_advantages
from scheduled_rollout_count_ips import (
    RolloutScheduleConfig,
    ScheduledFrozenPoolCountIPSTrainer,
    ScheduledGroupLocalCountIPSTrainer,
    scheduled_group_count,
)


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


class ScheduledRolloutCountIPSTest(unittest.TestCase):
    def test_integer_schedule_reaches_endpoints(self) -> None:
        self.assertEqual(
            [
                scheduled_group_count(
                    5,
                    1,
                    update_step=step,
                    anneal_updates=5,
                    schedule="linear",
                )
                for step in range(1, 6)
            ],
            [5, 4, 3, 2, 1],
        )
        self.assertEqual(
            scheduled_group_count(
                5,
                1,
                update_step=50,
                anneal_updates=5,
                schedule="cosine",
            ),
            1,
        )

    def test_strict_mode_anneals_groups_and_keeps_group_local_advantages(self) -> None:
        trainer = ScheduledGroupLocalCountIPSTrainer(
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
                epsilon_start=0.4,
                epsilon_end=0.1,
                temperature_start=2.0,
                temperature_end=1.0,
                anneal_updates=3,
                schedule="linear",
            ),
            rollout_schedule=RolloutScheduleConfig(
                groups_start=3,
                groups_end=1,
                anneal_updates=3,
                schedule="linear",
            ),
        )
        advantage_group_sizes: list[int] = []
        original = trainer._group_advantages

        def record_group_size(episodes: list[Episode]) -> float:
            advantage_group_sizes.append(len(episodes))
            return original(episodes)

        trainer._group_advantages = record_group_size  # type: ignore[method-assign]
        history = trainer.train()
        self.assertEqual([row["rollout_groups"] for row in history], [3, 2, 1])
        self.assertEqual(
            [row["rollouts_per_update"] for row in history], [12, 8, 4]
        )
        self.assertEqual(history[-1]["cumulative_rollouts"], 24)
        self.assertEqual(advantage_group_sizes, [4, 4, 4, 4, 4, 4])
        self.assertAlmostEqual(history[0]["exploration_epsilon"], 0.4)
        self.assertAlmostEqual(history[-1]["exploration_epsilon"], 0.1)

    def test_frozen_mode_uses_pool_counts_but_normalizes_inside_each_group(self) -> None:
        trainer = ScheduledFrozenPoolCountIPSTrainer(
            TrainConfig(
                budget=3,
                max_step=1,
                hidden_size=8,
                num_layers=1,
                group_size=2,
                num_groups=2,
                num_updates=1,
            ),
            rollout_schedule=RolloutScheduleConfig(
                groups_start=2,
                groups_end=2,
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

        first_expected, _ = frozen_behavior_count_ips_advantages(
            [1.0, 0.8],
            [state_a, state_b],
            {state_a: 2, state_b: 1, state_c: 1},
            estimation_size=4,
        )
        second_expected, _ = frozen_behavior_count_ips_advantages(
            [1.0, 0.2],
            [state_a, state_c],
            {state_a: 2, state_b: 1, state_c: 1},
            estimation_size=4,
        )
        np.testing.assert_allclose(
            [episode.steps[0].advantage for episode in groups[0]],
            first_expected,
        )
        np.testing.assert_allclose(
            [episode.steps[0].advantage for episode in groups[1]],
            second_expected,
        )
        self.assertEqual(metrics[0]["propensity_pool_size"], 4.0)
        self.assertEqual(metrics[0]["propensity_pool_unique_outcomes"], 3.0)
        self.assertAlmostEqual(metrics[0]["ips_prob_mean"], 0.375)

    def test_both_modes_train_and_round_trip_checkpoints(self) -> None:
        for trainer_type in (
            ScheduledGroupLocalCountIPSTrainer,
            ScheduledFrozenPoolCountIPSTrainer,
        ):
            with self.subTest(trainer_type=trainer_type.__name__):
                config = TrainConfig(
                    budget=3,
                    hidden_size=8,
                    num_layers=1,
                    group_size=8,
                    num_groups=2,
                    num_updates=1,
                    log_every=10,
                )
                schedule = RolloutScheduleConfig(
                    groups_start=2,
                    groups_end=1,
                    anneal_updates=2,
                    schedule="linear",
                )
                trainer = trainer_type(config, rollout_schedule=schedule)
                row = trainer.train()[0]
                self.assertTrue(np.isfinite(row["loss"]))
                self.assertEqual(row["rollout_groups"], 2)
                self.assertEqual(row["rollouts_per_update"], 16)
                if trainer_type is ScheduledFrozenPoolCountIPSTrainer:
                    self.assertEqual(row["propensity_pool_size"], 16.0)

                with TemporaryDirectory() as directory:
                    checkpoint = trainer.save(
                        Path(directory) / "checkpoint.pt", update_step=1
                    )
                    loaded = trainer_type.load(checkpoint)
                self.assertEqual(loaded.rollout_schedule, schedule)
                self.assertEqual(loaded.current_rollout_groups, 2)


if __name__ == "__main__":
    unittest.main()
