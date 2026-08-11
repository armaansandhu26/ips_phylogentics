from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import TrainConfig
from ema_count_ips import (
    EMACountIPSConfig,
    EMACountIPSTrainer,
    ema_count_ips_advantages,
    update_ema_outcome_frequencies,
)


class EMACountIPSTest(unittest.TestCase):
    def test_uniform_initialization_covers_every_terminal(self) -> None:
        config = TrainConfig(
            budget=3,
            hidden_size=8,
            num_layers=1,
            num_updates=1,
        )
        trainer = EMACountIPSTrainer(
            config,
            ema_ips=EMACountIPSConfig(alpha=0.1, initialization="uniform"),
        )
        self.assertEqual(len(trainer.ema_terminal_frequencies), 4)
        for probability in trainer.ema_terminal_frequencies.values():
            self.assertAlmostEqual(probability, 0.25)
        self.assertAlmostEqual(sum(trainer.ema_terminal_frequencies.values()), 1.0)

    def test_update_decays_outcomes_absent_from_batch(self) -> None:
        frequencies: dict[object, float] = {}
        update_ema_outcome_frequencies(
            ["a", "a", "b"], frequencies, alpha=0.25
        )
        self.assertAlmostEqual(frequencies["a"], 2.0 / 3.0)
        self.assertAlmostEqual(frequencies["b"], 1.0 / 3.0)

        update_ema_outcome_frequencies(["b", "c"], frequencies, alpha=0.25)
        self.assertAlmostEqual(frequencies["a"], 0.5)
        self.assertAlmostEqual(frequencies["b"], 0.375)
        self.assertAlmostEqual(frequencies["c"], 0.125)
        self.assertAlmostEqual(sum(frequencies.values()), 1.0)

    def test_update_can_keep_absent_outcomes_stale(self) -> None:
        frequencies: dict[object, float] = {
            "a": 0.25,
            "b": 0.25,
            "c": 0.25,
            "d": 0.25,
        }
        update_ema_outcome_frequencies(
            ["a", "a"],
            frequencies,
            alpha=0.1,
            decay_absent_outcomes=False,
        )
        self.assertAlmostEqual(frequencies["a"], 0.325)
        self.assertAlmostEqual(frequencies["b"], 0.25)
        self.assertAlmostEqual(frequencies["c"], 0.25)
        self.assertAlmostEqual(frequencies["d"], 0.25)
        self.assertAlmostEqual(sum(frequencies.values()), 1.075)

    def test_advantages_use_ema_frequencies(self) -> None:
        advantages, metrics = ema_count_ips_advantages(
            [1.0, 0.8, 0.2],
            ["a", "b", "c"],
            {"a": 0.5, "b": 0.375, "c": 0.125},
        )
        scaled = np.array([2.0, 0.8 / 0.375, 1.6])
        expected = (scaled - scaled.mean()) / (scaled.std() + 1e-8)
        np.testing.assert_allclose(advantages, expected)
        self.assertAlmostEqual(metrics["ema_probability_mass"], 1.0)

    def test_raw_mode_uses_unstandardized_inverse_scaled_rewards(self) -> None:
        weights, metrics = ema_count_ips_advantages(
            [1.0, 0.8, 0.2],
            ["a", "b", "c"],
            {"a": 0.5, "b": 0.375, "c": 0.125},
            tracker_eps=1e-6,
            normalize=False,
        )
        np.testing.assert_allclose(weights, [2.0, 0.8 / 0.375, 1.6])
        self.assertAlmostEqual(metrics["advantage_mean"], float(weights.mean()))

    def test_raw_weight_training_smoke(self) -> None:
        config = TrainConfig(
            budget=3,
            hidden_size=8,
            num_layers=1,
            group_size=8,
            num_groups=1,
            num_updates=1,
            log_every=10,
        )
        trainer = EMACountIPSTrainer(
            config,
            ema_ips=EMACountIPSConfig(
                alpha=0.005,
                initialization="uniform",
                decay_absent_outcomes=True,
                tracker_eps=1e-6,
                ips_weight_mode="raw",
            ),
        )
        row = trainer.train()[0]
        self.assertTrue(np.isfinite(row["loss"]))
        self.assertGreater(row["advantage_min"], 0.0)
        self.assertEqual(row["ema_raw_ips_weights"], 1.0)
        self.assertAlmostEqual(row["ema_probability_mass"], 1.0)

    def test_training_and_checkpoint_smoke(self) -> None:
        config = TrainConfig(
            budget=3,
            hidden_size=16,
            num_layers=1,
            group_size=16,
            num_groups=1,
            num_updates=2,
            log_every=10,
        )
        trainer = EMACountIPSTrainer(
            config, ema_ips=EMACountIPSConfig(alpha=0.2)
        )
        history = trainer.train()
        self.assertEqual(len(history), 2)
        self.assertTrue(np.isfinite(history[-1]["loss"]))
        self.assertAlmostEqual(history[-1]["ema_probability_mass"], 1.0)
        self.assertEqual(sum(trainer.lifetime_terminal_counts.values()), 32)

        with TemporaryDirectory() as directory:
            checkpoint = trainer.save(Path(directory) / "ema.pt", update_step=2)
            restored = EMACountIPSTrainer.load(checkpoint)
            self.assertEqual(restored.ema_ips.alpha, 0.2)
            self.assertEqual(
                dict(restored.lifetime_terminal_counts),
                dict(trainer.lifetime_terminal_counts),
            )
            for state, probability in trainer.ema_terminal_frequencies.items():
                self.assertAlmostEqual(
                    restored.ema_terminal_frequencies[state], probability
                )


if __name__ == "__main__":
    unittest.main()
