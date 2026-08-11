from __future__ import annotations

import unittest

import torch

from flat_ema_ips import FlatEMAConfig, ema_update, train_flat_ema


class FlatEMAIPSTest(unittest.TestCase):
    def test_ema_update_decays_every_outcome_and_preserves_mass(self) -> None:
        tracker = torch.full((4,), 0.25)
        ema_update(
            tracker,
            torch.tensor([0, 0]),
            batch_size=2,
            alpha=0.1,
        )
        torch.testing.assert_close(
            tracker,
            torch.tensor([0.325, 0.225, 0.225, 0.225]),
        )
        self.assertAlmostEqual(float(tracker.sum()), 1.0)

    def test_training_smoke(self) -> None:
        policy, tracker, history = train_flat_ema(
            FlatEMAConfig(
                budget=3,
                batch_size=8,
                num_updates=5,
                log_every=10,
                final_samples=16,
            )
        )
        self.assertEqual(policy.shape, (4,))
        self.assertEqual(tracker.shape, (4,))
        self.assertEqual(len(history), 5)
        self.assertAlmostEqual(float(policy.sum()), 1.0)
        self.assertAlmostEqual(float(tracker.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
