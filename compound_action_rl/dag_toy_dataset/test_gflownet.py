from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from config import TrainConfig
from gflownet import TrajectoryBalanceGFlowNet, trajectory_balance_loss


class TrajectoryBalanceGFlowNetTest(unittest.TestCase):
    def test_trajectory_balance_loss_is_zero_at_equality(self) -> None:
        log_pf = torch.tensor([-1.2, -0.7])
        log_pb = torch.tensor([-0.8, -0.3])
        log_rewards = torch.tensor([-0.4, -0.4])
        # Both rows satisfy logZ + logPF = logR + logPB for logZ=0.
        loss, residual = trajectory_balance_loss(
            torch.tensor(0.0), log_pf, log_pb, log_rewards
        )
        torch.testing.assert_close(residual, torch.zeros(2))
        self.assertAlmostEqual(float(loss.item()), 0.0, places=12)

    def test_training_and_checkpoint_smoke(self) -> None:
        config = TrainConfig(
            budget=3,
            max_step=3,
            hidden_size=16,
            num_layers=1,
            group_size=16,
            num_groups=1,
            num_updates=2,
            log_every=10,
        )
        trainer = TrajectoryBalanceGFlowNet(config, z_lr=1e-2)
        initial_log_z = float(trainer.log_z.item())
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            history = trainer.train(eval_every=2, eval_episodes=64)

        self.assertEqual(len(history), 2)
        self.assertIn(
            f"global_outcomes={history[0]['global_unique_outcomes']:.0f}",
            stdout.getvalue(),
        )
        self.assertTrue(np.isfinite(history[-1]["loss"]))
        self.assertTrue(np.isfinite(history[-1]["tb_residual_abs_mean"]))
        self.assertNotEqual(float(trainer.log_z.item()), initial_log_z)
        self.assertEqual(sum(history[-1]["eval_outcome_counts"].values()), 64)

        with TemporaryDirectory() as directory:
            checkpoint = trainer.save(Path(directory) / "checkpoint.pt", update_step=2)
            restored = TrajectoryBalanceGFlowNet.load(checkpoint)
            self.assertAlmostEqual(
                float(restored.log_z.item()), float(trainer.log_z.item())
            )
            evaluation = restored.evaluate(32)
            self.assertEqual(sum(evaluation["eval_outcome_counts"].values()), 32)


if __name__ == "__main__":
    unittest.main()
