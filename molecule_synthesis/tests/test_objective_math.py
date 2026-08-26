from __future__ import annotations

import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ObjectiveMathTest(unittest.TestCase):
    def test_ppo_loss_has_a_policy_gradient(self):
        import torch

        from molecule_synthesis.objective_math import compute_grpo_policy_loss

        current = torch.tensor([[-0.4, -0.8], [-0.2, -1.0]], requires_grad=True)
        advantages = torch.tensor([1.0, -1.0])
        loss, _ = compute_grpo_policy_loss(current, advantages, log_paths_pf_old=current.detach())
        loss.backward()
        self.assertIsNotNone(current.grad)
        self.assertTrue(torch.isfinite(current.grad).all())


if __name__ == "__main__":
    unittest.main()
