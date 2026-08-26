from __future__ import annotations

import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ToyPipelineTest(unittest.TestCase):
    def test_mips_recovers_molecule_target(self):
        import torch

        from molecule_synthesis.toy.pipeline import train_method

        torch.set_num_threads(1)
        result = train_method(
            "mips_grpo",
            steps=100,
            batch_size=512,
            learning_rate=0.05,
            seed=0,
            device=torch.device("cpu"),
        )
        self.assertLess(result["l1_to_reward_target"], 0.05)


if __name__ == "__main__":
    unittest.main()
