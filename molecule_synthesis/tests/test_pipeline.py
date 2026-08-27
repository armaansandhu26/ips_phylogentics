from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from molecule_synthesis.config import load_suite
from molecule_synthesis.pipeline import build_parser, build_train_command
from molecule_synthesis.train import _method_bindings, build_parser as build_train_parser
from molecule_synthesis.upstream import validate_rgfn_root


class PipelineSmokeTest(unittest.TestCase):
    def test_accepts_wandb_mode_override(self):
        args = build_parser().parse_args(
            ["--suite", "seh_reduced_a100", "--wandb-mode", "disabled"]
        )
        self.assertEqual(args.wandb_mode, "disabled")

    def test_builds_every_method_command(self):
        suite = load_suite("qed_smoke")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "smoke.gin"
            config.touch()
            for method in suite.methods:
                command = build_train_command(
                    python="python",
                    method=method,
                    cfg=config,
                    rgfn_root=root,
                    output_root=root / "runs",
                    run_name=f"qed_smoke/{method}/test",
                    training=suite.training,
                )
                self.assertIn(method, command)
                self.assertIn("--reverse-loss-weight", command)
                self.assertNotIn("None", command)

    def test_upstream_contract_accepts_minimal_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (
                "pyproject.toml",
                "train.py",
                "rgfn/api/trajectories.py",
                "configs/rgfn_base.gin",
                "data/chemistry.xlsx",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            validate_rgfn_root(root)

    def test_mips_exploration_uses_exact_mixture_sampler(self):
        args = build_train_parser().parse_args(
            [
                "--method",
                "mips_grpo",
                "--cfg",
                "test.gin",
                "--exploration-rate",
                "0.05",
            ]
        )
        bindings = _method_bindings(args, "mips_grpo")
        self.assertIn("train/forward/RandomSampler.policy=%train_forward_policy", bindings)
        self.assertIn("ExploratoryPolicy.first_policy_weight=0.95", bindings)
        self.assertIn("MIPSGRPOObjective.exploration_rate=0.05", bindings)

    def test_paper_mips_configuration_is_frozen_and_on_policy(self):
        suite = load_suite("seh_paper_main")
        training = dict(suite.training)
        training.update(suite.method_overrides["mips_grpo"])
        self.assertEqual(training["iterations"], 4000)
        self.assertEqual(training["forward_trajectories"], 100)
        self.assertEqual(training["replay_trajectories"], 0)
        self.assertEqual(training["max_reactions"], 4)
        self.assertEqual(training["learning_rate"], 1e-4)
        self.assertEqual(training["reverse_learning_rate"], 1e-3)
        self.assertEqual(training["reverse_train_epochs"], 4)
        self.assertEqual(training["advantage_normalization"], "running")
        self.assertEqual(training["running_scale_decay"], 0.9)
        self.assertEqual(training["exploration_rate"], 0.0)

        args = build_train_parser().parse_args(
            [
                "--method",
                "mips_grpo",
                "--cfg",
                "test.gin",
                "--learning-rate",
                str(training["learning_rate"]),
                "--reverse-learning-rate",
                str(training["reverse_learning_rate"]),
                "--reverse-train-epochs",
                str(training["reverse_train_epochs"]),
                "--exploration-rate",
                str(training["exploration_rate"]),
            ]
        )
        bindings = _method_bindings(args, "mips_grpo")
        self.assertIn("train/forward/RandomSampler.policy=%forward_policy", bindings)
        self.assertIn("MIPSOptimizer.forward_lr=0.0001", bindings)
        self.assertIn("MIPSOptimizer.reverse_lr=0.001", bindings)
        self.assertIn("MIPSOptimizer.reverse_train_epochs=4", bindings)
        self.assertIn("MIPSGRPOObjective.exploration_rate=0.0", bindings)
        self.assertFalse(any("ExploratoryPolicy" in binding for binding in bindings))


if __name__ == "__main__":
    unittest.main()
