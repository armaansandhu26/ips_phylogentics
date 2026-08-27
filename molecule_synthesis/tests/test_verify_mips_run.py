from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from molecule_synthesis.verify_mips_run import EXPECTED_BINDINGS, verify


class VerifyMIPSRunTest(unittest.TestCase):
    def test_accepts_completed_frozen_paper_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite_dir = Path(tmp)
            run_dir = suite_dir / "mips_grpo" / "seed_0" / "run"
            (run_dir / "samples").mkdir(parents=True)
            (suite_dir / "suite.json").write_text(
                json.dumps({"runs": {"mips_grpo": {"0": str(run_dir)}}})
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "method": "mips_grpo",
                        "seed": 0,
                        "bindings": sorted(EXPECTED_BINDINGS),
                    }
                )
            )
            (run_dir / "samples" / "summary.json").write_text(
                json.dumps(
                    {
                        "n_requested": 100000,
                        "n_sampled": 100000,
                        "n_valid": 100000,
                        "valid_fraction": 1.0,
                        "n_unique": 20,
                        "mean_proxy": 7.0,
                        "importance_ess_fraction": 0.1,
                        "log_importance_weight_mean": 0.0,
                        "log_importance_weight_std": 1.0,
                        "train_final_reverse_loss": 0.5,
                    }
                )
            )
            verified_run, _ = verify(suite_dir, 0)
            self.assertEqual(verified_run, run_dir)

    def test_rejects_collapsed_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite_dir = Path(tmp)
            run_dir = suite_dir / "run"
            (run_dir / "samples").mkdir(parents=True)
            (suite_dir / "suite.json").write_text(
                json.dumps({"runs": {"mips_grpo": {"0": str(run_dir)}}})
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "method": "mips_grpo",
                        "seed": 0,
                        "bindings": sorted(EXPECTED_BINDINGS),
                    }
                )
            )
            (run_dir / "samples" / "summary.json").write_text(
                json.dumps(
                    {
                        "n_requested": 100000,
                        "n_sampled": 100000,
                        "valid_fraction": 1.0,
                        "n_unique": 1,
                        "mean_proxy": 7.0,
                        "importance_ess_fraction": 0.1,
                        "log_importance_weight_mean": 0.0,
                        "log_importance_weight_std": 1.0,
                        "train_final_reverse_loss": 0.5,
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "collapsed"):
                verify(suite_dir, 0)
