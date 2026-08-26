from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from molecule_synthesis.config import load_suite
from molecule_synthesis.pipeline import build_train_command
from molecule_synthesis.upstream import validate_rgfn_root


class PipelineSmokeTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
