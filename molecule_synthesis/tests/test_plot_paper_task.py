import json
import tempfile
import unittest
from pathlib import Path

from molecule_synthesis.plot_paper_task import (
    _forward_trajectories,
    _result_caption,
    _suite_label,
)


class PlotPaperTaskTest(unittest.TestCase):
    def test_reduced_suite_has_reduced_space_label(self):
        self.assertEqual(
            _suite_label({"suite": "seh_reduced_a100"}), "reduced-space sEH"
        )

    def test_forward_trajectories_come_from_run_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "bindings": [
                            "Trainer.n_iterations=1200",
                            "Trainer.train_forward_n_trajectories=64",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(_forward_trajectories(run_dir), 64)

    def test_result_caption_does_not_imply_replication_for_one_seed(self):
        self.assertEqual(_result_caption({"rgfn": [{"seed": 0}]}), "seed 0")
        self.assertEqual(
            _result_caption({"rgfn": [{"seed": 0}, {"seed": 1}]}),
            "mean ± SD over 2 seeds",
        )


if __name__ == "__main__":
    unittest.main()
