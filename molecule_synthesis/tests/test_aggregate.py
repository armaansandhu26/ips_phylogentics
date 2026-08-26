from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from molecule_synthesis.aggregate import aggregate_suite


class AggregateSmokeTest(unittest.TestCase):
    def test_aggregate_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite_dir = Path(tmp) / "qed_smoke"
            run_dir = suite_dir / "mips_grpo" / "test"
            sample_dir = run_dir / "samples"
            sample_dir.mkdir(parents=True)
            (suite_dir / "suite.json").write_text(
                json.dumps({"runs": {"mips_grpo": str(run_dir)}}), encoding="utf-8"
            )
            (sample_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "n_sampled": 8,
                        "valid_fraction": 1.0,
                        "n_unique": 7,
                        "unique_fraction": 0.875,
                        "mean_log_reward": -1.0,
                        "mean_proxy": 0.7,
                    }
                ),
                encoding="utf-8",
            )
            json_path, csv_path = aggregate_suite(suite_dir)
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertIn("mips_grpo", csv_path.read_text(encoding="utf-8"))
            self.assertTrue((suite_dir / "results" / "comparison_summary.csv").is_file())

    def test_aggregate_accepts_seeded_run_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite_dir = Path(tmp) / "seh"
            runs = {}
            for seed, value in ((0, 0.8), (1, 0.6)):
                run_dir = suite_dir / "mips_grpo" / f"seed_{seed}"
                sample_dir = run_dir / "samples"
                sample_dir.mkdir(parents=True)
                (sample_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "n_sampled": 100,
                            "valid_fraction": 1.0,
                            "n_unique": 90,
                            "unique_fraction": 0.9,
                            "mean_log_reward": 1.0,
                            "mean_proxy": value,
                            "importance_ess_fraction": 0.9,
                        }
                    ),
                    encoding="utf-8",
                )
                runs[str(seed)] = str(run_dir)
            (suite_dir / "suite.json").write_text(
                json.dumps({"runs": {"mips_grpo": runs}}), encoding="utf-8"
            )
            aggregate_suite(suite_dir)
            summary = json.loads(
                (suite_dir / "results" / "comparison_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary[0]["n_seeds"], 2)
            self.assertAlmostEqual(summary[0]["mean_proxy_mean"], 0.7)


if __name__ == "__main__":
    unittest.main()
