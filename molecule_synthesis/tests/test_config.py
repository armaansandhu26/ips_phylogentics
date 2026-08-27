from __future__ import annotations

import unittest

from molecule_synthesis.config import list_suites, load_suite
from molecule_synthesis.methods import normalize_method_name


class ConfigSmokeTest(unittest.TestCase):
    def test_bundled_suites_validate(self):
        self.assertIn("qed_smoke", list_suites())
        self.assertIn("qed_cpu_pilot", list_suites())
        self.assertIn("seh_small", list_suites())
        self.assertIn("seh_reduced_a100", list_suites())
        self.assertIn("seh_paper_main", list_suites())
        self.assertIn("seh_paper_medium", list_suites())
        for name in list_suites():
            suite = load_suite(name)
            self.assertGreater(suite.training["iterations"], 0)
            self.assertGreaterEqual(suite.training["forward_trajectories"], 2)
            self.assertGreater(suite.sampling["n_samples"], 0)
        self.assertIsNotNone(load_suite("qed_cpu_pilot").enumeration)
        paper_suite = load_suite("seh_paper_main")
        self.assertEqual(paper_suite.seeds, (0, 1, 2))
        self.assertEqual(
            paper_suite.method_overrides["rgfn"]["replay_trajectories"], 20
        )
        self.assertEqual(paper_suite.evaluation["mode_threshold"], 7.0)
        reduced_suite = load_suite("seh_reduced_a100")
        self.assertEqual(reduced_suite.seeds, (0, 1, 2))
        self.assertEqual(reduced_suite.training["max_reactions"], 2)
        self.assertEqual(reduced_suite.training["iterations"], 1200)
        self.assertEqual(reduced_suite.sampling["n_samples"], 20000)
        self.assertEqual(
            reduced_suite.method_overrides["rgfn"]["replay_trajectories"], 13
        )
        medium_suite = load_suite("seh_paper_medium")
        self.assertEqual(medium_suite.seeds, (0, 1, 2))
        self.assertEqual(medium_suite.training["max_reactions"], 4)
        self.assertEqual(medium_suite.training["iterations"], 2500)
        self.assertEqual(medium_suite.sampling["n_samples"], 50000)
        self.assertEqual(
            medium_suite.method_overrides["rgfn"]["replay_trajectories"], 20
        )

    def test_cli_aliases(self):
        self.assertEqual(normalize_method_name("mips-grpo"), "mips_grpo")
        self.assertEqual(normalize_method_name("ips-grpo"), "count_ips_grpo")
        self.assertEqual(normalize_method_name("count-ips"), "count_ips_grpo")


if __name__ == "__main__":
    unittest.main()
