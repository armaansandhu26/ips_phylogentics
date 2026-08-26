from __future__ import annotations

import unittest

from molecule_synthesis.evaluation import exact_distribution_metrics


class ExactDistributionMetricsTest(unittest.TestCase):
    def setUp(self):
        self.target = {
            "outcomes": [
                {"smiles": "A", "target_probability": 0.25, "qed": 0.5},
                {"smiles": "B", "target_probability": 0.75, "qed": 0.8},
            ]
        }

    def test_exact_match_has_zero_distribution_error(self):
        rows = [
            *[self._row("A", 0.5) for _ in range(25)],
            *[self._row("B", 0.8) for _ in range(75)],
        ]
        metrics = exact_distribution_metrics(rows, self.target)
        self.assertAlmostEqual(metrics["tv_to_reward_target"], 0.0)
        self.assertAlmostEqual(metrics["js_to_reward_target"], 0.0)
        self.assertAlmostEqual(metrics["target_mass_covered"], 1.0)
        self.assertAlmostEqual(metrics["log_probability_calibration_slope"], 1.0)
        self.assertAlmostEqual(metrics["top_10_unique_mean_reward"], 1.0)

    def test_top_k_reward_deduplicates_repeated_molecules(self):
        rows = [
            *[self._row("A", 0.5, reward=10.0) for _ in range(10)],
            self._row("B", 0.8, reward=2.0),
        ]
        metrics = exact_distribution_metrics(rows, self.target)
        self.assertAlmostEqual(metrics["top_10_unique_mean_reward"], 6.0)

    def test_out_of_support_samples_are_penalized(self):
        rows = [self._row("A", 0.5), self._row("C", 0.1)]
        metrics = exact_distribution_metrics(rows, self.target)
        self.assertAlmostEqual(metrics["out_of_support_fraction"], 0.5)
        self.assertGreater(metrics["tv_to_reward_target"], 0.0)

    @staticmethod
    def _row(smiles: str, proxy: float, reward: float = 1.0) -> dict:
        return {"smiles": smiles, "proxy": proxy, "reward": reward}


if __name__ == "__main__":
    unittest.main()
