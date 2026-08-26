from __future__ import annotations

import importlib.util
import unittest

from molecule_synthesis.chemistry_evaluation import molecular_discovery_metrics


@unittest.skipUnless(importlib.util.find_spec("rdkit"), "requires RDKit")
class MolecularDiscoveryMetricsTest(unittest.TestCase):
    def test_leader_modes_deduplicate_similar_molecules(self):
        rows = [
            self._row("c1ccccc1", 8.0),
            self._row("Cc1ccccc1", 7.8),
            self._row("C1CCCCC1", 7.5),
            self._row("CCO", 6.0),
        ]
        metrics, modes = molecular_discovery_metrics(
            rows,
            mode_threshold=7.0,
            similarity_threshold=0.5,
            max_modes=10,
            top_k=10,
            scaffold_thresholds=(7.0,),
        )
        self.assertEqual(metrics["n_mode_candidates"], 3)
        self.assertGreaterEqual(metrics["n_modes"], 2)
        self.assertEqual(len(modes), metrics["n_modes"])
        self.assertIn("top_modes_mean_qed", metrics)
        self.assertGreaterEqual(metrics["n_scaffolds_proxy_gt_7"], 2)

    @staticmethod
    def _row(smiles: str, proxy: float) -> dict:
        return {"smiles": smiles, "proxy": proxy, "reward": proxy}


if __name__ == "__main__":
    unittest.main()
