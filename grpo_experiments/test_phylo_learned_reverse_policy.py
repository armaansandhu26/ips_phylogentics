from __future__ import annotations

import unittest

import torch

from grpo_experiments.learned_reverse_ips_grpo import parse_config
from grpo_experiments.phylo_learned_reverse_policy import (
    PhyloLearnedReversePolicy,
    build_reverse_batch,
    path_log_probabilities,
    reverse_action_mask,
    reverse_context,
    update_mlp_reverse_policy,
)


class _FakeEnv:
    sequences = ["A", "B", "C", "D", "E"]
    log_score_shift = 3600.0
    tree_pairs_dict = {
        5: [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        4: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        3: [(0, 1), (0, 2), (1, 2)],
        2: [(0, 1)],
    }


class PhyloLearnedReversePolicyTest(unittest.TestCase):
    def test_reverse_context_has_nine_features(self) -> None:
        context = reverse_context(
            num_taxa=5,
            step_index=0,
            num_trees_before=5,
            merge_pair=(0, 1),
            terminal_id="topology-a",
            terminal_log_score=-2100.0,
            log_score_shift=3600.0,
        )
        self.assertEqual(len(context), 9)

    def test_uniform_initial_path_probabilities_match_closed_form(self) -> None:
        policy = PhyloLearnedReversePolicy(5, hidden_size=8, num_layers=1)
        action_paths = [
            (0, 0, 0, 0),
            (1, 0, 0, 0),
        ]
        terminal_ids = ["topology-a", "topology-b"]
        log_scores = [-2100.0, -1950.0]
        log_q = path_log_probabilities(
            policy,
            _FakeEnv(),
            action_paths,
            terminal_ids=terminal_ids,
            terminal_log_scores=log_scores,
        )
        self.assertEqual(log_q.shape, (2,))
        self.assertAlmostEqual(float(log_q[0].exp()), 1.0 / 180.0, places=4)

    def test_mle_update_changes_parameters(self) -> None:
        policy = PhyloLearnedReversePolicy(5, hidden_size=8, num_layers=1)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
        batch = build_reverse_batch(
            _FakeEnv(),
            [(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)],
            terminal_ids=["topology-a", "topology-b", "topology-a"],
            terminal_log_scores=[-2100.0, -1950.0, -2050.0],
            device="cpu",
        )
        before = policy.head.weight.detach().clone()
        metrics = update_mlp_reverse_policy(
            policy,
            optimizer,
            batch,
            train_epochs=3,
            grad_clip_norm=1.0,
        )
        self.assertTrue(torch.isfinite(torch.tensor(metrics["reverse_loss"])))
        self.assertFalse(torch.equal(before, policy.head.weight.detach()))

    def test_reverse_mask_width_matches_num_taxa(self) -> None:
        mask = reverse_action_mask(5, max_actions=10)
        self.assertEqual(len(mask), 10)
        self.assertEqual(sum(mask), 10)

    def test_parse_config_accepts_mlp_reverse_settings(self) -> None:
        config = parse_config(
            [
                "--reverse-hidden-size",
                "32",
                "--reverse-num-layers",
                "1",
                "--on-policy-batch-size",
                "64",
            ]
        )
        self.assertEqual(config.reverse_policy_type, "mlp")
        self.assertFalse(config.only_train_tree_model)
        self.assertEqual(config.reverse_hidden_size, 32)
        self.assertEqual(config.reverse_num_layers, 1)


if __name__ == "__main__":
    unittest.main()
