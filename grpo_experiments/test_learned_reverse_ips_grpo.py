from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from grpo_experiments.learned_reverse_ips_grpo import (
    RunningLogWeightNormalizer,
    TabularTerminalReversePolicy,
    parse_config,
    terminal_log_rewards_from_scores,
    update_reverse_policy,
)


class LearnedReverseIPSGRPOTest(unittest.TestCase):
    def test_reward_targets_have_explicit_log_reward_semantics(self) -> None:
        scores = torch.tensor([2.0, 4.0])
        likelihood = terminal_log_rewards_from_scores(
            scores,
            reward_target="likelihood",
            reward_c=0.0,
            reward_scale=1.0,
        )
        shifted_linear = terminal_log_rewards_from_scores(
            scores,
            reward_target="shifted_linear",
            reward_c=0.0,
            reward_scale=1.0,
        )
        torch.testing.assert_close(likelihood, scores)
        torch.testing.assert_close(shifted_linear, scores.log())

    def test_shifted_linear_requires_positive_shifted_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            terminal_log_rewards_from_scores(
                torch.tensor([-1.0, 2.0]),
                reward_target="shifted_linear",
                reward_c=0.0,
                reward_scale=1.0,
            )

    def test_always_uses_full_model(self) -> None:
        config = parse_config(
            [
                "--outcome-level",
                "signature",
                "--on-policy-batch-size",
                "4096",
            ]
        )
        self.assertFalse(config.only_train_tree_model)
        self.assertEqual(config.reverse_policy_type, "mlp")
        self.assertEqual(config.outcome_level, "signature")
        self.assertEqual(config.on_policy_batch_size, 4096)

    def test_reward_target_flag_is_saved_in_config(self) -> None:
        config = parse_config(["--reward-target", "shifted_linear"])
        self.assertEqual(config.reward_target, "shifted_linear")

    def test_reverse_policy_is_normalized_per_terminal(self) -> None:
        policy = TabularTerminalReversePolicy(
            [(0, 0), (0, 1), (1, 0)],
            ["terminal-a", "terminal-a", "terminal-b"],
        )
        with torch.no_grad():
            policy.logits.copy_(torch.tensor([1.5, -0.25, 3.0]))
        log_q = policy.all_log_probabilities()
        self.assertAlmostEqual(float(log_q[:2].exp().sum()), 1.0, places=6)
        self.assertAlmostEqual(float(log_q[2].exp()), 1.0, places=6)
        self.assertLess(policy.normalization_error(), 1e-6)

    def test_reverse_mle_increases_probability_of_observed_path(self) -> None:
        policy = TabularTerminalReversePolicy(
            [(0, 0), (0, 1)],
            ["terminal-a", "terminal-a"],
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
        observed = torch.zeros(32, dtype=torch.long)
        before = float(policy.log_prob(observed[:1]).exp())
        update_reverse_policy(
            policy,
            optimizer,
            observed,
            train_epochs=8,
            grad_clip_norm=1.0,
        )
        after = float(policy.log_prob(observed[:1]).exp())
        self.assertGreater(after, before)
        self.assertLess(policy.normalization_error(), 1e-6)

    def test_running_normalizer_preserves_cross_batch_severity(self) -> None:
        mild = RunningLogWeightNormalizer()
        severe = RunningLogWeightNormalizer()
        baseline = np.zeros(16)
        mild.normalize(baseline)
        severe.normalize(baseline)

        mild_weights = np.zeros(16)
        mild_weights[0] = math.log(2.0)
        severe_weights = np.zeros(16)
        severe_weights[0] = math.log(1000.0)
        mild_advantages, _ = mild.normalize(mild_weights)
        severe_advantages, metrics = severe.normalize(severe_weights)

        self.assertAlmostEqual(float(mild_advantages[0]), 1.0)
        self.assertEqual(float(severe_advantages[0]), 10.0)
        self.assertEqual(metrics["running_advantage_clip_fraction"], 1.0 / 16.0)


if __name__ == "__main__":
    unittest.main()
