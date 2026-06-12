"""
Inverse Probability Scaling GRPO (IPS-GRPO) + optional policy importance sampling.

Policy loss: TRL token-level PPO surrogate in grpo_experiments.core.trainer.
Only reward/advantage computation differs from core GRPO.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import torch

from grpo_experiments.core.advantages import linear_rewards_from_log_scores
from grpo_experiments.core.trainer import GRPOTrainer


def compute_batch_outcome_probs(
    outcome_ids: Sequence[str],
    prob_floor: float,
) -> tuple[torch.Tensor, dict]:
    n = len(outcome_ids)
    if n == 0:
        raise ValueError("outcome_ids must be non-empty for IPS-GRPO.")

    counts = Counter(outcome_ids)
    probs = np.array([counts[oid] / n for oid in outcome_ids], dtype=np.float64)
    clipped = np.maximum(probs, prob_floor)

    metrics = {
        "ips_prob_mean": float(clipped.mean()),
        "ips_prob_min": float(clipped.min()),
        "ips_prob_max": float(clipped.max()),
        "ips_unique_outcomes": float(len(counts)),
        "ips_max_outcome_count": float(max(counts.values())),
        "ips_min_outcome_count": float(min(counts.values())),
    }
    return torch.tensor(clipped, dtype=torch.float64), metrics


def scale_rewards_ips(
    log_scores: torch.Tensor,
    outcome_ids: Sequence[str],
    prob_floor: float,
    reward_c: float,
    reward_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    p_hat, metrics = compute_batch_outcome_probs(outcome_ids, prob_floor)
    p_hat = p_hat.to(device=log_scores.device, dtype=log_scores.dtype)
    rewards = linear_rewards_from_log_scores(
        log_scores,
        reward_c=reward_c,
        reward_scale=reward_scale,
        mode="exp_linear",
    )
    scaled = rewards / p_hat
    metrics["ips_scaled_reward_mean"] = float(scaled.mean().item())
    metrics["ips_scaled_reward_std"] = float(scaled.std().item())
    return scaled, p_hat, metrics


class IPSGRPOTrainer(GRPOTrainer):
    """IPS-scaled advantages + TRL-style token-level policy loss."""

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 1e-4,
        max_grad_norm: float = 1.0,
        advantage_eps: float = 1e-8,
        ips_prob_floor: float = 1e-6,
        clip_eps: float = 0.2,
        clip_eps_high: float | None = None,
        log_ratio_clamp_max: float = 2.0,
        reward_c: float = 0.0,
        reward_scale: float = 1.0,
        entropy_coef: float = 0.0,
        num_iterations: int = 1,
    ):
        super().__init__(
            params=params,
            lr=lr,
            clip_eps=clip_eps,
            clip_eps_high=clip_eps_high,
            max_grad_norm=max_grad_norm,
            advantage_eps=advantage_eps,
            log_ratio_clamp_max=log_ratio_clamp_max,
            reward_c=reward_c,
            reward_scale=reward_scale,
            entropy_coef=entropy_coef,
            num_iterations=num_iterations,
        )
        self.ips_prob_floor = ips_prob_floor

    def precompute_advantages(
        self,
        log_scores: torch.Tensor,
        *,
        outcome_ids: list[str] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if outcome_ids is None:
            advantages = self.compute_advantages(log_scores)
            return advantages, {"ips_mode": "grpo"}
        if len(outcome_ids) != log_scores.shape[0]:
            raise ValueError(
                f"outcome_ids length ({len(outcome_ids)}) != batch size ({log_scores.shape[0]})."
            )
        scaled_rewards, p_hat, ips_metrics = scale_rewards_ips(
            log_scores,
            outcome_ids,
            self.ips_prob_floor,
            reward_c=self.reward_c,
            reward_scale=self.reward_scale,
        )
        advantages = self.compute_advantages_from_rewards(scaled_rewards)
        ips_metrics["mean_ips_prob"] = float(p_hat.mean().item())
        ips_metrics["ips_mode"] = "ips"
        return advantages, ips_metrics

    def precompute_grpo_advantages(self, log_scores: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self.precompute_advantages(log_scores)

    def precompute_ips_advantages(
        self,
        log_scores: torch.Tensor,
        outcome_ids: Sequence[str],
    ) -> tuple[torch.Tensor, dict]:
        return self.precompute_advantages(log_scores, outcome_ids=list(outcome_ids))

    def update(
        self,
        log_paths_pf: torch.Tensor,
        log_rewards: torch.Tensor,
        *,
        log_scores: torch.Tensor | None = None,
        outcome_ids: Sequence[str] | None = None,
        log_paths_pf_old: torch.Tensor | None = None,
        log_pf_old: torch.Tensor | None = None,
        fixed_advantages: torch.Tensor | None = None,
        fixed_ips_metrics: dict | None = None,
        paths_entropy: torch.Tensor | None = None,
        log_paths_pf_old_for_metrics: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        extra_metrics: dict | None = None,
    ) -> dict:
        merged_metrics = dict(extra_metrics or {})
        if fixed_ips_metrics:
            merged_metrics.update(fixed_ips_metrics)

        if fixed_advantages is None:
            if log_scores is None:
                raise ValueError("log_scores is required when fixed_advantages is not provided.")
            if outcome_ids is None:
                raise ValueError("IPS-GRPO requires outcome_ids when fixed_advantages is not provided.")
            fixed_advantages, ips_metrics = self.precompute_advantages(
                log_scores,
                outcome_ids=list(outcome_ids),
            )
            merged_metrics.update(ips_metrics)

        return super().update(
            log_paths_pf,
            log_rewards,
            log_scores=log_scores,
            log_paths_pf_old=log_paths_pf_old,
            log_pf_old=log_pf_old,
            fixed_advantages=fixed_advantages,
            paths_entropy=paths_entropy,
            log_paths_pf_old_for_metrics=log_paths_pf_old_for_metrics,
            mask=mask,
            extra_metrics=merged_metrics or None,
        )

    def update_on_policy(self, batch: dict, outcome_ids: Sequence[str]) -> dict:
        return self.update(
            batch["log_paths_pf"],
            batch["log_rewards"],
            log_scores=batch["log_scores"],
            outcome_ids=outcome_ids,
            log_paths_pf_old=None,
        )
