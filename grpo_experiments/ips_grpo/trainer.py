"""
Inverse Probability Scaling GRPO (IPS-GRPO) + optional policy importance sampling.

Outcome IPS (Sinha et al., arXiv:2601.21669):
  p_hat(o) = count(o in batch) / G
  r_tilde_g = r_g / max(p_hat(o_g), eps)
  A_i from r_tilde_i

Policy IS (when log_pf_old is provided):
  w_i = exp(sum_t log pi_new - sum_t log pi_old)
  L = -mean(w_i * A_i * log pi_new per step)
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


def compute_batch_outcome_probs(
    outcome_ids: Sequence[str],
    prob_floor: float,
) -> tuple[torch.Tensor, dict]:
    """p_hat(o_g) for each tree g, clipped at prob_floor."""
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
    log_rewards: torch.Tensor,
    outcome_ids: Sequence[str],
    prob_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """r_tilde_g = r_g / max(p_hat(o_g), eps)  (Eq. 9)."""
    p_hat, metrics = compute_batch_outcome_probs(outcome_ids, prob_floor)
    p_hat = p_hat.to(device=log_rewards.device, dtype=log_rewards.dtype)
    scaled = log_rewards.detach() / p_hat
    metrics["ips_scaled_reward_mean"] = float(scaled.mean().item())
    metrics["ips_scaled_reward_std"] = float(scaled.std().item())
    return scaled, p_hat, metrics


class IPSGRPOTrainer:
    """IPS-GRPO with optional pi_new/pi_old importance weights."""

    def __init__(
        self,
        params: list[nn.Parameter],
        lr: float = 1e-4,
        max_grad_norm: float = 1.0,
        advantage_eps: float = 1e-8,
        ips_prob_floor: float = 0.01,
        is_ratio_clip: float = 0.0,
        is_ratio_max: float = 0.0,
    ):
        self.params = params
        self.max_grad_norm = max_grad_norm
        self.advantage_eps = advantage_eps
        self.ips_prob_floor = ips_prob_floor
        self.is_ratio_clip = is_ratio_clip
        self.is_ratio_max = is_ratio_max
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        r = rewards.detach()
        return (r - r.mean()) / (r.std() + self.advantage_eps)

    def precompute_ips_advantages(
        self,
        log_rewards: torch.Tensor,
        outcome_ids: Sequence[str],
    ) -> tuple[torch.Tensor, dict]:
        """IPS-scaled group advantages fixed for a behavior-policy buffer."""
        if len(outcome_ids) != log_rewards.shape[0]:
            raise ValueError(
                f"outcome_ids length ({len(outcome_ids)}) != batch size ({log_rewards.shape[0]})."
            )
        scaled_rewards, p_hat, ips_metrics = scale_rewards_ips(
            log_rewards, outcome_ids, self.ips_prob_floor,
        )
        advantages = self.compute_advantages(scaled_rewards)
        ips_metrics["mean_ips_prob"] = float(p_hat.mean().item())
        return advantages, ips_metrics

    def importance_weights(
        self,
        log_paths_pf: torch.Tensor,
        log_pf_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_pf_new = log_paths_pf.sum(dim=-1)
        log_ratio = log_pf_new - log_pf_old.detach()
        weights = log_ratio.exp()

        if self.is_ratio_clip > 0:
            weights = torch.clamp(
                weights,
                1.0 - self.is_ratio_clip,
                1.0 + self.is_ratio_clip,
            )
        if self.is_ratio_max > 0:
            weights = torch.clamp(weights, max=self.is_ratio_max)

        return weights, log_ratio

    def update(
        self,
        log_paths_pf: torch.Tensor,
        log_rewards: torch.Tensor,
        outcome_ids: Sequence[str],
        log_pf_old: torch.Tensor | None = None,
        fixed_advantages: torch.Tensor | None = None,
        fixed_ips_metrics: dict | None = None,
    ) -> dict:
        self.optimizer.zero_grad()

        if fixed_advantages is not None:
            advantages = fixed_advantages
            ips_metrics = dict(fixed_ips_metrics or {})
        else:
            scaled_rewards, p_hat, ips_metrics = scale_rewards_ips(
                log_rewards, outcome_ids, self.ips_prob_floor,
            )
            advantages = self.compute_advantages(scaled_rewards)
            ips_metrics = {**ips_metrics, "mean_ips_prob": float(p_hat.mean().item())}

        if log_pf_old is not None:
            weights, log_ratio = self.importance_weights(log_paths_pf, log_pf_old)
        else:
            weights = torch.ones(log_paths_pf.shape[0], device=log_paths_pf.device, dtype=log_paths_pf.dtype)
            log_ratio = torch.zeros_like(weights)

        scale = log_paths_pf.detach().abs().mean().clamp(min=1.0)
        log_pf_scaled = log_paths_pf / scale
        pg_loss = -(weights.detach().unsqueeze(1) * advantages.detach().unsqueeze(1) * log_pf_scaled).mean()

        pg_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

        log_pf = log_paths_pf.sum(dim=-1).detach()
        param_norm = sum(p.data.norm().item() ** 2 for p in self.params) ** 0.5
        ratio = log_ratio.exp().detach()

        out = {
            "loss": float(pg_loss.item()),
            "pg_loss": float(pg_loss.item()),
            "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "param_norm": param_norm,
            "mean_advantage": float(advantages.mean().item()),
            "std_advantage": float(advantages.std().item()),
            "mean_log_pf": float(log_pf.mean().item()),
            "mean_step_logprob": float(log_paths_pf.detach().mean().item()),
            "grpo_group_size": int(log_rewards.shape[0]),
            **ips_metrics,
        }
        if log_pf_old is not None:
            out.update({
                "mean_log_importance_ratio": float(log_ratio.mean().item()),
                "std_log_importance_ratio": float(log_ratio.std().item()),
                "mean_importance_ratio": float(ratio.mean().item()),
                "max_importance_ratio": float(ratio.max().item()),
                "min_importance_ratio": float(ratio.min().item()),
            })
        return out

    def update_on_policy(self, batch: dict, outcome_ids: Sequence[str]) -> dict:
        """One-step on-policy IPS-GRPO (no policy IS)."""
        return self.update(
            batch["log_paths_pf"],
            batch["log_rewards"],
            outcome_ids,
            log_pf_old=None,
        )

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
