"""Tempered log-space IPS advantages for GRPO.

Per batch of G trajectories with log-reward ell_g = log R_{o_g}:

    p_hat(o) = (1/G) sum_g 1[o_g = o]
    log c_g = ell_g / tau - log max(p_hat(o_g), eps / G)
    log c_g <- log c_g - max_g log c_g
    c_g = exp(log c_g)
    A_g = (c_g - mean(c)) / (std(c) + delta)

Feed A into the standard PPO/GRPO clipped surrogate (core/loss.py).
Temperature tau defaults to std(ell) / 3 per batch when not provided.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import torch

from grpo_experiments.core.advantages import group_advantages, log_rewards_from_scores


def _batch_outcome_probs(outcome_ids: Sequence[str]) -> tuple[torch.Tensor, dict]:
    n = len(outcome_ids)
    if n == 0:
        raise ValueError("outcome_ids must be non-empty for tempered log IPS-GRPO.")

    counts = Counter(outcome_ids)
    probs = np.array([counts[oid] / n for oid in outcome_ids], dtype=np.float64)
    metrics = {
        "ips_prob_mean": float(probs.mean()),
        "ips_prob_min": float(probs.min()),
        "ips_prob_max": float(probs.max()),
        "ips_unique_outcomes": float(len(counts)),
        "ips_max_outcome_count": float(max(counts.values())),
        "ips_min_outcome_count": float(min(counts.values())),
    }
    return torch.tensor(probs, dtype=torch.float64), metrics


def tempered_log_ips_coefficients(
    ell: torch.Tensor,
    p_hat: torch.Tensor,
    *,
    tau: float | None = None,
    tau_divisor: float = 3.0,
    tau_floor: float = 1e-8,
    ips_prob_floor: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Map log-rewards and outcome frequencies to tempered IPS coefficients c."""
    if ell.ndim != 1 or p_hat.ndim != 1:
        raise ValueError(f"ell and p_hat must be 1-D, got {ell.shape} and {p_hat.shape}.")
    if ell.shape[0] != p_hat.shape[0]:
        raise ValueError(f"ell batch ({ell.shape[0]}) != p_hat batch ({p_hat.shape[0]}).")

    ell = ell.detach()
    p_hat = p_hat.detach().to(device=ell.device, dtype=ell.dtype)
    group_size = ell.shape[0]

    if tau is None:
        tau = float(ell.std().item()) / tau_divisor
    tau = max(float(tau), tau_floor)

    phat_floor = ips_prob_floor / group_size
    logc = ell / tau - p_hat.clamp(min=phat_floor).log()
    logc = logc - logc.max()
    c = logc.exp()

    metrics = {
        "tempered_log_ips_tau": tau,
        "tempered_log_ips_mean_c": float(c.mean().item()),
        "tempered_log_ips_std_c": float(c.std().item()),
        "tempered_log_ips_mean_logc": float(logc.mean().item()),
    }
    return c, metrics


def compute_tempered_log_ips_advantages(
    log_scores: torch.Tensor,
    outcome_ids: Sequence[str],
    *,
    reward_c: float,
    reward_scale: float,
    ips_prob_floor: float = 1e-6,
    tau: float | None = None,
    tau_divisor: float = 3.0,
    advantage_eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """IPS-corrected tempered advantages for GRPO policy loss."""
    if len(outcome_ids) != log_scores.shape[0]:
        raise ValueError(
            f"outcome_ids length ({len(outcome_ids)}) != batch size ({log_scores.shape[0]})."
        )

    p_hat, metrics = _batch_outcome_probs(outcome_ids)
    p_hat = p_hat.to(device=log_scores.device, dtype=log_scores.dtype)

    ell = log_rewards_from_scores(
        log_scores,
        reward_c=reward_c,
        reward_scale=reward_scale,
    )
    c, coeff_metrics = tempered_log_ips_coefficients(
        ell,
        p_hat,
        tau=tau,
        tau_divisor=tau_divisor,
        ips_prob_floor=ips_prob_floor,
    )
    advantages = group_advantages(c, eps=advantage_eps)

    metrics = dict(metrics)
    metrics.update(coeff_metrics)
    metrics["ips_mode"] = "tempered_log_ips"
    metrics["mean_ell"] = float(ell.mean().item())
    metrics["std_ell"] = float(ell.std().item())
    metrics["mean_ips_prob"] = float(p_hat.mean().item())
    return advantages, metrics
