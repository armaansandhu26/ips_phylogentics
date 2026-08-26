"""Dependency-light GRPO policy loss shared by the molecule objectives.

The formula intentionally mirrors `grpo_experiments/core/loss.py`. Keeping the
adapter local avoids importing the phylogenetics environment (ETE/fvcore) into
RGFN's environment.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_grpo_policy_loss(
    log_paths_pf: Tensor,
    advantages: Tensor,
    *,
    log_paths_pf_old: Tensor | None = None,
    clip_eps: float = 0.2,
    clip_eps_high: float | None = None,
    log_ratio_clamp_max: float = 0.0,
    mask: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """TRL-style token/action-level clipped PPO surrogate."""
    if log_paths_pf.ndim != 2:
        raise ValueError(f"log_paths_pf must be (B, T), got {tuple(log_paths_pf.shape)}")
    if advantages.ndim != 1 or advantages.shape[0] != log_paths_pf.shape[0]:
        raise ValueError("advantages must have shape (B,)")
    mask = torch.ones_like(log_paths_pf) if mask is None else mask.to(log_paths_pf.dtype)
    old = log_paths_pf.detach() if log_paths_pf_old is None else log_paths_pf_old.detach()
    if old.shape != log_paths_pf.shape:
        raise ValueError("old and current action log probabilities must have the same shape")

    raw_log_ratio = log_paths_pf - old
    log_ratio = raw_log_ratio
    if log_ratio_clamp_max > 0:
        log_ratio = log_ratio.clamp(-log_ratio_clamp_max, log_ratio_clamp_max)
    ratio = log_ratio.exp()
    high = clip_eps if clip_eps_high is None else clip_eps_high
    clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + high)
    advantage = advantages.unsqueeze(1)
    per_action_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
    action_counts = mask.sum(dim=-1).clamp_min(1.0)
    loss = ((per_action_loss * mask).sum(dim=-1) / action_counts).mean()

    with torch.no_grad():
        valid = mask.bool()
        metrics = {
            "mean_log_importance_ratio": float(
                (raw_log_ratio * mask).sum().item() / mask.sum().clamp_min(1.0).item()
            ),
            "mean_importance_ratio": float(
                (ratio * mask).sum().item() / mask.sum().clamp_min(1.0).item()
            ),
            "clip_fraction": float(((ratio != clipped_ratio) & valid).sum().item() / valid.sum().clamp_min(1).item()),
        }
    return loss, metrics
