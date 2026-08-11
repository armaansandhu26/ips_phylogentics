"""Group-relative advantages for GRPO.

Reward transform is centralized here so runners stay thin. Default ``exp_linear``
matches historical behavior: exp(log_reward - max) then batch mean/std normalization.
Switch to ``log_reward`` after inspecting advantage distributions (see next_steps.md).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

AdvantageRewardMode = Literal["exp_linear", "log_reward"]


def log_rewards_from_scores(
    log_scores: torch.Tensor,
    *,
    reward_c: float,
    reward_scale: float,
) -> torch.Tensor:
    return (reward_c + log_scores.detach()) / reward_scale


def linear_rewards_from_log_scores(
    log_scores: torch.Tensor,
    *,
    reward_c: float,
    reward_scale: float,
    mode: AdvantageRewardMode = "exp_linear",
) -> torch.Tensor:
    log_r = log_rewards_from_scores(log_scores, reward_c=reward_c, reward_scale=reward_scale)
    if mode == "exp_linear":
        return torch.exp(log_r - log_r.max())
    if mode == "log_reward":
        return log_r
    raise ValueError(f"Unknown advantage reward mode: {mode!r}")


def group_advantages(rewards: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    r = rewards.detach()
    return (r - r.mean()) / (r.std() + eps)


def advantage_distribution_stats(
    values: torch.Tensor,
    *,
    prefix: str = "advantage",
) -> dict[str, float]:
    """Within-group shape statistics for advantages, rewards, etc."""
    arr = values.detach().float().cpu().numpy().reshape(-1)
    if arr.size == 0:
        return {}
    qs = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    std = float(arr.std())
    mean = float(arr.mean())
    skew = 0.0
    if std > 0:
        skew = float(((arr - mean) ** 3).mean() / (std**3))
    return {
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_p05": float(qs[0]),
        f"{prefix}_p25": float(qs[1]),
        f"{prefix}_p50": float(qs[2]),
        f"{prefix}_p75": float(qs[3]),
        f"{prefix}_p95": float(qs[4]),
        f"{prefix}_skew": skew,
        f"{prefix}_frac_below_median": float((arr < qs[2]).mean()),
        f"{prefix}_frac_near_floor": float((arr < -0.12).mean()),
        f"{prefix}_n_positive": float((arr > 0).sum()),
        f"{prefix}_n_negative": float((arr < 0).sum()),
    }
