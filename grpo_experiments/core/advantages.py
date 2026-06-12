"""Group-relative advantages for GRPO.

Reward transform is centralized here so runners stay thin. Default ``exp_linear``
matches historical behavior: exp(log_reward - max) then batch mean/std normalization.
Switch to ``log_reward`` after inspecting advantage distributions (see next_steps.md).
"""

from __future__ import annotations

from typing import Literal

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
