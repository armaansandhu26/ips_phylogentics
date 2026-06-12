"""Construct GRPOTrainer with shared defaults."""

from __future__ import annotations

import torch.nn as nn

from grpo_experiments.core.advantages import AdvantageRewardMode
from grpo_experiments.core.trainer import GRPOTrainer


def build_grpo_trainer(
    params: list[nn.Parameter],
    *,
    lr: float,
    reward_c: float,
    reward_scale: float,
    clip_eps: float = 0.2,
    clip_eps_high: float | None = None,
    max_grad_norm: float = 1.0,
    advantage_eps: float = 1e-8,
    log_ratio_clamp_max: float = 2.0,
    entropy_coef: float = 0.0,
    num_iterations: int = 1,
    advantage_reward_mode: AdvantageRewardMode = "exp_linear",
) -> GRPOTrainer:
    """Build a GRPOTrainer. ``num_iterations`` (TRL mu) applies to on-policy buffering only."""
    return GRPOTrainer(
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
        advantage_reward_mode=advantage_reward_mode,
    )
