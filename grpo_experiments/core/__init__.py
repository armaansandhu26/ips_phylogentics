"""Shared GRPO training primitives (loss, trainer, replay, advantages)."""

from grpo_experiments.core.advantages import (
    AdvantageRewardMode,
    group_advantages,
    linear_rewards_from_log_scores,
    log_rewards_from_scores,
)
from grpo_experiments.core.build_trainer import build_grpo_trainer
from grpo_experiments.core.loss import aggregate_step_entropy, compute_grpo_policy_loss
from grpo_experiments.core.on_policy_buffer import run_on_policy_grpo_step, run_policy_is_grpo_cycles
from grpo_experiments.core.trainer import GRPOTrainer

__all__ = [
    "AdvantageRewardMode",
    "GRPOTrainer",
    "aggregate_step_entropy",
    "build_grpo_trainer",
    "compute_grpo_policy_loss",
    "group_advantages",
    "linear_rewards_from_log_scores",
    "log_rewards_from_scores",
    "run_on_policy_grpo_step",
    "run_policy_is_grpo_cycles",
]
