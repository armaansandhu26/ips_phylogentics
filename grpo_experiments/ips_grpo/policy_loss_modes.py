"""Registry for --policy-loss-mode: five objective functions, one flag."""

from __future__ import annotations

from typing import Literal

from grpo_experiments.core.loss import compute_grpo_policy_loss
from grpo_experiments.core.loss_log_ips import compute_log_ips_policy_loss
from grpo_experiments.core.loss_magnitude_weighted_ppo import compute_magnitude_weighted_ppo_policy_loss
from grpo_experiments.core.loss_split_ppo import compute_split_ppo_policy_loss
from grpo_experiments.core.loss_terminal_seq_pf import compute_terminal_seq_pf_loss
from grpo_experiments.core.loss_terminal_seq_ratio import compute_terminal_seq_ratio_loss
from grpo_experiments.core.loss_terminal_token_ratio import compute_terminal_token_ratio_loss

PolicyLossMode = Literal[
    "ppo",
    "split_ppo",
    "magnitude_weighted_ppo",
    "tempered_log_ips",
    "log_ips",
    "terminal_seq_pf",
    "terminal_token_ratio",
    "terminal_seq_ratio",
]

ALL_POLICY_LOSS_MODES: tuple[PolicyLossMode, ...] = (
    "ppo",
    "split_ppo",
    "magnitude_weighted_ppo",
    "tempered_log_ips",
    "log_ips",
    "terminal_seq_pf",
    "terminal_token_ratio",
    "terminal_seq_ratio",
)

# IPS-scaled group advantages + TRL PPO surrogate (core/loss.py via IPSGRPOTrainer).
PPO_POLICY_LOSS_MODES = frozenset({"ppo"})

# Tree/edge split-credit PPO variants (core/loss_split_ppo.py, core/loss_magnitude_weighted_ppo.py).
SPLIT_CREDIT_POLICY_LOSS_MODES = frozenset({"split_ppo", "magnitude_weighted_ppo"})

# Tempered log-space IPS advantages + TRL PPO surrogate (core/advantages_tempered_log_ips.py).
TEMPERED_LOG_IPS_POLICY_LOSS_MODES = frozenset({"tempered_log_ips"})

# Direct terminal / log-IPS objectives (core/loss_*.py via IPSLogLossTrainer).
TERMINAL_POLICY_LOSS_MODES = frozenset({"log_ips", "terminal_seq_pf", "terminal_token_ratio", "terminal_seq_ratio"})

POLICY_LOSS_FN = {
    "ppo": compute_grpo_policy_loss,
    "split_ppo": compute_split_ppo_policy_loss,
    "magnitude_weighted_ppo": compute_magnitude_weighted_ppo_policy_loss,
    "log_ips": compute_log_ips_policy_loss,
    "terminal_seq_pf": compute_terminal_seq_pf_loss,
    "terminal_token_ratio": compute_terminal_token_ratio_loss,
    "terminal_seq_ratio": compute_terminal_seq_ratio_loss,
}
