"""
Group Relative Policy Optimization (GRPO) for phylogenetic tree sampling.

TRL-aligned objective (loss_type="grpo"):
    r_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
    L = -mean_i( sum_t min(r_t A_i, clip(r_t) A_i) / T_i )

Entropy regularization (we use this instead of TRL beta / KL-to-reference):
    L = L_policy - entropy_coef * mean_{i,t} H(pi_new(·|s_{i,t}))
"""

from __future__ import annotations

import torch
import torch.nn as nn

from grpo_experiments.core.advantages import (
    AdvantageRewardMode,
    group_advantages,
    linear_rewards_from_log_scores,
)
from grpo_experiments.core.loss import aggregate_step_entropy, compute_grpo_policy_loss


def sequence_importance_metrics(
    log_paths_pf: torch.Tensor,
    log_paths_pf_old: torch.Tensor,
) -> dict[str, float]:
    """Sequence-level IS diagnostics (training loss uses token-level ratios)."""
    log_ratio_raw = log_paths_pf.sum(dim=-1) - log_paths_pf_old.sum(dim=-1).detach()
    return {
        "mean_log_pf_old": float(log_paths_pf_old.sum(dim=-1).mean().item()),
        "mean_step_logprob_old": float(log_paths_pf_old.detach().mean().item()),
        "mean_log_importance_ratio_seq": float(log_ratio_raw.mean().item()),
        "mean_log_importance_ratio_raw_seq": float(log_ratio_raw.mean().item()),
    }


class GRPOTrainer:
    """GRPO trainer with TRL-style PPO surrogate and token-level importance ratios."""

    def __init__(
        self,
        params: list[nn.Parameter],
        lr: float = 1e-4,
        clip_eps: float = 0.2,
        clip_eps_high: float | None = None,
        max_grad_norm: float = 1.0,
        advantage_eps: float = 1e-8,
        log_ratio_clamp_max: float = 2.0,
        reward_c: float = 0.0,
        reward_scale: float = 1.0,
        entropy_coef: float = 0.0,
        num_iterations: int = 1,
        advantage_reward_mode: AdvantageRewardMode = "exp_linear",
    ):
        self.params = params
        self.clip_eps = clip_eps
        self.clip_eps_high = clip_eps_high
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = max_grad_norm
        self.advantage_eps = advantage_eps
        self.log_ratio_clamp_max = log_ratio_clamp_max
        self.reward_c = float(reward_c)
        self.reward_scale = float(reward_scale)
        self.advantage_reward_mode = advantage_reward_mode
        self.num_iterations = max(1, int(num_iterations))
        if self.reward_scale == 0:
            raise ValueError("reward_scale must be non-zero.")
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def batch_rewards(self, log_scores: torch.Tensor) -> torch.Tensor:
        return linear_rewards_from_log_scores(
            log_scores,
            reward_c=self.reward_c,
            reward_scale=self.reward_scale,
            mode=self.advantage_reward_mode,
        )

    def compute_advantages(self, log_scores: torch.Tensor) -> torch.Tensor:
        rewards = self.batch_rewards(log_scores)
        return group_advantages(rewards, eps=self.advantage_eps)

    def compute_advantages_from_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return group_advantages(rewards, eps=self.advantage_eps)

    def precompute_advantages(
        self,
        log_scores: torch.Tensor,
        *,
        outcome_ids: list[str] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Batch advantages for buffering (μ reuse / policy IS). Subclasses may scale rewards first."""
        del outcome_ids
        return self.compute_advantages(log_scores), {}

    def compute_policy_loss(
        self,
        log_paths_pf: torch.Tensor,
        advantages: torch.Tensor,
        *,
        log_paths_pf_old: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        return compute_grpo_policy_loss(
            log_paths_pf,
            advantages,
            log_paths_pf_old=log_paths_pf_old,
            clip_eps=self.clip_eps,
            clip_eps_high=self.clip_eps_high,
            log_ratio_clamp_max=self.log_ratio_clamp_max,
            mask=mask,
        )

    def update(
        self,
        log_paths_pf: torch.Tensor,
        log_rewards: torch.Tensor,
        *,
        log_scores: torch.Tensor | None = None,
        log_paths_pf_old: torch.Tensor | None = None,
        log_pf_old: torch.Tensor | None = None,
        fixed_advantages: torch.Tensor | None = None,
        paths_entropy: torch.Tensor | None = None,
        log_paths_pf_old_for_metrics: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        extra_metrics: dict | None = None,
    ) -> dict:
        self.optimizer.zero_grad()

        if fixed_advantages is not None:
            advantages = fixed_advantages
        else:
            if log_scores is None:
                raise ValueError("log_scores is required when fixed_advantages is not provided.")
            advantages, precomputed_metrics = self.precompute_advantages(log_scores)
            if extra_metrics is None:
                extra_metrics = precomputed_metrics
            else:
                extra_metrics = {**precomputed_metrics, **extra_metrics}

        pg_loss, loss_metrics = self.compute_policy_loss(
            log_paths_pf,
            advantages,
            log_paths_pf_old=log_paths_pf_old,
            mask=mask,
        )

        entropy_loss = torch.zeros((), device=log_paths_pf.device, dtype=log_paths_pf.dtype)
        mean_policy_entropy = 0.0
        if paths_entropy is not None and self.entropy_coef > 0:
            mean_policy_entropy = float(aggregate_step_entropy(paths_entropy, mask).item())
            entropy_loss = -self.entropy_coef * aggregate_step_entropy(paths_entropy, mask)

        total_loss = pg_loss + entropy_loss
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

        log_pf = log_paths_pf.sum(dim=-1).detach()
        param_norm = sum(p.data.norm().item() ** 2 for p in self.params) ** 0.5

        out = {
            "loss": float(total_loss.item()),
            "pg_loss": float(pg_loss.item()),
            "entropy_loss": float(entropy_loss.item()),
            "mean_policy_entropy": mean_policy_entropy,
            "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "param_norm": param_norm,
            "mean_advantage": float(advantages.mean().item()),
            "std_advantage": float(advantages.std().item()),
            "mean_log_pf": float(log_pf.mean().item()),
            "mean_step_logprob": float(log_paths_pf.detach().mean().item()),
            "grpo_group_size": int(log_rewards.shape[0]),
            "grpo_clip_eps": float(self.clip_eps),
            "num_iterations": self.num_iterations,
            **loss_metrics,
        }
        if extra_metrics:
            out.update(extra_metrics)

        metrics_old = log_paths_pf_old_for_metrics if log_paths_pf_old_for_metrics is not None else log_paths_pf_old
        if metrics_old is not None:
            out.update(sequence_importance_metrics(log_paths_pf, metrics_old))
        elif log_pf_old is not None:
            out["mean_log_pf_old"] = float(log_pf_old.detach().mean().item())

        return out

    def update_on_policy(self, batch: dict, *, log_paths_pf_old: torch.Tensor | None = None) -> dict:
        return self.update(
            batch["log_paths_pf"],
            batch["log_rewards"],
            log_scores=batch["log_scores"],
            log_paths_pf_old=log_paths_pf_old,
        )

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
