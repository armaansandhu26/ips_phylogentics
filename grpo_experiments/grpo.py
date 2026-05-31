"""
Group Relative Policy Optimization (GRPO) for phylogenetic tree sampling.

Objective
---------
Sample G trees {tau_i}, compute group-relative advantages on log-rewards, and
update the forward policy:

    L = - mean_i( A_i * mean_t log P_F(a_{i,t} | s_{i,t}) )

where A_i = (r_i - mean(r)) / (std(r) + eps), r_i = log R(x_i).

Optional policy importance sampling (--enable-policy-is): sample trajectories under
a frozen behavior policy pi_old, then update with:

    w_i = exp( sum_t log pi_new(a_{i,t}|s_{i,t}) - sum_t log pi_old(a_{i,t}|s_{i,t}) )
    L = - mean_i( w_i * A_i * mean_t log pi_new(a_{i,t}|s_{i,t}) )

We average log-probs over trajectory steps (not sum) so loss magnitude stays O(1)
for long phylogenetic trajectories (~26 steps). Gradient direction is unchanged
up to a constant factor.

Reference: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models" (GRPO); applied here to PhyloGFN's
tree-construction policy instead of LLM token generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRPOTrainer:
    """GRPO trainer with optional policy importance weights (pi_new / pi_old)."""

    def __init__(
        self,
        params: list[nn.Parameter],
        lr: float = 1e-4,
        clip_eps: float = 0.0,
        beta: float = 0.0,
        max_grad_norm: float = 1.0,
        advantage_eps: float = 1e-8,
        is_ratio_clip: float = 0.0,
        is_ratio_max: float = 0.0,
    ):
        self.params = params
        self.clip_eps = clip_eps
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.advantage_eps = advantage_eps
        self.is_ratio_clip = is_ratio_clip
        self.is_ratio_max = is_ratio_max
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def compute_advantages(self, log_rewards: torch.Tensor) -> torch.Tensor:
        """Group-relative advantages: A_i = (r_i - mean) / (std + eps)."""
        rewards = log_rewards.detach()
        return (rewards - rewards.mean()) / (rewards.std() + self.advantage_eps)

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
        *,
        log_pf_old: torch.Tensor | None = None,
        fixed_advantages: torch.Tensor | None = None,
    ) -> dict:
        self.optimizer.zero_grad()

        if fixed_advantages is not None:
            advantages = fixed_advantages
        else:
            advantages = self.compute_advantages(log_rewards)

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

    def update_on_policy(self, batch: dict) -> dict:
        """One-step on-policy GRPO (w = 1)."""
        return self.update(batch["log_paths_pf"], batch["log_rewards"])

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
