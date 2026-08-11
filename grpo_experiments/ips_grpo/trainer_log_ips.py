"""
IPS-GRPO with direct terminal / log-IPS objectives (no group advantages, no PPO clip).

Dispatches to grpo_experiments.core.loss_* via policy_loss_mode.
The original IPSGRPOTrainer in trainer.py is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import torch

from grpo_experiments.core.advantages import AdvantageRewardMode
from grpo_experiments.core.loss import aggregate_step_entropy
from grpo_experiments.core.trainer import GRPOTrainer, sequence_importance_metrics
from grpo_experiments.ips_grpo.policy_loss_modes import POLICY_LOSS_FN, TERMINAL_POLICY_LOSS_MODES
from grpo_experiments.ips_grpo.trainer import compute_batch_outcome_probs

PolicyLossFn = Callable[..., tuple[torch.Tensor, dict]]

_TERMINAL_POLICY_LOSS_FN: dict[str, PolicyLossFn] = {
    mode: POLICY_LOSS_FN[mode] for mode in TERMINAL_POLICY_LOSS_MODES
}


class IPSLogLossTrainer(GRPOTrainer):
    """IPS terminal objectives selected by policy_loss_mode."""

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 1e-4,
        max_grad_norm: float = 1.0,
        ips_prob_floor: float = 1e-6,
        log_ratio_clamp_max: float = 0.0,
        reward_c: float = 0.0,
        reward_scale: float = 1.0,
        entropy_coef: float = 0.0,
        num_iterations: int = 1,
        advantage_reward_mode: AdvantageRewardMode = "exp_linear",
        policy_loss_mode: str = "log_ips",
        **_ignored,
    ):
        super().__init__(
            params=params,
            lr=lr,
            clip_eps=0.0,
            clip_eps_high=None,
            max_grad_norm=max_grad_norm,
            log_ratio_clamp_max=log_ratio_clamp_max,
            reward_c=reward_c,
            reward_scale=reward_scale,
            entropy_coef=entropy_coef,
            num_iterations=num_iterations,
            advantage_reward_mode=advantage_reward_mode,
        )
        self.ips_prob_floor = ips_prob_floor
        if policy_loss_mode not in _TERMINAL_POLICY_LOSS_FN:
            raise ValueError(
                f"Unsupported policy_loss_mode {policy_loss_mode!r}. "
                f"Choose from: {sorted(_TERMINAL_POLICY_LOSS_FN)}."
            )
        self.policy_loss_mode = policy_loss_mode

    def log_p_hat_from_outcome_ids(
        self,
        outcome_ids: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict]:
        p_hat, metrics = compute_batch_outcome_probs(outcome_ids, self.ips_prob_floor)
        p_hat = p_hat.to(device=device, dtype=dtype)
        log_p_hat = p_hat.clamp(min=self.ips_prob_floor).log()
        metrics = dict(metrics)
        metrics["mean_log_p_hat"] = float(log_p_hat.mean().item())
        metrics["ips_mode"] = "log_ips"
        return log_p_hat, metrics

    def precompute_advantages(
        self,
        log_scores: torch.Tensor,
        *,
        outcome_ids: list[str] | None = None,
        log_paths_pf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compatibility hook for on-policy buffers and advantage logging."""
        del log_paths_pf
        if outcome_ids is None:
            raise ValueError("IPS log-loss training requires outcome_ids.")
        if len(outcome_ids) != log_scores.shape[0]:
            raise ValueError(
                f"outcome_ids length ({len(outcome_ids)}) != batch size ({log_scores.shape[0]})."
            )
        log_p_hat, metrics = self.log_p_hat_from_outcome_ids(
            outcome_ids,
            device=log_scores.device,
            dtype=log_scores.dtype,
        )
        pseudo_advantages = (log_scores - log_p_hat).detach()
        metrics["mean_log_ips_terminal"] = float(pseudo_advantages.mean().item())
        return pseudo_advantages, metrics

    def compute_log_ips_loss(
        self,
        log_paths_pf: torch.Tensor,
        log_scores: torch.Tensor,
        log_p_hat: torch.Tensor,
        *,
        log_paths_pf_old: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        loss_fn = _TERMINAL_POLICY_LOSS_FN[self.policy_loss_mode]
        return loss_fn(
            log_paths_pf,
            log_scores,
            log_p_hat,
            log_paths_pf_old=log_paths_pf_old,
            mask=mask,
        )

    def update(
        self,
        log_paths_pf: torch.Tensor,
        log_rewards: torch.Tensor,
        *,
        log_scores: torch.Tensor | None = None,
        outcome_ids: Sequence[str] | None = None,
        log_paths_pf_old: torch.Tensor | None = None,
        log_pf_old: torch.Tensor | None = None,
        fixed_advantages: torch.Tensor | None = None,
        paths_entropy: torch.Tensor | None = None,
        log_paths_pf_old_for_metrics: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        extra_metrics: dict | None = None,
        **_ignored,
    ) -> dict:
        del fixed_advantages, log_pf_old

        if log_scores is None:
            raise ValueError("log_scores is required for IPS log-loss training.")
        if outcome_ids is None:
            raise ValueError("IPS log-loss training requires outcome_ids.")

        self.optimizer.zero_grad()

        log_p_hat, ips_metrics = self.log_p_hat_from_outcome_ids(
            outcome_ids,
            device=log_scores.device,
            dtype=log_scores.dtype,
        )
        merged_metrics = dict(extra_metrics or {})
        merged_metrics.update(ips_metrics)

        pg_loss, loss_metrics = self.compute_log_ips_loss(
            log_paths_pf,
            log_scores,
            log_p_hat,
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
        terminal = (log_scores - log_p_hat).detach()

        out = {
            "loss": float(total_loss.item()),
            "pg_loss": float(pg_loss.item()),
            "entropy_loss": float(entropy_loss.item()),
            "mean_policy_entropy": mean_policy_entropy,
            "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "param_norm": param_norm,
            "mean_advantage": float(terminal.mean().item()),
            "std_advantage": float(terminal.std().item()),
            "mean_log_ips_terminal": float(terminal.mean().item()),
            "std_log_ips_terminal": float(terminal.std().item()),
            "mean_log_pf": float(log_pf.mean().item()),
            "mean_step_logprob": float(log_paths_pf.detach().mean().item()),
            "grpo_group_size": int(log_rewards.shape[0]),
            "num_iterations": self.num_iterations,
            **loss_metrics,
            **merged_metrics,
        }

        metrics_old = log_paths_pf_old_for_metrics if log_paths_pf_old_for_metrics is not None else log_paths_pf_old
        if metrics_old is not None:
            out.update(sequence_importance_metrics(log_paths_pf, metrics_old))

        return out

    def update_on_policy(self, batch: dict, outcome_ids: Sequence[str]) -> dict:
        return self.update(
            batch["log_paths_pf"],
            batch["log_rewards"],
            log_scores=batch["log_scores"],
            outcome_ids=outcome_ids,
            log_paths_pf_old=None,
        )
