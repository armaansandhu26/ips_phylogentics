"""
Inverse Probability Scaling GRPO (IPS-GRPO) + optional policy importance sampling.

Policy loss: TRL token-level PPO surrogate in grpo_experiments.core.trainer.
Only reward/advantage computation differs from core GRPO.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal, Sequence

import numpy as np
import torch

from grpo_experiments.core.advantages import AdvantageRewardMode, linear_rewards_from_log_scores
from grpo_experiments.core.advantages_tempered_log_ips import compute_tempered_log_ips_advantages
from grpo_experiments.core.trainer import GRPOTrainer

IPSPropensityMode = Literal["count", "exact"]


def compute_batch_outcome_probs(
    outcome_ids: Sequence[str],
    prob_floor: float,
) -> tuple[torch.Tensor, dict]:
    n = len(outcome_ids)
    if n == 0:
        raise ValueError("outcome_ids must be non-empty for IPS-GRPO.")

    counts = Counter(outcome_ids)
    probs = np.array([counts[oid] / n for oid in outcome_ids], dtype=np.float64)
    clipped = np.maximum(probs, prob_floor)

    metrics = {
        "ips_prob_mean": float(clipped.mean()),
        "ips_prob_min": float(clipped.min()),
        "ips_prob_max": float(clipped.max()),
        "ips_unique_outcomes": float(len(counts)),
        "ips_max_outcome_count": float(max(counts.values())),
        "ips_min_outcome_count": float(min(counts.values())),
    }
    return torch.tensor(clipped, dtype=torch.float64), metrics


def scale_rewards_ips(
    log_scores: torch.Tensor,
    outcome_ids: Sequence[str],
    prob_floor: float,
    reward_c: float,
    reward_scale: float,
    mode: AdvantageRewardMode = "exp_linear",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    p_hat, metrics = compute_batch_outcome_probs(outcome_ids, prob_floor)
    p_hat = p_hat.to(device=log_scores.device, dtype=log_scores.dtype)
    rewards = linear_rewards_from_log_scores(
        log_scores,
        reward_c=reward_c,
        reward_scale=reward_scale,
        mode=mode,
    )
    scaled = rewards / p_hat
    metrics["ips_scaled_reward_mean"] = float(scaled.mean().item())
    metrics["ips_scaled_reward_std"] = float(scaled.std().item())
    return scaled, p_hat, metrics


def _ess_fraction_at_temperature(log_p_tau: torch.Tensor, beta: float) -> float:
    tiny = torch.finfo(log_p_tau.dtype).tiny
    log_w = -beta * log_p_tau
    log_w = log_w - log_w.max()
    w = log_w.exp()
    ess = w.sum().square() / w.square().sum().clamp(min=tiny)
    return float((ess / max(log_p_tau.numel(), 1)).item())


def solve_temperature_for_ess(
    log_p_tau: torch.Tensor,
    target_ess_fraction: float,
    *,
    beta_min: float = 1e-3,
    beta_max: float = 1.0,
    iters: int = 30,
) -> float:
    """Largest temperature beta in [beta_min, beta_max] whose SNIPS ESS >= target.

    ESS fraction is monotonically decreasing in beta (beta -> 0 is uniform => ESS 1),
    so we bisect for the boundary and return the most-exact beta that still clears the
    target. This adapts per batch: as the policy sharpens and log pi(tau) spreads out,
    the solved beta shrinks automatically to hold ESS constant.
    """
    target = float(target_ess_fraction)
    if _ess_fraction_at_temperature(log_p_tau, beta_max) >= target:
        return beta_max
    if _ess_fraction_at_temperature(log_p_tau, beta_min) <= target:
        return beta_min
    lo, hi = beta_min, beta_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _ess_fraction_at_temperature(log_p_tau, mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo


def snips_exact_weights(
    log_p_tau: torch.Tensor,
    *,
    max_inverse_weight: float,
    weight_temperature: float = 1.0,
    snips_truncate_ratio: float | None = None,
    target_ess_fraction: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Self-normalized exact inverse-propensity weights (mean 1) for one group.

    ``exp(-log pi(tau))`` has unbounded, heavy-tailed variance in high-dimensional
    trajectory spaces: an absolute ``max_inverse_weight`` cap either binds for every
    sample (inert, == plain GRPO) while the policy is diffuse, or lets a single rare
    trajectory dominate SNIPS once the policy sharpens (ESS collapse). Two knobs tame
    the tail while keeping the estimator exact in expectation up to normalization:

    - ``weight_temperature`` (beta in (0, 1]): raise the inverse propensity to the
      power beta before normalizing. beta == 1 is pure exact IPS; beta -> 0 recovers
      plain GRPO. Lower beta directly shrinks the log-weight spread that drives ESS
      collapse.
    - ``snips_truncate_ratio``: after normalizing to mean 1, clip weights to this
      multiple of the mean and renormalize (truncated importance sampling). Bounds the
      worst-case single-sample influence regardless of how sharp the policy gets.

    All computation is in log-space so it is numerically stable no matter how negative
    ``log pi(tau)`` is. Returns weights normalized to mean 1 plus diagnostics.
    """

    tiny = torch.finfo(log_p_tau.dtype).tiny
    n = log_p_tau.numel()

    solved_beta = None
    if target_ess_fraction is not None:
        solved_beta = solve_temperature_for_ess(log_p_tau, target_ess_fraction)
        weight_temperature = solved_beta

    legacy = weight_temperature == 1.0 and snips_truncate_ratio is None
    if legacy:
        # Preserve historical raw-exp + absolute-cap behavior for reproducibility.
        raw_weights = torch.exp(-log_p_tau)
        cap = torch.as_tensor(max_inverse_weight, dtype=raw_weights.dtype, device=raw_weights.device)
        weights = torch.minimum(raw_weights, cap)
        snips = weights * (float(n) / weights.sum().clamp(min=tiny))
        clipped_fraction = float((raw_weights > cap).to(torch.float32).mean().item())
    else:
        beta = float(weight_temperature)
        # log inverse propensity, tempered, then stable SNIPS (mean of weights == 1).
        log_w = -beta * log_p_tau
        log_w = log_w - (torch.logsumexp(log_w, dim=0) - float(torch.log(torch.tensor(float(n)))))
        snips = log_w.exp()
        if snips_truncate_ratio is not None:
            trunc = torch.as_tensor(float(snips_truncate_ratio), dtype=snips.dtype, device=snips.device)
            snips = torch.minimum(snips, trunc)
            snips = snips * (float(n) / snips.sum().clamp(min=tiny))
        clipped_fraction = (
            float((snips >= float(snips_truncate_ratio)).to(torch.float32).mean().item())
            if snips_truncate_ratio is not None
            else 0.0
        )

    ess = snips.sum().square() / snips.square().sum().clamp(min=tiny)
    metrics = {
        "ips_inverse_weight_mean": float(snips.mean().item()),
        "ips_inverse_weight_min": float(snips.min().item()),
        "ips_inverse_weight_max": float(snips.max().item()),
        "ips_inverse_weight_clipped_fraction": clipped_fraction,
        "ips_weight_temperature": float(weight_temperature),
        "ips_snips_weight_mean": float(snips.mean().item()),
        "ips_ess": float(ess.item()),
        "ips_ess_fraction": float((ess / max(n, 1)).item()),
        "ips_log_prop_min": float((-log_p_tau).min().item()),
        "ips_log_prop_max": float((-log_p_tau).max().item()),
    }
    if solved_beta is not None:
        metrics["ips_solved_temperature"] = float(solved_beta)
    return snips, metrics


def scale_rewards_exact_ips(
    log_scores: torch.Tensor,
    log_paths_pf: torch.Tensor,
    max_inverse_weight: float,
    reward_c: float,
    reward_scale: float,
    mode: AdvantageRewardMode = "exp_linear",
    weight_temperature: float = 1.0,
    snips_truncate_ratio: float | None = None,
    target_ess_fraction: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Scale rewards by exact inverse trajectory propensity with SNIPS normalization."""
    if log_paths_pf.ndim != 2:
        raise ValueError(f"log_paths_pf must be (B, T), got {tuple(log_paths_pf.shape)}.")
    if log_paths_pf.shape[0] != log_scores.shape[0]:
        raise ValueError(
            f"log_paths_pf batch size ({log_paths_pf.shape[0]}) != log_scores ({log_scores.shape[0]})."
        )
    log_p_tau = log_paths_pf.detach().sum(dim=-1)
    snips_weights, weight_metrics = snips_exact_weights(
        log_p_tau,
        max_inverse_weight=max_inverse_weight,
        weight_temperature=weight_temperature,
        snips_truncate_ratio=snips_truncate_ratio,
        target_ess_fraction=target_ess_fraction,
    )

    rewards = linear_rewards_from_log_scores(
        log_scores,
        reward_c=reward_c,
        reward_scale=reward_scale,
        mode=mode,
    )
    scaled = rewards * snips_weights.to(dtype=rewards.dtype)
    effective_probs = 1.0 / snips_weights.clamp(min=torch.finfo(snips_weights.dtype).tiny)
    metrics = {
        "ips_propensity_mode": "exact",
        "ips_prob_mean": float(effective_probs.mean().item()),
        "ips_prob_min": float(effective_probs.min().item()),
        "ips_prob_max": float(effective_probs.max().item()),
        "ips_scaled_reward_mean": float(scaled.mean().item()),
        "ips_scaled_reward_std": float(scaled.std().item()),
    }
    metrics.update(weight_metrics)
    return scaled, effective_probs, metrics


class IPSGRPOTrainer(GRPOTrainer):
    """IPS-scaled advantages + TRL-style token-level policy loss."""

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 1e-4,
        max_grad_norm: float = 1.0,
        advantage_eps: float = 1e-8,
        ips_prob_floor: float = 1e-6,
        clip_eps: float = 0.2,
        clip_eps_high: float | None = None,
        log_ratio_clamp_max: float = 2.0,
        reward_c: float = 0.0,
        reward_scale: float = 1.0,
        entropy_coef: float = 0.0,
        num_iterations: int = 1,
        advantage_reward_mode: AdvantageRewardMode = "exp_linear",
        policy_loss_mode: str = "ppo",
        tree_loss_weight: float = 0.5,
        edge_loss_weight: float = 0.5,
        ips_propensity_mode: IPSPropensityMode = "count",
        max_inverse_weight: float = 2560.0,
        ips_weight_temperature: float = 1.0,
        snips_truncate_ratio: float | None = None,
        ips_target_ess_fraction: float | None = None,
    ):
        super().__init__(
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
            policy_loss_mode=policy_loss_mode,
            tree_loss_weight=tree_loss_weight,
            edge_loss_weight=edge_loss_weight,
        )
        self.ips_prob_floor = ips_prob_floor
        self.ips_propensity_mode = ips_propensity_mode
        self.max_inverse_weight = float(max_inverse_weight)
        self.ips_weight_temperature = float(ips_weight_temperature)
        self.snips_truncate_ratio = (
            float(snips_truncate_ratio) if snips_truncate_ratio is not None else None
        )
        self.ips_target_ess_fraction = (
            float(ips_target_ess_fraction) if ips_target_ess_fraction is not None else None
        )

    def precompute_advantages(
        self,
        log_scores: torch.Tensor,
        *,
        outcome_ids: list[str] | None = None,
        log_paths_pf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if outcome_ids is None:
            advantages = self.compute_advantages(log_scores)
            return advantages, {"ips_mode": "grpo"}
        if self.ips_propensity_mode == "exact":
            if log_paths_pf is None:
                raise ValueError("Exact IPS propensity requires log_paths_pf.")
            scaled_rewards, p_hat, ips_metrics = scale_rewards_exact_ips(
                log_scores,
                log_paths_pf,
                self.max_inverse_weight,
                reward_c=self.reward_c,
                reward_scale=self.reward_scale,
                mode=self.advantage_reward_mode,
                weight_temperature=self.ips_weight_temperature,
                snips_truncate_ratio=self.snips_truncate_ratio,
                target_ess_fraction=self.ips_target_ess_fraction,
            )
            advantages = self.compute_advantages_from_rewards(scaled_rewards)
            ips_metrics["mean_ips_prob"] = float(p_hat.mean().item())
            ips_metrics["ips_mode"] = "ips_exact"
            return advantages, ips_metrics
        if len(outcome_ids) != log_scores.shape[0]:
            raise ValueError(
                f"outcome_ids length ({len(outcome_ids)}) != batch size ({log_scores.shape[0]})."
            )
        scaled_rewards, p_hat, ips_metrics = scale_rewards_ips(
            log_scores,
            outcome_ids,
            self.ips_prob_floor,
            reward_c=self.reward_c,
            reward_scale=self.reward_scale,
            mode=self.advantage_reward_mode,
        )
        advantages = self.compute_advantages_from_rewards(scaled_rewards)
        ips_metrics["mean_ips_prob"] = float(p_hat.mean().item())
        ips_metrics["ips_propensity_mode"] = "count"
        ips_metrics["ips_mode"] = "ips_count"
        return advantages, ips_metrics

    def precompute_grpo_advantages(self, log_scores: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self.precompute_advantages(log_scores)

    def precompute_ips_advantages(
        self,
        log_scores: torch.Tensor,
        outcome_ids: Sequence[str],
        log_paths_pf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        return self.precompute_advantages(
            log_scores,
            outcome_ids=list(outcome_ids),
            log_paths_pf=log_paths_pf,
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
        fixed_ips_metrics: dict | None = None,
        paths_entropy: torch.Tensor | None = None,
        log_paths_pf_old_for_metrics: torch.Tensor | None = None,
        log_paths_pf_tree: torch.Tensor | None = None,
        log_paths_pf_edge: torch.Tensor | None = None,
        log_paths_pf_tree_old: torch.Tensor | None = None,
        log_paths_pf_edge_old: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        extra_metrics: dict | None = None,
        ips_log_paths_pf: torch.Tensor | None = None,
    ) -> dict:
        merged_metrics = dict(extra_metrics or {})
        if fixed_ips_metrics:
            merged_metrics.update(fixed_ips_metrics)

        if fixed_advantages is None:
            if log_scores is None:
                raise ValueError("log_scores is required when fixed_advantages is not provided.")
            if outcome_ids is None:
                raise ValueError("IPS-GRPO requires outcome_ids when fixed_advantages is not provided.")
            fixed_advantages, ips_metrics = self.precompute_advantages(
                log_scores,
                outcome_ids=list(outcome_ids),
                log_paths_pf=ips_log_paths_pf if ips_log_paths_pf is not None else log_paths_pf_old,
            )
            merged_metrics.update(ips_metrics)

        return super().update(
            log_paths_pf,
            log_rewards,
            log_scores=log_scores,
            log_paths_pf_old=log_paths_pf_old,
            log_pf_old=log_pf_old,
            fixed_advantages=fixed_advantages,
            paths_entropy=paths_entropy,
            log_paths_pf_old_for_metrics=log_paths_pf_old_for_metrics,
            log_paths_pf_tree=log_paths_pf_tree,
            log_paths_pf_edge=log_paths_pf_edge,
            log_paths_pf_tree_old=log_paths_pf_tree_old,
            log_paths_pf_edge_old=log_paths_pf_edge_old,
            mask=mask,
            extra_metrics=merged_metrics or None,
        )

    def update_on_policy(self, batch: dict, outcome_ids: Sequence[str]) -> dict:
        return self.update(
            batch["log_paths_pf"],
            batch["log_rewards"],
            log_scores=batch["log_scores"],
            outcome_ids=outcome_ids,
            log_paths_pf_old=None,
        )


class TemperedLogIPSGRPOTrainer(IPSGRPOTrainer):
    """Tempered log-space IPS advantages + TRL-style token-level PPO surrogate."""

    def __init__(
        self,
        *args,
        tempered_ips_tau: float | None = None,
        tempered_ips_tau_divisor: float = 3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tempered_ips_tau = tempered_ips_tau
        self.tempered_ips_tau_divisor = tempered_ips_tau_divisor

    def precompute_advantages(
        self,
        log_scores: torch.Tensor,
        *,
        outcome_ids: list[str] | None = None,
        log_paths_pf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        del log_paths_pf
        if outcome_ids is None:
            raise ValueError("Tempered log IPS-GRPO requires outcome_ids.")
        advantages, metrics = compute_tempered_log_ips_advantages(
            log_scores,
            outcome_ids,
            reward_c=self.reward_c,
            reward_scale=self.reward_scale,
            ips_prob_floor=self.ips_prob_floor,
            tau=self.tempered_ips_tau,
            tau_divisor=self.tempered_ips_tau_divisor,
            advantage_eps=self.advantage_eps,
        )
        return advantages, metrics
