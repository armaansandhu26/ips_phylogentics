from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Sequence

import torch

from grpo_experiments.tree_edge_ips_v2.config import PropensityMode
from grpo_experiments.tree_edge_ips_v2.data import Episode, OutcomeKey


@dataclass(frozen=True)
class IPSMetrics:
    ips_mode: str
    weight_mean: float
    weight_max: float
    weight_min: float
    snips_weight_mean: float
    snips_weight_std: float
    snips_weight_max: float
    snips_weight_min: float
    ess: float
    ess_fraction: float
    clipped_fraction: float
    weight_temperature: float
    solved_temperature: float | None
    snips_truncate_ratio: float | None
    target_ess_fraction: float | None
    active: float
    log_prop_min: float
    log_prop_max: float
    scaled_return_mean: float
    scaled_return_std: float
    advantage_mean: float
    advantage_std: float
    extra: dict[str, float] = field(default_factory=dict)

    def as_flat_dict(self) -> dict[str, float | str | None]:
        data = asdict(self)
        extra = data.pop("extra", {}) or {}
        data.update(extra)
        return data


def linear_rewards_from_log_scores(
    log_scores: torch.Tensor,
    *,
    reward_c: float,
    reward_scale: float,
    mode: str,
) -> torch.Tensor:
    if mode == "raw":
        return log_scores
    if mode == "log_reward":
        return log_scores
    if mode != "exp_linear":
        raise ValueError(f"Unsupported reward mode: {mode}")
    return ((reward_c + log_scores) / reward_scale).clamp(min=0.0)


def _ess_fraction_at_temperature(log_p_tau: torch.Tensor, beta: float) -> float:
    tiny = torch.finfo(log_p_tau.dtype).tiny
    log_w = -float(beta) * log_p_tau
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

    ESS fraction decreases in beta (beta -> 0 is uniform => ESS 1), so we bisect
    for the most-exact beta that still clears the target.
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

    Pure ``exp(-log pi(tau))`` is heavy-tailed on phylo trajectories: an absolute
    ``max_inverse_weight`` cap either binds for every sample (inert == plain GRPO)
    while the policy is diffuse, or lets a rare trajectory dominate SNIPS once the
    policy sharpens. Prefer ``target_ess_fraction`` (e.g. 0.5) or a fixed
    ``weight_temperature`` < 1 over the legacy absolute-cap path.
    """
    if log_p_tau.ndim != 1:
        raise ValueError("log_p_tau must be a 1D tensor.")
    if log_p_tau.numel() == 0:
        raise ValueError("Cannot compute IPS weights for an empty group.")

    tiny = torch.finfo(log_p_tau.dtype).tiny
    n = log_p_tau.numel()
    log_prop = -log_p_tau.detach()

    solved_beta = None
    if target_ess_fraction is not None:
        solved_beta = solve_temperature_for_ess(log_p_tau.detach(), target_ess_fraction)
        weight_temperature = solved_beta

    legacy = weight_temperature == 1.0 and snips_truncate_ratio is None and target_ess_fraction is None
    if legacy:
        # Historical raw-exp + absolute-cap (often inert on phylo).
        raw_weights = torch.exp(log_prop)
        cap = torch.as_tensor(max_inverse_weight, dtype=raw_weights.dtype, device=raw_weights.device)
        weights = torch.minimum(raw_weights, cap)
        snips = weights * (float(n) / weights.sum().clamp(min=tiny))
        clipped_fraction = float((raw_weights > cap).to(torch.float32).mean().item())
        pre_snips = weights
    else:
        beta = float(weight_temperature)
        log_w = -beta * log_p_tau.detach()
        log_w = log_w - (torch.logsumexp(log_w, dim=0) - float(torch.log(torch.tensor(float(n), device=log_w.device))))
        snips = log_w.exp()
        if snips_truncate_ratio is not None:
            trunc = torch.as_tensor(float(snips_truncate_ratio), dtype=snips.dtype, device=snips.device)
            snips = torch.minimum(snips, trunc)
            snips = snips * (float(n) / snips.sum().clamp(min=tiny))
            clipped_fraction = float((snips >= float(snips_truncate_ratio)).to(torch.float32).mean().item())
        else:
            clipped_fraction = 0.0
        pre_snips = snips

    ess = snips.sum().square() / snips.square().sum().clamp(min=tiny)
    snips_std = float(snips.std(unbiased=False).item())
    # ips_active ~ 0 when SNIPS is nearly uniform (inert); grows as weights vary.
    ips_active = float(snips_std / max(abs(float(snips.mean().item())), tiny))

    metrics = {
        "ips_mode": "exact",
        "weight_mean": float(pre_snips.mean().item()),
        "weight_max": float(pre_snips.max().item()),
        "weight_min": float(pre_snips.min().item()),
        "snips_weight_mean": float(snips.mean().item()),
        "snips_weight_std": snips_std,
        "snips_weight_max": float(snips.max().item()),
        "snips_weight_min": float(snips.min().item()),
        "ess": float(ess.item()),
        "ess_fraction": float((ess / max(n, 1)).item()),
        "clipped_fraction": float(clipped_fraction),
        "weight_temperature": float(weight_temperature),
        "solved_temperature": float(solved_beta) if solved_beta is not None else None,
        "snips_truncate_ratio": float(snips_truncate_ratio) if snips_truncate_ratio is not None else None,
        "target_ess_fraction": float(target_ess_fraction) if target_ess_fraction is not None else None,
        "active": ips_active,
        "log_prop_min": float(log_prop.min().item()),
        "log_prop_max": float(log_prop.max().item()),
        "legacy_absolute_cap": float(1.0 if legacy else 0.0),
    }
    return snips, metrics


def compute_ips_weights(
    log_prob_joint: torch.Tensor,
    *,
    outcome_ids: Sequence[OutcomeKey] | None = None,
    mode: PropensityMode = "exact",
    max_inverse_weight: float = 2560.0,
    count_eps: float = 1e-8,
    weight_temperature: float = 1.0,
    snips_truncate_ratio: float | None = None,
    target_ess_fraction: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Return SNIPS (mean-1) inverse propensities for one GRPO group."""
    if mode == "exact":
        return snips_exact_weights(
            log_prob_joint,
            max_inverse_weight=max_inverse_weight,
            weight_temperature=weight_temperature,
            snips_truncate_ratio=snips_truncate_ratio,
            target_ess_fraction=target_ess_fraction,
        )
    if mode != "count":
        raise ValueError(f"Unsupported propensity mode: {mode}")
    if outcome_ids is None:
        raise ValueError("outcome_ids are required for count propensity mode.")
    if len(outcome_ids) != int(log_prob_joint.numel()):
        raise ValueError("outcome_ids length must match log_prob_joint.")

    counts = Counter(outcome_ids)
    n = float(len(outcome_ids))
    probs = torch.as_tensor(
        [max(counts[outcome] / n, count_eps) for outcome in outcome_ids],
        dtype=log_prob_joint.dtype,
        device=log_prob_joint.device,
    )
    raw = 1.0 / probs
    cap = torch.as_tensor(max_inverse_weight, dtype=raw.dtype, device=raw.device)
    clipped = torch.minimum(raw, cap)
    tiny = torch.finfo(clipped.dtype).tiny
    snips = clipped * (float(len(outcome_ids)) / clipped.sum().clamp(min=tiny))
    ess = snips.sum().square() / snips.square().sum().clamp(min=tiny)
    snips_std = float(snips.std(unbiased=False).item())
    return snips, {
        "ips_mode": "count",
        "weight_mean": float(clipped.mean().item()),
        "weight_max": float(clipped.max().item()),
        "weight_min": float(clipped.min().item()),
        "snips_weight_mean": float(snips.mean().item()),
        "snips_weight_std": snips_std,
        "snips_weight_max": float(snips.max().item()),
        "snips_weight_min": float(snips.min().item()),
        "ess": float(ess.item()),
        "ess_fraction": float((ess / max(len(outcome_ids), 1)).item()),
        "clipped_fraction": float((raw > cap).to(torch.float32).mean().item()),
        "weight_temperature": 1.0,
        "solved_temperature": None,
        "snips_truncate_ratio": None,
        "target_ess_fraction": None,
        "active": float(snips_std / max(abs(float(snips.mean().item())), tiny)),
        "log_prop_min": float("nan"),
        "log_prop_max": float("nan"),
        "legacy_absolute_cap": 0.0,
    }


def normalize_advantages(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = values.mean()
    std = values.std(unbiased=False)
    if float(std.item()) < eps:
        return values - mean
    return (values - mean) / (std + eps)


def compute_group_advantages(
    returns: torch.Tensor,
    log_prob_joint: torch.Tensor,
    *,
    outcome_ids: Sequence[OutcomeKey] | None = None,
    propensity_mode: PropensityMode = "exact",
    max_inverse_weight: float = 2560.0,
    count_eps: float = 1e-8,
    advantage_eps: float = 1e-8,
    weight_temperature: float = 1.0,
    snips_truncate_ratio: float | None = None,
    target_ess_fraction: float | None = None,
) -> tuple[torch.Tensor, IPSMetrics]:
    """Compute SNIPS-scaled trajectory advantages for one group."""
    if returns.shape != log_prob_joint.shape:
        raise ValueError("returns and log_prob_joint must have the same shape.")
    snips_weights, weight_metrics = compute_ips_weights(
        log_prob_joint,
        outcome_ids=outcome_ids,
        mode=propensity_mode,
        max_inverse_weight=max_inverse_weight,
        count_eps=count_eps,
        weight_temperature=weight_temperature,
        snips_truncate_ratio=snips_truncate_ratio,
        target_ess_fraction=target_ess_fraction,
    )
    scaled_returns = returns * snips_weights.to(dtype=returns.dtype)
    advantages = normalize_advantages(scaled_returns, eps=advantage_eps)

    metrics = IPSMetrics(
        ips_mode=str(weight_metrics["ips_mode"]),
        weight_mean=float(weight_metrics["weight_mean"]),
        weight_max=float(weight_metrics["weight_max"]),
        weight_min=float(weight_metrics["weight_min"]),
        snips_weight_mean=float(weight_metrics["snips_weight_mean"]),
        snips_weight_std=float(weight_metrics["snips_weight_std"]),
        snips_weight_max=float(weight_metrics["snips_weight_max"]),
        snips_weight_min=float(weight_metrics["snips_weight_min"]),
        ess=float(weight_metrics["ess"]),
        ess_fraction=float(weight_metrics["ess_fraction"]),
        clipped_fraction=float(weight_metrics["clipped_fraction"]),
        weight_temperature=float(weight_metrics["weight_temperature"]),
        solved_temperature=weight_metrics.get("solved_temperature"),
        snips_truncate_ratio=weight_metrics.get("snips_truncate_ratio"),
        target_ess_fraction=weight_metrics.get("target_ess_fraction"),
        active=float(weight_metrics["active"]),
        log_prop_min=float(weight_metrics["log_prop_min"]),
        log_prop_max=float(weight_metrics["log_prop_max"]),
        scaled_return_mean=float(scaled_returns.mean().item()),
        scaled_return_std=float(scaled_returns.std(unbiased=False).item()),
        advantage_mean=float(advantages.mean().item()),
        advantage_std=float(advantages.std(unbiased=False).item()),
        extra={"legacy_absolute_cap": float(weight_metrics.get("legacy_absolute_cap", 0.0))},
    )
    return advantages, metrics


def assign_group_advantages(
    episodes: list[Episode],
    *,
    propensity_mode: PropensityMode = "exact",
    max_inverse_weight: float = 2560.0,
    count_eps: float = 1e-8,
    advantage_eps: float = 1e-8,
    weight_temperature: float = 1.0,
    snips_truncate_ratio: float | None = None,
    target_ess_fraction: float | None = None,
    device: torch.device | str | None = None,
) -> IPSMetrics:
    returns = torch.as_tensor([ep.return_ for ep in episodes], dtype=torch.float32, device=device)
    log_prob_joint = torch.as_tensor([ep.log_prob_joint for ep in episodes], dtype=torch.float32, device=device)
    advantages, metrics = compute_group_advantages(
        returns,
        log_prob_joint,
        outcome_ids=[ep.outcome for ep in episodes],
        propensity_mode=propensity_mode,
        max_inverse_weight=max_inverse_weight,
        count_eps=count_eps,
        advantage_eps=advantage_eps,
        weight_temperature=weight_temperature,
        snips_truncate_ratio=snips_truncate_ratio,
        target_ess_fraction=target_ess_fraction,
    )
    for episode, advantage in zip(episodes, advantages.detach().cpu().tolist()):
        episode.set_trajectory_advantage(float(advantage))
    return metrics
