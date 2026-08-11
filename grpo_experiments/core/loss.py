"""TRL-aligned GRPO policy loss (PPO clipped surrogate, token-level IS)."""

from __future__ import annotations

import torch


def compute_grpo_policy_loss(
    log_paths_pf: torch.Tensor,
    advantages: torch.Tensor,
    *,
    log_paths_pf_old: torch.Tensor | None = None,
    clip_eps: float = 0.0,
    clip_eps_high: float | None = None,
    log_ratio_clamp_max: float = 0.0,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """PPO-style GRPO surrogate matching HuggingFace TRL GRPOTrainer."""
    if log_paths_pf.ndim != 2:
        raise ValueError(f"log_paths_pf must be (B, T), got {tuple(log_paths_pf.shape)}.")
    if advantages.ndim != 1:
        raise ValueError(f"advantages must be (B,), got {tuple(advantages.shape)}.")

    if mask is None:
        mask = torch.ones_like(log_paths_pf, dtype=log_paths_pf.dtype)
    else:
        mask = mask.to(dtype=log_paths_pf.dtype)

    if log_paths_pf_old is None:
        old_per_token_logps = log_paths_pf.detach()
    else:
        if log_paths_pf_old.shape != log_paths_pf.shape:
            raise ValueError(
                f"log_paths_pf_old shape {tuple(log_paths_pf_old.shape)} != "
                f"log_paths_pf shape {tuple(log_paths_pf.shape)}."
            )
        old_per_token_logps = log_paths_pf_old.detach()

    log_ratio = log_paths_pf - old_per_token_logps
    log_ratio_raw = log_ratio.detach()
    if log_ratio_clamp_max > 0:
        log_ratio = log_ratio.clamp(-log_ratio_clamp_max, log_ratio_clamp_max)

    coef_1 = log_ratio.exp()
    eps_low = float(clip_eps)
    eps_high = float(clip_eps if clip_eps_high is None else clip_eps_high)
    coef_2 = coef_1.clamp(1.0 - eps_low, 1.0 + eps_high)

    adv = advantages.unsqueeze(1)
    per_token_loss1 = coef_1 * adv
    per_token_loss2 = coef_2 * adv
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    token_counts = mask.sum(dim=-1).clamp(min=1.0)
    loss = ((per_token_loss * mask).sum(dim=-1) / token_counts).mean()

    with torch.no_grad():
        seq_log_ratio = (log_ratio_raw * mask).sum(dim=-1)
        seq_weights = seq_log_ratio.exp()
        masked_coef = coef_1 * mask
        metrics = {
            "policy_loss_mode": "ppo",
            "mean_log_importance_ratio": float((log_ratio_raw * mask).sum().item() / mask.sum().clamp(min=1.0).item()),
            "std_log_importance_ratio": float(log_ratio_raw.std().item()),
            "mean_log_importance_ratio_raw": float(log_ratio_raw.mean().item()),
            "max_log_importance_ratio_raw": float(log_ratio_raw.max().item()),
            "mean_importance_ratio": float(masked_coef.sum().item() / mask.sum().clamp(min=1.0).item()),
            "max_importance_ratio": float(masked_coef.max().item()),
            "min_importance_ratio": float(masked_coef[mask.bool()].min().item()) if mask.bool().any() else 1.0,
            "mean_sequence_importance_ratio": float(seq_weights.mean().item()),
            "clip_ratio/region_mean": float(
                ((coef_1 != coef_2) & (mask.bool())).float().sum().item() / mask.sum().clamp(min=1.0).item()
            ),
        }

    return loss, metrics


def aggregate_step_entropy(paths_entropy: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-sequence mean entropy, then batch mean (matches policy loss aggregation)."""
    if mask is None:
        return paths_entropy.mean()
    token_counts = mask.sum(dim=-1).clamp(min=1.0)
    return ((paths_entropy * mask).sum(dim=-1) / token_counts).mean()
