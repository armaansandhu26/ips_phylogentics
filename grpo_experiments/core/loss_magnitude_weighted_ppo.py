"""Option 2: magnitude-weighted combination of tree/edge log-probs before PPO surrogate."""

from __future__ import annotations

import torch

from grpo_experiments.core.loss import compute_grpo_policy_loss

POLICY_LOSS_MODE = "magnitude_weighted_ppo"


def magnitude_weights_from_log_probs(
    log_paths_pf_tree: torch.Tensor,
    log_paths_pf_edge: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    weight_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token weights from detached |log p| magnitudes (tree share, edge share)."""
    mag_tree = log_paths_pf_tree.detach().abs()
    mag_edge = log_paths_pf_edge.detach().abs()
    denom = mag_tree + mag_edge + float(weight_eps)
    w_tree = mag_tree / denom
    w_edge = mag_edge / denom
    if mask is not None:
        mask_t = mask.to(dtype=w_tree.dtype)
        w_tree = w_tree * mask_t
        w_edge = w_edge * mask_t
    return w_tree, w_edge


def compute_magnitude_weighted_ppo_policy_loss(
    log_paths_pf_tree: torch.Tensor,
    log_paths_pf_edge: torch.Tensor,
    advantages: torch.Tensor,
    *,
    log_paths_pf_tree_old: torch.Tensor | None = None,
    log_paths_pf_edge_old: torch.Tensor | None = None,
    clip_eps: float = 0.0,
    clip_eps_high: float | None = None,
    log_ratio_clamp_max: float = 0.0,
    mask: torch.Tensor | None = None,
    weight_eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """PPO on w_tree*log_tree + w_edge*log_edge with detached magnitude weights."""
    if log_paths_pf_tree.shape != log_paths_pf_edge.shape:
        raise ValueError(
            f"log_paths_pf_tree shape {tuple(log_paths_pf_tree.shape)} != "
            f"log_paths_pf_edge shape {tuple(log_paths_pf_edge.shape)}."
        )

    if log_paths_pf_tree_old is None:
        ref_tree = log_paths_pf_tree.detach()
    else:
        ref_tree = log_paths_pf_tree_old.detach()
    if log_paths_pf_edge_old is None:
        ref_edge = log_paths_pf_edge.detach()
    else:
        ref_edge = log_paths_pf_edge_old.detach()

    w_tree, w_edge = magnitude_weights_from_log_probs(
        ref_tree,
        ref_edge,
        mask=mask,
        weight_eps=weight_eps,
    )

    log_combined = w_tree * log_paths_pf_tree + w_edge * log_paths_pf_edge
    log_combined_old = w_tree * ref_tree + w_edge * ref_edge

    loss, metrics = compute_grpo_policy_loss(
        log_combined,
        advantages,
        log_paths_pf_old=log_combined_old,
        clip_eps=clip_eps,
        clip_eps_high=clip_eps_high,
        log_ratio_clamp_max=log_ratio_clamp_max,
        mask=mask,
    )
    metrics = dict(metrics)
    metrics["policy_loss_mode"] = POLICY_LOSS_MODE
    metrics["magnitude_weight_eps"] = float(weight_eps)

    with torch.no_grad():
        if mask is None:
            mask_t = torch.ones_like(log_combined, dtype=log_combined.dtype)
        else:
            mask_t = mask.to(dtype=log_combined.dtype)
        denom = mask_t.sum().clamp(min=1.0)
        metrics["mean_tree_magnitude_weight"] = float((w_tree * mask_t).sum().item() / denom.item())
        metrics["mean_edge_magnitude_weight"] = float((w_edge * mask_t).sum().item() / denom.item())
        metrics["mean_log_pf_tree"] = float((log_paths_pf_tree * mask_t).sum().item() / denom.item())
        metrics["mean_log_pf_edge"] = float((log_paths_pf_edge * mask_t).sum().item() / denom.item())

    return loss, metrics
