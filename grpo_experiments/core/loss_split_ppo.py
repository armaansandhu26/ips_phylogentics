"""Option 1: separate PPO clipped surrogates on tree and edge log-prob trajectories."""

from __future__ import annotations

import torch

from grpo_experiments.core.loss import compute_grpo_policy_loss

POLICY_LOSS_MODE = "split_ppo"


def compute_split_ppo_policy_loss(
    log_paths_pf_tree: torch.Tensor,
    log_paths_pf_edge: torch.Tensor,
    advantages: torch.Tensor,
    *,
    log_paths_pf_tree_old: torch.Tensor | None = None,
    log_paths_pf_edge_old: torch.Tensor | None = None,
    tree_weight: float = 0.5,
    edge_weight: float = 0.5,
    clip_eps: float = 0.0,
    clip_eps_high: float | None = None,
    log_ratio_clamp_max: float = 0.0,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Independent token-level PPO on tree and edge components, then weighted sum."""
    if log_paths_pf_tree.shape != log_paths_pf_edge.shape:
        raise ValueError(
            f"log_paths_pf_tree shape {tuple(log_paths_pf_tree.shape)} != "
            f"log_paths_pf_edge shape {tuple(log_paths_pf_edge.shape)}."
        )

    tree_loss, tree_metrics = compute_grpo_policy_loss(
        log_paths_pf_tree,
        advantages,
        log_paths_pf_old=log_paths_pf_tree_old,
        clip_eps=clip_eps,
        clip_eps_high=clip_eps_high,
        log_ratio_clamp_max=log_ratio_clamp_max,
        mask=mask,
    )
    edge_loss, edge_metrics = compute_grpo_policy_loss(
        log_paths_pf_edge,
        advantages,
        log_paths_pf_old=log_paths_pf_edge_old,
        clip_eps=clip_eps,
        clip_eps_high=clip_eps_high,
        log_ratio_clamp_max=log_ratio_clamp_max,
        mask=mask,
    )

    loss = float(tree_weight) * tree_loss + float(edge_weight) * edge_loss

    metrics = {
        "policy_loss_mode": POLICY_LOSS_MODE,
        "tree_loss_weight": float(tree_weight),
        "edge_loss_weight": float(edge_weight),
        "tree_pg_loss": float(tree_loss.item()),
        "edge_pg_loss": float(edge_loss.item()),
    }
    for key, value in tree_metrics.items():
        if key == "policy_loss_mode":
            continue
        metrics[f"tree_{key}"] = value
    for key, value in edge_metrics.items():
        if key == "policy_loss_mode":
            continue
        metrics[f"edge_{key}"] = value

    with torch.no_grad():
        combined = log_paths_pf_tree + log_paths_pf_edge
        if mask is None:
            mask_t = torch.ones_like(combined, dtype=combined.dtype)
        else:
            mask_t = mask.to(dtype=combined.dtype)
        metrics["mean_log_pf_tree"] = float((log_paths_pf_tree * mask_t).sum().item() / mask_t.sum().clamp(min=1.0).item())
        metrics["mean_log_pf_edge"] = float((log_paths_pf_edge * mask_t).sum().item() / mask_t.sum().clamp(min=1.0).item())

    return loss, metrics
