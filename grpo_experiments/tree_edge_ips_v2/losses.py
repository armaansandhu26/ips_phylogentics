from __future__ import annotations

import torch


def ppo_clipped_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float = 0.4,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Standard PPO clipped surrogate for one policy component."""

    if log_prob.shape != old_log_prob.shape:
        raise ValueError("log_prob and old_log_prob must have the same shape.")
    if advantages.shape != log_prob.shape:
        if advantages.ndim == 1 and log_prob.ndim == 2 and advantages.shape[0] == log_prob.shape[0]:
            advantages = advantages.unsqueeze(1)
        else:
            advantages = advantages.reshape(log_prob.shape)

    ratio = torch.exp(log_prob - old_log_prob)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)

    if mask is not None:
        mask = mask.to(dtype=objective.dtype, device=objective.device)
        denom = mask.sum().clamp(min=1.0)
        loss = -(objective * mask).sum() / denom
        ratio_mean = (ratio * mask).sum() / denom
        clipped_fraction = (((ratio - clipped_ratio).abs() > 0) * mask.bool()).to(torch.float32).sum() / denom
    else:
        loss = -objective.mean()
        ratio_mean = ratio.mean()
        clipped_fraction = ((ratio - clipped_ratio).abs() > 0).to(torch.float32).mean()

    return loss, {
        "ratio_mean": float(ratio_mean.detach().item()),
        "clipped_fraction": float(clipped_fraction.detach().item()),
        "loss": float(loss.detach().item()),
    }


def split_tree_edge_ppo_loss(
    log_prob_tree: torch.Tensor,
    log_prob_edge: torch.Tensor,
    old_log_prob_tree: torch.Tensor,
    old_log_prob_edge: torch.Tensor,
    advantage_tree: torch.Tensor,
    advantage_edge: torch.Tensor | None = None,
    *,
    clip_eps: float = 0.4,
    entropy_tree: torch.Tensor | None = None,
    entropy_edge: torch.Tensor | None = None,
    entropy_coef: float = 0.01,
    aux_loss: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """v2 split PPO: independent ratios for tree and edge policies."""

    if advantage_edge is None:
        advantage_edge = advantage_tree

    tree_loss, tree_metrics = ppo_clipped_loss(
        log_prob_tree,
        old_log_prob_tree,
        advantage_tree,
        clip_eps=clip_eps,
        mask=mask,
    )
    edge_loss, edge_metrics = ppo_clipped_loss(
        log_prob_edge,
        old_log_prob_edge,
        advantage_edge,
        clip_eps=clip_eps,
        mask=mask,
    )

    entropy_bonus = torch.zeros((), dtype=tree_loss.dtype, device=tree_loss.device)
    if entropy_tree is not None:
        entropy_bonus = entropy_bonus + entropy_tree.mean()
    if entropy_edge is not None:
        entropy_bonus = entropy_bonus + entropy_edge.mean()

    total = tree_loss + edge_loss - float(entropy_coef) * entropy_bonus
    if aux_loss is not None:
        total = total + aux_loss

    metrics = {
        "loss": float(total.detach().item()),
        "tree_policy_loss": tree_metrics["loss"],
        "edge_policy_loss": edge_metrics["loss"],
        "tree_ratio_mean": tree_metrics["ratio_mean"],
        "edge_ratio_mean": edge_metrics["ratio_mean"],
        "tree_clipped_fraction": tree_metrics["clipped_fraction"],
        "edge_clipped_fraction": edge_metrics["clipped_fraction"],
        "entropy_bonus": float(entropy_bonus.detach().item()),
    }
    if aux_loss is not None:
        metrics["aux_loss"] = float(aux_loss.detach().item())
    return total, metrics
