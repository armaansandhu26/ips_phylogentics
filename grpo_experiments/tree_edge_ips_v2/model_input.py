from __future__ import annotations

import torch
import torch.nn.functional as F


def edge_input_from_rep(
    rep: torch.Tensor,
    tree_action: torch.Tensor,
    *,
    num_tree_actions: int,
    detach_rep: bool = True,
) -> torch.Tensor:
    """Build v2 edge-model input: [tree representation ; one_hot(tree_action)]."""

    if rep.ndim != 2:
        raise ValueError(f"rep must be [batch, hidden], got shape {tuple(rep.shape)}.")
    tree_action = tree_action.to(device=rep.device, dtype=torch.long).reshape(-1)
    if tree_action.shape[0] != rep.shape[0]:
        raise ValueError("tree_action batch size must match rep batch size.")
    if detach_rep:
        rep = rep.detach()
    action_one_hot = F.one_hot(tree_action, num_classes=num_tree_actions).to(dtype=rep.dtype)
    return torch.cat([rep, action_one_hot], dim=-1)


def selected_pair_reps(trees_reps: torch.Tensor, tree_pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left/right subtree reps for sampled merge pairs."""

    if trees_reps.ndim != 3:
        raise ValueError("trees_reps must be [batch, num_trees, hidden].")
    if tree_pairs.ndim != 2 or tree_pairs.shape[-1] != 2:
        raise ValueError("tree_pairs must be [batch, 2].")
    batch_idx = torch.arange(trees_reps.shape[0], device=trees_reps.device)
    left = trees_reps[batch_idx, tree_pairs[:, 0].long()]
    right = trees_reps[batch_idx, tree_pairs[:, 1].long()]
    return left, right
