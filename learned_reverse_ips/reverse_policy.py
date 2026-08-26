from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class TabularTerminalReversePolicy(nn.Module):
    """Exactly normalized q(structural history | terminal topology)."""

    def __init__(
        self,
        trajectories: Sequence[tuple[int, ...]],
        terminal_ids: Sequence[str],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if not trajectories or len(trajectories) != len(terminal_ids):
            raise ValueError("trajectory catalog and terminal IDs must be non-empty and aligned")
        if len(set(trajectories)) != len(trajectories):
            raise ValueError("trajectory catalog contains duplicates")

        self.trajectories = tuple(tuple(int(action) for action in path) for path in trajectories)
        self.terminal_ids = tuple(str(item) for item in terminal_ids)
        self.trajectory_to_index = {
            trajectory: index for index, trajectory in enumerate(self.trajectories)
        }

        grouped: dict[str, list[int]] = defaultdict(list)
        for index, terminal_id in enumerate(self.terminal_ids):
            grouped[terminal_id].append(index)
        self.group_ids = tuple(sorted(grouped))
        self.group_indices = tuple(
            torch.tensor(grouped[group_id], dtype=torch.long, device=device)
            for group_id in self.group_ids
        )
        group_lookup = {group_id: group for group, group_id in enumerate(self.group_ids)}
        self.register_buffer(
            "trajectory_groups",
            torch.tensor(
                [group_lookup[terminal_id] for terminal_id in self.terminal_ids],
                dtype=torch.long,
                device=device,
            ),
        )
        self.logits = nn.Parameter(torch.zeros(len(self.trajectories), device=device))

    def catalog_indices(self, action_paths: Sequence[tuple[int, ...]]) -> Tensor:
        try:
            indices = [self.trajectory_to_index[tuple(path)] for path in action_paths]
        except KeyError as exc:
            raise ValueError(f"rollout produced an unknown tree-action trajectory: {exc.args[0]}") from exc
        return torch.tensor(indices, dtype=torch.long, device=self.logits.device)

    def all_log_probabilities(self) -> Tensor:
        output = torch.empty_like(self.logits)
        for indices in self.group_indices:
            output[indices] = self.logits[indices] - torch.logsumexp(
                self.logits[indices], dim=0
            )
        return output

    def log_prob(self, catalog_indices: Tensor) -> Tensor:
        return self.all_log_probabilities()[catalog_indices]

    def entropy_for_indices(self, catalog_indices: Tensor) -> Tensor:
        log_q = self.all_log_probabilities()
        entropies = torch.empty(len(self.group_indices), device=self.logits.device)
        for group, indices in enumerate(self.group_indices):
            probabilities = log_q[indices].exp()
            entropies[group] = -(probabilities * log_q[indices]).sum()
        return entropies[self.trajectory_groups[catalog_indices]]

    @torch.no_grad()
    def normalization_error(self) -> float:
        log_q = self.all_log_probabilities()
        errors = [
            abs(float(log_q[indices].exp().sum().item()) - 1.0)
            for indices in self.group_indices
        ]
        return max(errors)


def _edge_action(step: int, num_steps: int) -> int | list[int]:
    return 0 if step == num_steps - 1 else [0, 0]


def enumerate_tree_action_catalog(
    env,
) -> tuple[list[tuple[int, ...]], list[str]]:
    """Enumerate every merge-action path and its structural terminal ID."""
    num_taxa = len(env.sequences)
    if num_taxa != 5:
        raise ValueError(
            f"this exact tabular experiment requires 5 taxa, found {num_taxa}"
        )
    action_ranges = [
        range(num_trees * (num_trees - 1) // 2)
        for num_trees in range(num_taxa, 1, -1)
    ]
    trajectories: list[tuple[int, ...]] = []
    terminal_ids: list[str] = []
    for action_path in itertools.product(*action_ranges):
        actions = [
            {
                "tree_action": int(tree_action),
                "edge_action": _edge_action(step, len(action_path)),
            }
            for step, tree_action in enumerate(action_path)
        ]
        trajectory = env.actions_to_trajectory(actions)
        tree = trajectory.current_state.subtrees[0]
        trajectories.append(tuple(int(action) for action in action_path))
        terminal_ids.append(str(tree.ete_node.get_topology_id()))
    return trajectories, terminal_ids


def rollout_tree_action_paths(batch: dict[str, Any]) -> list[tuple[int, ...]]:
    action_tensors = batch.get("action_tensors")
    if action_tensors is None:
        raise ValueError("rollout batch is missing action_tensors")
    tree_actions = torch.stack(action_tensors.tree_actions, dim=1)
    return [tuple(int(value) for value in row) for row in tree_actions.cpu().tolist()]


def trajectory_indices_from_paths(
    action_paths: Sequence[tuple[int, ...]],
    *,
    device: torch.device | str,
) -> Tensor:
    lookup: dict[tuple[int, ...], int] = {}
    indices: list[int] = []
    for path in action_paths:
        if path not in lookup:
            lookup[path] = len(lookup)
        indices.append(lookup[path])
    return torch.tensor(indices, dtype=torch.long, device=device)


def update_reverse_policy(
    policy: TabularTerminalReversePolicy,
    optimizer: torch.optim.Optimizer,
    trajectory_indices: Tensor,
    *,
    train_epochs: int,
    grad_clip_norm: float,
) -> dict[str, float]:
    """MLE update performed only after the forward-policy update."""
    grad_norm_total = 0.0
    for _ in range(train_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = -policy.log_prob(trajectory_indices).mean()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
        grad_norm_total += float(grad_norm.item())

    with torch.inference_mode():
        log_probabilities = policy.log_prob(trajectory_indices)
        entropy = policy.entropy_for_indices(trajectory_indices)
        return {
            "reverse_loss": float(-log_probabilities.mean().item()),
            "reverse_path_probability_mean": float(log_probabilities.exp().mean().item()),
            "reverse_path_entropy": float(entropy.mean().item()),
            "reverse_grad_norm": grad_norm_total / train_epochs,
            "reverse_param_norm": float(policy.logits.norm().item()),
            "reverse_normalization_error": policy.normalization_error(),
        }
