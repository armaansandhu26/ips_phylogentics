"""Per-step MLP reverse proposal for phylogenetic tree construction.

This mirrors ``compound_action_rl/dag_toy_dataset/learned_reverse_ips.py`` but
operates on tree-merge actions instead of a tabular catalog.  For ``N`` taxa
each reverse step is a masked categorical over valid merge indices at the
pre-merge forest size, conditioned on terminal topology and log-score.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical


def num_merge_actions(num_subtrees: int) -> int:
    if num_subtrees < 2:
        return 0
    return num_subtrees * (num_subtrees - 1) // 2


def max_merge_actions(num_taxa: int) -> int:
    return num_merge_actions(num_taxa)


def _topology_features(terminal_id: str) -> tuple[float, float]:
    digest = hashlib.sha256(terminal_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    angle = (value % 10_000) / 10_000.0 * (2.0 * math.pi)
    return math.sin(angle), math.cos(angle)


def reverse_context(
    *,
    num_taxa: int,
    step_index: int,
    num_trees_before: int,
    merge_pair: tuple[int, int],
    terminal_id: str,
    terminal_log_score: float,
    log_score_shift: float,
) -> tuple[float, ...]:
    """Continuous features for one reverse merge decision."""
    if num_taxa < 2:
        raise ValueError("num_taxa must be >= 2")
    num_steps = num_taxa - 1
    if step_index < 0 or step_index >= num_steps:
        raise ValueError("step_index out of range")
    if num_trees_before != num_taxa - step_index:
        raise ValueError("num_trees_before inconsistent with step_index")
    num_trees_after = num_trees_before - 1
    topology_sin, topology_cos = _topology_features(terminal_id)
    shifted_score = terminal_log_score - log_score_shift
    return (
        num_trees_before / num_taxa,
        num_trees_after / num_taxa,
        step_index / num_steps,
        (num_steps - step_index) / num_steps,
        merge_pair[0] / num_taxa,
        merge_pair[1] / num_taxa,
        shifted_score / 1000.0,
        topology_sin,
        topology_cos,
    )


def reverse_action_mask(num_trees_before: int, *, max_actions: int) -> tuple[bool, ...]:
    valid = num_merge_actions(num_trees_before)
    if valid > max_actions:
        raise ValueError("num_trees_before exceeds configured max_actions")
    return tuple(index < valid for index in range(max_actions))


@dataclass(frozen=True)
class PhyloReverseBatch:
    contexts: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    episode_indices: torch.Tensor
    num_episodes: int


@dataclass(frozen=True)
class PhyloLearnedReverseConfig:
    hidden_size: int = 128
    num_layers: int = 2

    def validate(self) -> None:
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("reverse hidden_size and num_layers must be >= 1")


class PhyloLearnedReversePolicy(nn.Module):
    """Terminal-conditioned categorical over valid tree-merge indices."""

    CONTEXT_DIM = 9

    def __init__(
        self,
        num_taxa: int,
        *,
        hidden_size: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_taxa < 2:
            raise ValueError("num_taxa must be >= 2")
        self.num_taxa = int(num_taxa)
        self.max_actions = max_merge_actions(self.num_taxa)
        layers: list[nn.Module] = []
        width = self.CONTEXT_DIM
        for _ in range(num_layers):
            layers.extend((nn.Linear(width, hidden_size), nn.Tanh()))
            width = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, self.max_actions)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def dist(self, contexts: torch.Tensor, masks: torch.Tensor) -> Categorical:
        if contexts.ndim != 2 or contexts.shape[1] != self.CONTEXT_DIM:
            raise ValueError(
                f"reverse contexts must have shape (edges, {self.CONTEXT_DIM})"
            )
        if masks.ndim != 2 or masks.shape != (contexts.shape[0], self.max_actions):
            raise ValueError("reverse masks must have shape (edges, max_actions)")
        if torch.any(~masks.bool().any(dim=-1)):
            raise ValueError("every reverse state must have a valid parent merge")
        logits = self.head(self.trunk(contexts))
        logits = torch.where(masks.bool(), logits, torch.full_like(logits, -1e9))
        return Categorical(logits=logits)


def build_reverse_batch(
    env,
    action_paths: list[tuple[int, ...]],
    *,
    terminal_ids: list[str],
    terminal_log_scores: list[float],
    device: torch.device | str,
) -> PhyloReverseBatch:
    """Expand forward merge paths into per-edge reverse training rows."""
    if not action_paths:
        raise ValueError("action_paths must be non-empty")
    if len(terminal_ids) != len(action_paths) or len(terminal_log_scores) != len(
        action_paths
    ):
        raise ValueError("terminal metadata must align with action_paths")

    num_taxa = len(env.sequences)
    expected_steps = num_taxa - 1
    max_actions = max_merge_actions(num_taxa)
    log_score_shift = float(getattr(env, "log_score_shift", 0.0))

    contexts: list[tuple[float, ...]] = []
    masks: list[tuple[bool, ...]] = []
    actions: list[int] = []
    episode_indices: list[int] = []

    for episode_index, (action_path, terminal_id, terminal_log_score) in enumerate(
        zip(action_paths, terminal_ids, terminal_log_scores)
    ):
        if len(action_path) != expected_steps:
            raise ValueError(
                f"expected {expected_steps} merge actions for {num_taxa} taxa, "
                f"got {len(action_path)}"
            )
        for step_index, tree_action in enumerate(action_path):
            num_trees_before = num_taxa - step_index
            pair = env.tree_pairs_dict[num_trees_before][tree_action]
            contexts.append(
                reverse_context(
                    num_taxa=num_taxa,
                    step_index=step_index,
                    num_trees_before=num_trees_before,
                    merge_pair=pair,
                    terminal_id=terminal_id,
                    terminal_log_score=float(terminal_log_score),
                    log_score_shift=log_score_shift,
                )
            )
            masks.append(
                reverse_action_mask(
                    num_trees_before, max_actions=max_actions
                )
            )
            if tree_action < 0 or tree_action >= num_merge_actions(num_trees_before):
                raise ValueError(
                    f"invalid tree_action {tree_action} at num_trees={num_trees_before}"
                )
            actions.append(int(tree_action))
            episode_indices.append(episode_index)

    device = torch.device(device)
    return PhyloReverseBatch(
        contexts=torch.tensor(contexts, dtype=torch.float32, device=device),
        masks=torch.tensor(masks, dtype=torch.bool, device=device),
        actions=torch.tensor(actions, dtype=torch.long, device=device),
        episode_indices=torch.tensor(episode_indices, dtype=torch.long, device=device),
        num_episodes=len(action_paths),
    )


def path_log_probabilities_tensor(
    policy: PhyloLearnedReversePolicy, batch: PhyloReverseBatch
) -> tuple[torch.Tensor, Categorical]:
    distribution = policy.dist(batch.contexts, batch.masks)
    edge_log_probabilities = distribution.log_prob(batch.actions)
    path_log_probabilities = torch.zeros(
        batch.num_episodes, dtype=torch.float32, device=batch.contexts.device
    )
    path_log_probabilities.scatter_add_(
        0, batch.episode_indices, edge_log_probabilities
    )
    return path_log_probabilities, distribution


@torch.inference_mode()
def path_log_probabilities(
    policy: PhyloLearnedReversePolicy,
    env,
    action_paths: list[tuple[int, ...]],
    *,
    terminal_ids: list[str],
    terminal_log_scores: list[float],
) -> torch.Tensor:
    batch = build_reverse_batch(
        env,
        action_paths,
        terminal_ids=terminal_ids,
        terminal_log_scores=terminal_log_scores,
        device=next(policy.parameters()).device,
    )
    path_log_probabilities, _ = path_log_probabilities_tensor(policy, batch)
    return path_log_probabilities


def update_mlp_reverse_policy(
    policy: PhyloLearnedReversePolicy,
    optimizer: torch.optim.Optimizer,
    batch: PhyloReverseBatch,
    *,
    train_epochs: int,
    grad_clip_norm: float,
) -> dict[str, float]:
    grad_norm_total = 0.0
    for _ in range(train_epochs):
        optimizer.zero_grad(set_to_none=True)
        path_log_probabilities, _ = path_log_probabilities_tensor(policy, batch)
        loss = -path_log_probabilities.mean()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
        grad_norm_total += float(grad_norm.item())

    with torch.inference_mode():
        path_log_probabilities, distribution = path_log_probabilities_tensor(
            policy, batch
        )
        predictions = distribution.logits.argmax(dim=-1)
        parameter_norm = sum(
            parameter.detach().norm().item() ** 2
            for parameter in policy.parameters()
        ) ** 0.5
        return {
            "reverse_loss": float(-path_log_probabilities.mean().item()),
            "reverse_path_probability_mean": float(
                path_log_probabilities.exp().mean().item()
            ),
            "reverse_edge_accuracy": float(
                (predictions == batch.actions).float().mean().item()
            ),
            "reverse_edge_entropy": float(distribution.entropy().mean().item()),
            "reverse_grad_norm": grad_norm_total / train_epochs,
            "reverse_param_norm": float(parameter_norm),
            "reverse_normalization_error": 0.0,
        }
