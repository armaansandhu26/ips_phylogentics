"""Learned reverse policy for Hyper-Grid trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class HyperGridReverseBatch:
    contexts: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    episode_indices: torch.Tensor
    num_episodes: int


class HyperGridReversePolicy(nn.Module):
    """Terminal-conditioned reverse policy over decrement-x / decrement-y."""

    CONTEXT_DIM = 7

    def __init__(self, *, hidden_size: int = 128, num_layers: int = 2, H: int = 64):
        super().__init__()
        self.H = int(H)
        self.max_actions = 2
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
        logits = self.head(self.trunk(contexts))
        logits = torch.where(masks.bool(), logits, torch.full_like(logits, -1e9))
        return Categorical(logits=logits)


def reverse_context(
    *,
    coords: tuple[int, int],
    terminal_coords: tuple[int, int],
    step_index: int,
    path_len: int,
    log_reward: float,
    H: int,
) -> tuple[float, ...]:
    denom = float(max(H - 1, 1))
    x, y = coords
    tx, ty = terminal_coords
    return (
        x / denom,
        y / denom,
        tx / denom,
        ty / denom,
        step_index / max(path_len, 1),
        (path_len - step_index) / max(path_len, 1),
        log_reward / 3.0,
    )


def reverse_action_mask(coords: tuple[int, int]) -> tuple[bool, bool]:
    x, y = coords
    return (x > 0, y > 0)


def forward_action_paths_from_batch(
    actions: torch.Tensor,
    mask: torch.Tensor,
    *,
    terminate_action: int,
) -> list[tuple[int, ...]]:
    action_rows = actions.detach().cpu().numpy()
    mask_rows = mask.detach().cpu().numpy()
    paths: list[tuple[int, ...]] = []
    for action_row, mask_row in zip(action_rows, mask_rows):
        path: list[int] = []
        for action, active in zip(action_row, mask_row):
            if not active:
                break
            action = int(action)
            if action == terminate_action:
                break
            path.append(action)
        paths.append(tuple(path))
    return paths


def build_reverse_batch(
    action_paths: list[tuple[int, ...]],
    *,
    terminal_coords: torch.Tensor,
    terminal_log_rewards: torch.Tensor,
    H: int,
    device: torch.device | str,
) -> HyperGridReverseBatch:
    if not action_paths:
        raise ValueError("action_paths must be non-empty")
    device = torch.device(device)
    contexts: list[tuple[float, ...]] = []
    masks: list[tuple[bool, bool]] = []
    actions: list[int] = []
    episode_indices: list[int] = []

    terminals = terminal_coords.detach().cpu().numpy()
    log_rewards = terminal_log_rewards.detach().cpu().tolist()
    for episode_index, (path, terminal_row, log_reward) in enumerate(
        zip(action_paths, terminals, log_rewards)
    ):
        tx, ty = int(terminal_row[0]), int(terminal_row[1])
        x, y = tx, ty
        path_len = len(path)
        for step_index, forward_action in enumerate(reversed(path)):
            if forward_action == 0:
                reverse_action = 0
                state = (x, y)
                x -= 1
            elif forward_action == 1:
                reverse_action = 1
                state = (x, y)
                y -= 1
            else:
                raise ValueError(f"unsupported forward action {forward_action}")
            valid = reverse_action_mask(state)
            if not valid[reverse_action]:
                raise ValueError(f"invalid reverse transition at state={state}")
            contexts.append(
                reverse_context(
                    coords=state,
                    terminal_coords=(tx, ty),
                    step_index=step_index,
                    path_len=path_len,
                    log_reward=float(log_reward),
                    H=H,
                )
            )
            masks.append(valid)
            actions.append(reverse_action)
            episode_indices.append(episode_index)

    if not contexts:
        zero = torch.zeros(0, HyperGridReversePolicy.CONTEXT_DIM, device=device)
        return HyperGridReverseBatch(
            contexts=zero,
            masks=torch.zeros(0, 2, dtype=torch.bool, device=device),
            actions=torch.zeros(0, dtype=torch.long, device=device),
            episode_indices=torch.zeros(0, dtype=torch.long, device=device),
            num_episodes=len(action_paths),
        )

    return HyperGridReverseBatch(
        contexts=torch.tensor(contexts, dtype=torch.float32, device=device),
        masks=torch.tensor(masks, dtype=torch.bool, device=device),
        actions=torch.tensor(actions, dtype=torch.long, device=device),
        episode_indices=torch.tensor(episode_indices, dtype=torch.long, device=device),
        num_episodes=len(action_paths),
    )


def path_log_probabilities_tensor(
    policy: HyperGridReversePolicy, batch: HyperGridReverseBatch
) -> tuple[torch.Tensor, Categorical]:
    if batch.contexts.numel() == 0:
        zeros = torch.zeros(batch.num_episodes, device=policy.head.weight.device)
        return zeros, policy.dist(batch.contexts, batch.masks)
    distribution = policy.dist(batch.contexts, batch.masks)
    edge_log_probabilities = distribution.log_prob(batch.actions)
    path_log_probabilities = torch.zeros(
        batch.num_episodes, dtype=torch.float32, device=batch.contexts.device
    )
    path_log_probabilities.scatter_add_(0, batch.episode_indices, edge_log_probabilities)
    return path_log_probabilities, distribution


@torch.inference_mode()
def path_log_probabilities(
    policy: HyperGridReversePolicy,
    action_paths: list[tuple[int, ...]],
    *,
    terminal_coords: torch.Tensor,
    terminal_log_rewards: torch.Tensor,
    H: int,
) -> torch.Tensor:
    batch = build_reverse_batch(
        action_paths,
        terminal_coords=terminal_coords,
        terminal_log_rewards=terminal_log_rewards,
        H=H,
        device=next(policy.parameters()).device,
    )
    path_log_probabilities, _ = path_log_probabilities_tensor(policy, batch)
    return path_log_probabilities


def update_reverse_policy(
    policy: HyperGridReversePolicy,
    optimizer: torch.optim.Optimizer,
    batch: HyperGridReverseBatch,
    *,
    train_epochs: int,
    grad_clip_norm: float,
) -> dict[str, float]:
    if batch.contexts.numel() == 0:
        return {
            "reverse_loss": 0.0,
            "reverse_path_probability_mean": 0.0,
            "reverse_edge_accuracy": 0.0,
            "reverse_edge_entropy": 0.0,
            "reverse_grad_norm": 0.0,
            "reverse_param_norm": 0.0,
        }

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
        path_log_probabilities, distribution = path_log_probabilities_tensor(policy, batch)
        predictions = distribution.logits.argmax(dim=-1)
        parameter_norm = sum(p.detach().norm().item() ** 2 for p in policy.parameters()) ** 0.5
        return {
            "reverse_loss": float(-path_log_probabilities.mean().item()),
            "reverse_path_probability_mean": float(path_log_probabilities.exp().mean().item()),
            "reverse_edge_accuracy": float((predictions == batch.actions).float().mean().item()),
            "reverse_edge_entropy": float(distribution.entropy().mean().item()),
            "reverse_grad_norm": grad_norm_total / train_epochs,
            "reverse_param_norm": float(parameter_norm),
        }
