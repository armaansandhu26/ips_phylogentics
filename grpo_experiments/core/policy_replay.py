"""Fixed-trajectory replay for policy importance sampling (pi_new / pi_old)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from grpo_experiments.core.forward_replay import forward_replay_fixed_actions
from src.gfn.action_tensors import TensorActionBatch, concat_tensor_action_batches


def _buffer_action_source(buffer: ReplayBuffer) -> TensorActionBatch | list[list[dict]]:
    if buffer.action_tensors is not None:
        return buffer.action_tensors
    return buffer.actions_set


def _action_source_size(source: TensorActionBatch | list) -> int:
    if isinstance(source, TensorActionBatch):
        return source.size
    return len(source)


def _clean_scalar(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def clean_action(action: dict) -> dict:
    cleaned: dict[str, Any] = {}
    for key, value in action.items():
        if key == "tree_action":
            cleaned[key] = int(value)
        elif key == "edge_action":
            if value is None:
                cleaned[key] = value
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:
                    cleaned[key] = _clean_scalar(value.item())
                else:
                    cleaned[key] = [_clean_scalar(x) for x in value.tolist()]
            elif isinstance(value, (np.integer, np.floating, int, float)):
                cleaned[key] = _clean_scalar(value)
            elif isinstance(value, (list, tuple)):
                cleaned[key] = [_clean_scalar(x) for x in value]
            else:
                cleaned[key] = value
        else:
            cleaned[key] = value
    return cleaned


def trajectory_actions(trajectory) -> list[dict]:
    return [clean_action(a) for a in trajectory.actions]


@dataclass
class ReplayBuffer:
    actions_set: list[list[dict]]
    trajectories: list
    log_paths_pf_old: torch.Tensor
    log_rewards: torch.Tensor
    log_scores: torch.Tensor
    random_spec: dict | None
    log_paths_pb: torch.Tensor | None = None
    action_tensors: TensorActionBatch | None = None
    log_paths_pf_tree_old: torch.Tensor | None = None
    log_paths_pf_edge_old: torch.Tensor | None = None

    @property
    def size(self) -> int:
        if self.action_tensors is not None:
            return self.action_tensors.size
        return len(self.actions_set)

    @property
    def log_pf_old(self) -> torch.Tensor:
        return self.log_paths_pf_old.sum(dim=-1)


def sample_replay_buffer(
    rollout_worker,
    generator,
    *,
    buffer_size: int,
    chunk_size: int,
    random_spec: dict | None,
    device: str,
) -> ReplayBuffer:
    all_actions: list[list[dict]] = []
    all_trajectories: list = []
    pf_chunks: list[torch.Tensor] = []
    pb_chunks: list[torch.Tensor] = []
    reward_chunks: list[torch.Tensor] = []
    score_chunks: list[torch.Tensor] = []
    action_tensor_chunks: list[TensorActionBatch] = []

    remaining = buffer_size
    while remaining > 0:
        n = min(chunk_size, remaining)
        data, trajectories = rollout_worker.rollout(
            generator,
            n,
            random_spec=random_spec,
            generate_full_trajectories=False,
        )
        for traj in trajectories:
            if data.get("action_tensors") is None:
                all_actions.append(trajectory_actions(traj))
        all_trajectories.extend(trajectories)

        pf_chunks.append(data["log_paths_pf"].detach())
        if "log_paths_pb" in data:
            pb_chunks.append(data["log_paths_pb"].detach())
        reward_chunks.append(data["log_rewards"].detach())
        score_chunks.append(data["log_scores"].detach())
        if data.get("action_tensors") is not None:
            action_tensor_chunks.append(data["action_tensors"].detach())
        remaining -= n

    buffer = ReplayBuffer(
        actions_set=all_actions,
        trajectories=all_trajectories,
        log_paths_pf_old=torch.cat(pf_chunks, dim=0).to(device),
        log_paths_pb=torch.cat(pb_chunks, dim=0).to(device) if pb_chunks else None,
        log_rewards=torch.cat(reward_chunks, dim=0).to(device),
        log_scores=torch.cat(score_chunks, dim=0).to(device),
        random_spec=random_spec,
        action_tensors=concat_tensor_action_batches(action_tensor_chunks),
    )
    with torch.no_grad():
        reeval_out = reevaluate_log_paths_pf(
            rollout_worker,
            generator,
            buffer,
            chunk_size=chunk_size,
            device=device,
            return_split=True,
        )
        if len(reeval_out) == 4:
            buffer.log_paths_pf_old, _, buffer.log_paths_pf_tree_old, buffer.log_paths_pf_edge_old = reeval_out
        else:
            buffer.log_paths_pf_old, _ = reeval_out
    return buffer


def reevaluate_log_paths_pf(
    rollout_worker,
    generator,
    buffer: ReplayBuffer,
    *,
    chunk_size: int,
    device: str,
    return_split: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pf_chunks: list[torch.Tensor] = []
    tree_chunks: list[torch.Tensor] = []
    edge_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []
    actions = _buffer_action_source(buffer)
    total = _action_source_size(actions)
    for start in range(0, total, chunk_size):
        if isinstance(actions, TensorActionBatch):
            chunk_actions = actions.slice(start, start + chunk_size)
        else:
            chunk_actions = actions[start : start + chunk_size]
        replay_out = forward_replay_fixed_actions(
            rollout_worker,
            generator,
            chunk_actions,
            random_spec=buffer.random_spec,
            device=device,
            return_split=return_split,
        )
        if return_split:
            log_paths_pf, paths_entropy, log_tree, log_edge = replay_out
            tree_chunks.append(log_tree)
            edge_chunks.append(log_edge)
        else:
            log_paths_pf, paths_entropy = replay_out
        pf_chunks.append(log_paths_pf)
        entropy_chunks.append(paths_entropy)
    log_paths_pf = torch.cat(pf_chunks, dim=0)
    paths_entropy = torch.cat(entropy_chunks, dim=0)
    if return_split:
        return (
            log_paths_pf,
            paths_entropy,
            torch.cat(tree_chunks, dim=0),
            torch.cat(edge_chunks, dim=0),
        )
    return log_paths_pf, paths_entropy
