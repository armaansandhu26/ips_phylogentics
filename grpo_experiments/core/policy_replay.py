"""Fixed-trajectory replay for policy importance sampling (pi_new / pi_old)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from grpo_experiments.core.forward_replay import forward_replay_fixed_actions


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
                    cleaned[key] = int(value.item())
                else:
                    cleaned[key] = [int(x) for x in value.tolist()]
            elif isinstance(value, (np.integer,)):
                cleaned[key] = int(value)
            elif isinstance(value, (list, tuple)):
                cleaned[key] = [int(x) for x in value]
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

    @property
    def size(self) -> int:
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
    reward_chunks: list[torch.Tensor] = []
    score_chunks: list[torch.Tensor] = []

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
            all_actions.append(trajectory_actions(traj))
        all_trajectories.extend(trajectories)

        pf_chunks.append(data["log_paths_pf"].detach().cpu())
        reward_chunks.append(data["log_rewards"].detach().cpu())
        score_chunks.append(data["log_scores"].detach().cpu())
        remaining -= n

    buffer = ReplayBuffer(
        actions_set=all_actions,
        trajectories=all_trajectories,
        log_paths_pf_old=torch.cat(pf_chunks, dim=0).to(device),
        log_rewards=torch.cat(reward_chunks, dim=0).to(device),
        log_scores=torch.cat(score_chunks, dim=0).to(device),
        random_spec=random_spec,
    )
    buffer.log_paths_pf_old = reevaluate_log_paths_pf(
        rollout_worker,
        generator,
        buffer,
        chunk_size=chunk_size,
        device=device,
    )
    return buffer


def reevaluate_log_paths_pf(
    rollout_worker,
    generator,
    buffer: ReplayBuffer,
    *,
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    pf_chunks: list[torch.Tensor] = []
    actions = buffer.actions_set
    for start in range(0, len(actions), chunk_size):
        chunk_actions = actions[start : start + chunk_size]
        log_paths_pf, _ = forward_replay_fixed_actions(
            rollout_worker,
            generator,
            chunk_actions,
            random_spec=buffer.random_spec,
            device=device,
        )
        pf_chunks.append(log_paths_pf)
    return torch.cat(pf_chunks, dim=0)
