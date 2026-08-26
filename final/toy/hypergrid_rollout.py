"""Batched rollouts for Hyper-Grid GRPO / IPS training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from final.toy.hypergrid_env import HyperGridDataset
from final.toy.hypergrid_policy import HyperGridPolicy


@dataclass
class HyperGridRolloutBatch:
    log_paths_pf: torch.Tensor
    log_rewards: torch.Tensor
    log_scores: torch.Tensor
    terminal_coords: torch.Tensor
    outcome_ids: list[str]
    actions: torch.Tensor
    mask: torch.Tensor

    def as_training_batch(self) -> dict:
        return {
            "log_paths_pf": self.log_paths_pf,
            "log_rewards": self.log_rewards,
            "log_scores": self.log_scores,
            "mask": self.mask,
        }


def rollout_batch(
    policy: HyperGridPolicy,
    dataset: HyperGridDataset,
    *,
    batch_size: int,
    device: str | torch.device,
) -> HyperGridRolloutBatch:
    device = torch.device(device)
    policy = policy.to(device)
    spec = dataset.spec
    dim = spec.D
    terminate = dataset.terminate_action

    coords = torch.zeros(batch_size, dim, dtype=torch.long, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    terminal_coords = torch.zeros(batch_size, dim, dtype=torch.long, device=device)

    log_probs_steps: list[torch.Tensor] = []
    action_steps: list[torch.Tensor] = []
    max_steps = dim * (spec.H - 1) + 1

    for _ in range(max_steps):
        if not active.any():
            break
        logits = policy.forward_logits(coords)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)

        terminate_now = active & (actions == terminate)
        terminal_coords[terminate_now] = coords[terminate_now]

        for axis in range(dim):
            inc = active & (actions == axis)
            coords[inc, axis] += 1

        step_log_prob = torch.zeros(batch_size, device=device, dtype=log_prob.dtype)
        step_actions = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        step_log_prob[active] = log_prob[active]
        step_actions[active] = actions[active]

        active = active & ~terminate_now
        log_probs_steps.append(step_log_prob)
        action_steps.append(step_actions)

    if active.any():
        terminal_coords[active] = coords[active]

    log_paths_pf = torch.stack(log_probs_steps, dim=1)
    actions = torch.stack(action_steps, dim=1)
    mask = actions >= 0

    terminal_np = terminal_coords.detach().cpu().numpy()
    rewards_np = dataset.reward_at(terminal_np)
    rewards = torch.tensor(rewards_np, device=device, dtype=torch.float32)
    log_rewards = torch.log(rewards.clamp_min(1e-12))
    outcome_ids = [
        dataset.terminal_outcome_id(int(row[0]), int(row[1]) if dim > 1 else 0)
        for row in terminal_np
    ]

    return HyperGridRolloutBatch(
        log_paths_pf=log_paths_pf,
        log_rewards=log_rewards,
        log_scores=log_rewards,
        terminal_coords=terminal_coords,
        outcome_ids=outcome_ids,
        actions=actions,
        mask=mask,
    )


@torch.no_grad()
def sample_terminals(
    policy: HyperGridPolicy,
    dataset: HyperGridDataset,
    *,
    num_samples: int,
    batch_size: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    coords_all: list[torch.Tensor] = []
    rewards_all: list[torch.Tensor] = []
    remaining = num_samples
    while remaining > 0:
        current = min(batch_size, remaining)
        batch = rollout_batch(policy, dataset, batch_size=current, device=device)
        coords_all.append(batch.terminal_coords.cpu())
        rewards_all.append(torch.exp(batch.log_rewards).cpu())
        remaining -= current
    return torch.cat(coords_all, dim=0), torch.cat(rewards_all, dim=0)
