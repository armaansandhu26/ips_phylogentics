"""Masked categorical policy over merge (pair) actions."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

NEG_INF = -1e9


def build_trunk(input_dim: int, hidden_size: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.Tanh())
        in_dim = hidden_size
    return nn.Sequential(*layers)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits of invalid actions to -inf (mask: 1=valid)."""
    return torch.where(mask.bool(), logits, torch.full_like(logits, NEG_INF))


class MergePolicyNet(nn.Module):
    """pi(pair | forest observation), with per-state action masking."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.trunk = build_trunk(obs_dim, hidden_size, num_layers)
        self.head = nn.Linear(hidden_size, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))

    def dist(self, obs: torch.Tensor, mask: torch.Tensor) -> Categorical:
        logits = masked_logits(self.forward(obs), mask)
        return Categorical(logits=logits)

    def sample(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[int, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            dist = self.dist(obs_t, mask_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def log_prob_and_entropy(
        self,
        obs_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(obs_batch, mask_batch)
        return dist.log_prob(action_batch), dist.entropy()
