"""Two hierarchical policies: direction first, then masked step count."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

NUM_DIRECTIONS = 2
NEG_INF = -1e9


def _trunk(input_dim: int, hidden_size: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(num_layers):
        layers.extend((nn.Linear(width, hidden_size), nn.Tanh()))
        width = hidden_size
    return nn.Sequential(*layers)


def _masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if torch.any(~mask.bool().any(dim=-1)):
        raise ValueError("every sampled policy row must have at least one valid action")
    return torch.where(mask.bool(), logits, torch.full_like(logits, NEG_INF))


class DirectionPolicy(nn.Module):
    """Model 1: observation -> direction and a state representation."""

    def __init__(self, obs_dim: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.trunk = _trunk(obs_dim, hidden_size, num_layers)
        self.head = nn.Linear(hidden_size, NUM_DIRECTIONS)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def dist_with_rep(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        """Return the direction distribution and its already-computed encoding."""
        representation = self.encode(obs)
        distribution = Categorical(
            logits=_masked(self.head(representation), mask)
        )
        return distribution, representation

    def dist(self, obs: torch.Tensor, mask: torch.Tensor) -> Categorical:
        distribution, _ = self.dist_with_rep(obs, mask)
        return distribution

    def sample_with_rep(
        self, obs: np.ndarray, mask: np.ndarray
    ) -> Tuple[int, float, np.ndarray]:
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            dist, rep = self.dist_with_rep(obs_t, mask_t)
            action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            rep.squeeze(0).cpu().numpy(),
        )


class StepPolicy(nn.Module):
    """Model 2: (model-1 representation, chosen direction) -> step count."""

    def __init__(
        self, rep_dim: int, max_step: int, hidden_size: int, num_layers: int
    ) -> None:
        super().__init__()
        self.max_step = max_step
        self.net = nn.Sequential(
            _trunk(rep_dim + NUM_DIRECTIONS, hidden_size, num_layers),
            nn.Linear(hidden_size, max_step),
        )

    def dist(
        self, rep: torch.Tensor, direction: torch.Tensor, mask: torch.Tensor
    ) -> Categorical:
        direction_oh = F.one_hot(direction, NUM_DIRECTIONS).float()
        logits = self.net(torch.cat((rep, direction_oh), dim=-1))
        return Categorical(logits=_masked(logits, mask))

    def sample(
        self, rep: np.ndarray, direction: int, mask: np.ndarray
    ) -> Tuple[int, float]:
        device = next(self.parameters()).device
        rep_t = torch.as_tensor(rep, dtype=torch.float32, device=device).unsqueeze(0)
        direction_t = torch.tensor([direction], dtype=torch.long, device=device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            dist = self.dist(rep_t, direction_t, mask_t)
            action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())
