"""MLP policy for the Hyper-Grid environment."""

from __future__ import annotations

import torch
import torch.nn as nn


class HyperGridPolicy(nn.Module):
    """State-conditioned categorical policy over increment / terminate actions."""

    def __init__(
        self,
        *,
        dim: int = 2,
        num_actions: int = 3,
        hidden_size: int = 256,
        num_layers: int = 2,
        H: int = 4096,
    ):
        super().__init__()
        self.dim = dim
        self.num_actions = num_actions
        self.H = H
        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
            in_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, num_actions)

    def normalized_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Map integer grid coords to signed normalized features in [-1, 1]."""
        denom = max(self.H - 1, 1)
        return 2.0 * coords.to(dtype=torch.float32) / float(denom) - 1.0

    def action_mask(self, coords: torch.Tensor) -> torch.Tensor:
        """True where action is legal."""
        batch = coords.shape[0]
        mask = torch.ones(batch, self.num_actions, dtype=torch.bool, device=coords.device)
        for axis in range(self.dim):
            mask[:, axis] = coords[:, axis] < (self.H - 1)
        return mask

    def forward_logits(self, coords: torch.Tensor) -> torch.Tensor:
        features = self.normalized_coords(coords)
        logits = self.head(self.backbone(features))
        logits = logits.masked_fill(~self.action_mask(coords), -1.0e9)
        return logits

    def dist(self, coords: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.forward_logits(coords))

    def log_prob_actions(self, coords: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.dist(coords).log_prob(actions)

    def entropy(self, coords: torch.Tensor) -> torch.Tensor:
        return self.dist(coords).entropy()
