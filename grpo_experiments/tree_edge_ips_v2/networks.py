from __future__ import annotations

import torch
import torch.nn as nn


class TreePolicyNet(nn.Module):
    """Spec MLP tree policy: obs -> rep -> tree logits (+ optional aux heads)."""

    def __init__(
        self,
        obs_dim: int,
        num_tree_actions: int,
        *,
        hidden_dim: int = 128,
        aux_head_sizes: dict[str, int] | None = None,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.logits_tree = nn.Linear(hidden_dim, num_tree_actions)
        self.aux_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, size) for name, size in (aux_head_sizes or {}).items()}
        )

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        rep = self.trunk(obs)
        return {
            "rep": rep,
            "logits_tree": self.logits_tree(rep),
            "aux_logits": {name: head(rep) for name, head in self.aux_heads.items()},
        }


class EdgePolicyNet(nn.Module):
    """Spec MLP edge policy over [rep ; one_hot(tree_action)]."""

    def __init__(
        self,
        input_dim: int,
        num_edge_actions: int,
        *,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_edge_actions),
        )

    def forward(self, edge_input: torch.Tensor) -> torch.Tensor:
        return self.net(edge_input)
