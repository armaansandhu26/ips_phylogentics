"""Neural policies with position auxiliary head on model 1."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from grid_paths import NUM_COLOR_ACTIONS, NUM_MOVE_ACTIONS
from model_input import model2_input_from_rep


def build_trunk(input_dim: int, hidden_size: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.Tanh())
        in_dim = hidden_size
    return nn.Sequential(*layers)


def build_mlp(input_dim: int, output_dim: int, hidden_size: int, num_layers: int) -> nn.Sequential:
    return nn.Sequential(
        build_trunk(input_dim, hidden_size, num_layers),
        nn.Linear(hidden_size, output_dim),
    )


class PathPolicyNet(nn.Module):
    """Model 1: π_move(a | obs). Exposes encode(obs) as state representation."""

    def __init__(
        self,
        obs_dim: int,
        hidden_size: int,
        num_layers: int,
        *,
        grid_size: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.trunk = build_trunk(obs_dim, hidden_size, num_layers)
        self.head = nn.Linear(hidden_size, NUM_MOVE_ACTIONS)
        self.row_head = nn.Linear(hidden_size, grid_size)
        self.col_head = nn.Linear(hidden_size, grid_size)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(obs))

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))

    def position_aux_loss(
        self,
        rep: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
    ) -> torch.Tensor:
        row_logits = self.row_head(rep)
        col_logits = self.col_head(rep)
        return F.cross_entropy(row_logits, rows) + F.cross_entropy(col_logits, cols)

    def sample_with_rep(self, obs: np.ndarray) -> Tuple[int, float, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            rep = self.encode(obs_t)
            dist = self.dist(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), rep.squeeze(0).cpu().numpy()


class ColorPolicyNet(nn.Module):
    """Model 2: π_color(c | state_rep, move). Does NOT see grid obs."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.net = build_mlp(input_dim, NUM_COLOR_ACTIONS, hidden_size, num_layers)

    def dist(self, model_input: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(model_input))

    def sample(self, state_rep: np.ndarray, move_action: int) -> Tuple[int, float]:
        model_input = torch.as_tensor(
            model2_input_from_rep(state_rep, move_action), dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            dist = self.dist(model_input)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def log_prob_and_entropy(
        self,
        state_rep_batch: torch.Tensor,
        move_batch: torch.Tensor,
        color_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        move_oh = torch.nn.functional.one_hot(move_batch, NUM_MOVE_ACTIONS).float()
        model_input = torch.cat([state_rep_batch, move_oh], dim=-1)
        dist = self.dist(model_input)
        return dist.log_prob(color_batch), dist.entropy()
