"""Neural policies for the hierarchical color-trajectory agent."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from grid_environment_2 import NUM_COLOR_ACTIONS, NUM_MOVE_ACTIONS
from hierarchical import model2_input_from_rep


def build_trunk(input_dim: int, hidden_size: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.Tanh())
        in_dim = hidden_size
    return nn.Sequential(*layers)


def build_mlp(input_dim: int, output_dim: int, hidden_size: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = [build_trunk(input_dim, hidden_size, num_layers)]
    layers.append(nn.Linear(hidden_size, output_dim))
    return nn.Sequential(*layers)


class MovePolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.trunk = build_trunk(obs_dim, hidden_size, num_layers)
        self.head = nn.Linear(hidden_size, NUM_MOVE_ACTIONS)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(obs))

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))

    def _select_from_obs(
        self, obs: np.ndarray, *, greedy: bool
    ) -> Tuple[int, float, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            rep = self.encode(obs_t)
            dist = Categorical(logits=self.head(rep))
            if greedy:
                action = int(dist.probs.argmax(dim=-1).item())
            else:
                action = int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor(action)).item())
            rep_np = rep.squeeze(0).cpu().numpy()
        return action, log_prob, rep_np

    def select_action(
        self, obs: np.ndarray, rng: np.random.Generator | None = None, *, greedy: bool = False
    ) -> Tuple[int, float]:
        del rng
        action, log_prob, _rep = self._select_from_obs(obs, greedy=greedy)
        return action, log_prob

    def sample_with_rep(
        self, obs: np.ndarray, rng: np.random.Generator
    ) -> Tuple[int, float, np.ndarray]:
        del rng
        return self._select_from_obs(obs, greedy=False)

    def sample(self, obs: np.ndarray, rng: np.random.Generator) -> Tuple[int, float]:
        action, log_prob, _rep = self.sample_with_rep(obs, rng)
        return action, log_prob


class ColorPolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.net = build_mlp(input_dim, NUM_COLOR_ACTIONS, hidden_size, num_layers)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.net(model_input)

    def dist(self, model_input: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(model_input))

    def _model_input(self, move_rep: np.ndarray, move_action: int) -> torch.Tensor:
        return torch.as_tensor(
            model2_input_from_rep(move_rep, move_action),
            dtype=torch.float32,
        ).unsqueeze(0)

    def select_action(
        self,
        move_rep: np.ndarray,
        move_action: int,
        rng: np.random.Generator | None = None,
        *,
        greedy: bool = False,
    ) -> Tuple[int, float]:
        del rng
        model_input = self._model_input(move_rep, move_action)
        with torch.no_grad():
            dist = self.dist(model_input)
            if greedy:
                action = int(dist.probs.argmax(dim=-1).item())
            else:
                action = int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor(action)).item())
        return action, log_prob

    def sample(
        self, move_rep: np.ndarray, move_action: int, rng: np.random.Generator
    ) -> Tuple[int, float]:
        return self.select_action(move_rep, move_action, rng, greedy=False)

    def log_prob_and_entropy(
        self,
        move_rep_batch: torch.Tensor,
        move_batch: torch.Tensor,
        color_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        move_oh = torch.nn.functional.one_hot(move_batch, NUM_MOVE_ACTIONS).float()
        model_input = torch.cat([move_rep_batch, move_oh], dim=-1)
        dist = self.dist(model_input)
        log_prob = dist.log_prob(color_batch)
        entropy = dist.entropy()
        return log_prob, entropy
