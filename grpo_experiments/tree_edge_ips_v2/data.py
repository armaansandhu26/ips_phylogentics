from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import torch


OutcomeKey = Hashable


@dataclass
class StepRecord:
    """One environment step for split tree/edge PPO replay."""

    obs: Any
    tree_action: int
    edge_action: int | float | tuple[int | float, ...]
    log_prob_tree: float
    log_prob_edge: float
    aux_targets: dict[str, int] = field(default_factory=dict)
    advantage_tree: float | None = None
    advantage_edge: float | None = None

    @property
    def log_prob_joint(self) -> float:
        return float(self.log_prob_tree) + float(self.log_prob_edge)


@dataclass
class Episode:
    """Terminal trajectory plus cached quantities used by IPS-GRPO v2."""

    steps: list[StepRecord]
    return_: float
    tree_path: tuple[int, ...]
    edge_sequence: tuple[Any, ...]
    trajectory_index: int | None = None
    outcome: OutcomeKey | None = None

    def __post_init__(self) -> None:
        if self.outcome is None:
            self.outcome = (self.tree_path, self.edge_sequence)

    @property
    def log_prob_joint(self) -> float:
        return sum(step.log_prob_joint for step in self.steps)

    @property
    def log_prob_tree(self) -> float:
        return sum(float(step.log_prob_tree) for step in self.steps)

    @property
    def log_prob_edge(self) -> float:
        return sum(float(step.log_prob_edge) for step in self.steps)

    def set_trajectory_advantage(self, advantage: float) -> None:
        for step in self.steps:
            step.advantage_tree = float(advantage)
            step.advantage_edge = float(advantage)


def episode_tensors(episodes: list[Episode], device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    """Convert per-episode terminal quantities to tensors."""

    return {
        "returns": torch.as_tensor([ep.return_ for ep in episodes], dtype=torch.float32, device=device),
        "log_prob_joint": torch.as_tensor([ep.log_prob_joint for ep in episodes], dtype=torch.float32, device=device),
        "log_prob_tree": torch.as_tensor([ep.log_prob_tree for ep in episodes], dtype=torch.float32, device=device),
        "log_prob_edge": torch.as_tensor([ep.log_prob_edge for ep in episodes], dtype=torch.float32, device=device),
    }
