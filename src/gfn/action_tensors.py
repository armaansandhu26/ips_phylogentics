from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


def _edge_tensor(values, device: torch.device | str | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(values, device=device)
    if torch.is_floating_point(tensor):
        return tensor.to(dtype=torch.float32)
    return tensor.to(dtype=torch.long)


def _python_scalar(value):
    if isinstance(value, (bool, int)):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


@dataclass
class TensorActionBatch:
    """Per-step action tensors for a fixed batch of trajectories."""

    tree_actions: tuple[torch.Tensor, ...]
    edge_actions: tuple[torch.Tensor | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.edge_actions is not None and len(self.edge_actions) != len(self.tree_actions):
            raise ValueError("tree_actions and edge_actions must have the same number of steps.")
        if not self.tree_actions:
            return
        batch_size = self.tree_actions[0].shape[0]
        for step, actions in enumerate(self.tree_actions):
            if actions.shape[0] != batch_size:
                raise ValueError(f"tree_actions[{step}] batch size does not match.")
        if self.edge_actions is not None:
            for step, actions in enumerate(self.edge_actions):
                if actions is not None and actions.shape[0] != batch_size:
                    raise ValueError(f"edge_actions[{step}] batch size does not match.")

    @property
    def size(self) -> int:
        if not self.tree_actions:
            return 0
        return int(self.tree_actions[0].shape[0])

    @property
    def num_steps(self) -> int:
        return len(self.tree_actions)

    def __len__(self) -> int:
        return self.size

    def detach(self) -> "TensorActionBatch":
        edge_actions = None
        if self.edge_actions is not None:
            edge_actions = tuple(None if x is None else x.detach() for x in self.edge_actions)
        return TensorActionBatch(
            tree_actions=tuple(x.detach() for x in self.tree_actions),
            edge_actions=edge_actions,
        )

    def to(self, device: torch.device | str) -> "TensorActionBatch":
        edge_actions = None
        if self.edge_actions is not None:
            edge_actions = tuple(None if x is None else x.to(device) for x in self.edge_actions)
        return TensorActionBatch(
            tree_actions=tuple(x.to(device) for x in self.tree_actions),
            edge_actions=edge_actions,
        )

    def slice(self, start: int, end: int) -> "TensorActionBatch":
        edge_actions = None
        if self.edge_actions is not None:
            edge_actions = tuple(None if x is None else x[start:end] for x in self.edge_actions)
        return TensorActionBatch(
            tree_actions=tuple(x[start:end] for x in self.tree_actions),
            edge_actions=edge_actions,
        )

    def step_tree_actions(self, step: int, device: torch.device | str) -> torch.Tensor:
        return self.tree_actions[step].to(device=device, dtype=torch.long)

    def step_edge_actions(self, step: int, device: torch.device | str) -> torch.Tensor | None:
        if self.edge_actions is None:
            return None
        actions = self.edge_actions[step]
        if actions is None:
            return None
        return actions.to(device=device)

    def tree_action_keys(self) -> list[tuple[int, ...]]:
        """One CPU sync: merge-step tree action indices per trajectory."""
        if not self.tree_actions:
            return []
        stacked = torch.stack(self.tree_actions, dim=1).detach().cpu().tolist()
        return [tuple(int(x) for x in row) for row in stacked]

    def actions_for_index(self, idx: int) -> list[dict]:
        """Action dicts for one trajectory (includes edge actions when present)."""
        actions: list[dict] = []
        edge_steps = self.edge_actions
        for step, tree_actions in enumerate(self.tree_actions):
            action = {"tree_action": int(tree_actions[idx].item())}
            if edge_steps is not None and edge_steps[step] is not None:
                edge = edge_steps[step][idx]
                if getattr(edge, "ndim", 0) == 0:
                    action["edge_action"] = _python_scalar(edge.item())
                else:
                    action["edge_action"] = [_python_scalar(x) for x in edge.detach().cpu().tolist()]
            actions.append(action)
        return actions

    def to_actions_set(self) -> list[list[dict]]:
        actions_set = [[] for _ in range(self.size)]
        edge_steps = self.edge_actions
        for step, tree_actions in enumerate(self.tree_actions):
            tree_cpu = tree_actions.detach().cpu().tolist()
            edge_cpu = None
            if edge_steps is not None and edge_steps[step] is not None:
                edge_cpu = edge_steps[step].detach().cpu().tolist()
            for idx, tree_action in enumerate(tree_cpu):
                action = {"tree_action": int(tree_action)}
                if edge_cpu is not None:
                    edge_action = edge_cpu[idx]
                    if isinstance(edge_action, list):
                        action["edge_action"] = [_python_scalar(x) for x in edge_action]
                    else:
                        action["edge_action"] = _python_scalar(edge_action)
                actions_set[idx].append(action)
        return actions_set

    @classmethod
    def from_actions_set(
        cls,
        actions_set: list[list[dict]],
        *,
        device: torch.device | str | None = None,
    ) -> "TensorActionBatch":
        if not actions_set:
            return cls(tree_actions=(), edge_actions=None)
        num_steps = len(actions_set[0])
        if any(len(actions) != num_steps for actions in actions_set):
            raise ValueError("All trajectories must have the same number of actions.")

        tree_steps: list[torch.Tensor] = []
        edge_steps: list[torch.Tensor | None] = []
        saw_edge_actions = False
        for step in range(num_steps):
            step_actions = [trajectory_actions[step] for trajectory_actions in actions_set]
            tree_steps.append(
                torch.as_tensor(
                    [action["tree_action"] for action in step_actions],
                    device=device,
                    dtype=torch.long,
                )
            )
            if all("edge_action" in action and action["edge_action"] is not None for action in step_actions):
                saw_edge_actions = True
                edge_steps.append(_edge_tensor([action["edge_action"] for action in step_actions], device=device))
            else:
                edge_steps.append(None)

        return cls(
            tree_actions=tuple(tree_steps),
            edge_actions=tuple(edge_steps) if saw_edge_actions else None,
        )


def concat_tensor_action_batches(batches: Iterable[TensorActionBatch | None]) -> TensorActionBatch | None:
    batches = [batch for batch in batches if batch is not None]
    if not batches:
        return None
    num_steps = batches[0].num_steps
    if any(batch.num_steps != num_steps for batch in batches):
        raise ValueError("Cannot concatenate action batches with different numbers of steps.")

    tree_steps = tuple(torch.cat([batch.tree_actions[step] for batch in batches], dim=0) for step in range(num_steps))
    has_edges = any(batch.edge_actions is not None for batch in batches)
    edge_steps = None
    if has_edges:
        merged_edges: list[torch.Tensor | None] = []
        for step in range(num_steps):
            step_edges = []
            for batch in batches:
                if batch.edge_actions is None or batch.edge_actions[step] is None:
                    raise ValueError("Cannot concatenate mixed edge/no-edge action batches.")
                step_edges.append(batch.edge_actions[step])
            merged_edges.append(torch.cat(step_edges, dim=0))
        edge_steps = tuple(merged_edges)
    return TensorActionBatch(tree_actions=tree_steps, edge_actions=edge_steps)
