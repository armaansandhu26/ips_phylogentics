"""Trajectory Balance (TB) training for the Hyper-Grid GFlowNet."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def uniform_backward_log_probs(
    action_paths: list[tuple[int, ...]],
    terminal_coords: torch.Tensor,
    *,
    dim: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Uniform backward policy: decrement any positive coordinate with equal prob."""
    device = torch.device(device)
    terminals = terminal_coords.detach().cpu().numpy()
    log_pb = torch.zeros(len(action_paths), dtype=torch.float32, device=device)
    for index, (path, terminal_row) in enumerate(zip(action_paths, terminals)):
        coords = [int(terminal_row[axis]) for axis in range(dim)]
        total = 0.0
        for forward_action in reversed(path):
            n_back = sum(1 for value in coords if value > 0)
            if n_back <= 0:
                break
            total += -math.log(n_back)
            coords[int(forward_action)] -= 1
        log_pb[index] = total
    return log_pb


class HyperGridTBTrainer:
    """TB loss: (log Z + log P_F(τ) - log R(x) - log P_B(τ))^2 with uniform backward."""

    def __init__(
        self,
        params: list[nn.Parameter],
        *,
        lr: float = 1e-3,
        max_grad_norm: float = 1.0,
    ):
        self.log_z = nn.Parameter(torch.zeros(1))
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer = torch.optim.Adam([*params, self.log_z], lr=lr)

    def state_dict(self) -> dict:
        return {
            "log_z": self.log_z.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, payload: dict) -> None:
        if "log_z" in payload:
            self.log_z.data.copy_(payload["log_z"].to(self.log_z.device))
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])

    def update(
        self,
        log_paths_pf: torch.Tensor,
        log_rewards: torch.Tensor,
        *,
        action_paths: list[tuple[int, ...]],
        terminal_coords: torch.Tensor,
        dim: int,
        extra_metrics: dict | None = None,
    ) -> dict[str, float]:
        log_pf = log_paths_pf.sum(dim=-1)
        log_pb = uniform_backward_log_probs(
            action_paths,
            terminal_coords,
            dim=dim,
            device=log_paths_pf.device,
        )
        log_z = self.log_z.reshape(())
        forward = log_z + log_pf
        backward = log_rewards + log_pb
        residual = forward - backward
        loss = (residual**2).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]["params"],
            self.max_grad_norm,
        )
        self.optimizer.step()

        info = {
            "loss": float(loss.item()),
            "tb_loss": float(loss.item()),
            "log_Z": float(log_z.item()),
            "mean_log_pf": float(log_pf.mean().item()),
            "mean_log_pb": float(log_pb.mean().item()),
            "mean_log_reward": float(log_rewards.mean().item()),
            "mean_tb_residual": float(residual.mean().item()),
            "std_tb_residual": float(residual.std(unbiased=False).item()),
            "grad_norm": float(grad_norm.item()),
            "mean_advantage": 0.0,
        }
        if extra_metrics:
            info.update(extra_metrics)
        return info
