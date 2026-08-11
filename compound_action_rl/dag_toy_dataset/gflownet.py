"""Minimal trajectory-balance GFlowNet for the direction/step DAG.

This is the small-DAG analogue of the core PhyloGFN objective.  For each
sampled trajectory ``tau`` ending at ``x`` it minimizes

    (log Z + log P_F(tau) - log R(x) - log P_B(tau | x)) ** 2

``P_F`` is the same hierarchical direction/step policy used by Count-IPS.
``P_B`` is fixed and uniform over the valid incoming compound edges of each
state.  It is locally normalized, so its probabilities sum to one over all
reverse trajectories from any terminal to the source.  Consequently the
terminal marginal targeted by trajectory balance is proportional to reward.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from config import TrainConfig
from count_ips import CountIPSTrainer, Episode, _pad_episode_values
from dag_env import uniform_backward_log_probability


def trajectory_balance_loss(
    log_z: torch.Tensor,
    log_pf: torch.Tensor,
    log_pb: torch.Tensor,
    log_rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean trajectory-balance loss and the per-trajectory residual."""
    if log_pf.ndim != 1:
        raise ValueError("log_pf must have shape (batch,)")
    if log_pb.shape != log_pf.shape or log_rewards.shape != log_pf.shape:
        raise ValueError("log_pf, log_pb, and log_rewards must have equal shapes")
    if log_z.numel() != 1:
        raise ValueError("log_z must be scalar")
    residual = log_z.reshape(()) + log_pf - log_rewards - log_pb
    return residual.square().mean(), residual


class TrajectoryBalanceGFlowNet(CountIPSTrainer):
    """A scalar-Z trajectory-balance GFlowNet with a fixed backward policy."""

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        z_lr: float = 1e-2,
        initial_log_z: float = 0.0,
    ) -> None:
        super().__init__(config, device=device)
        if self.config.entropy_coef != 0:
            raise ValueError(
                "set entropy_coef=0: trajectory balance already defines the target"
            )
        if z_lr <= 0:
            raise ValueError("z_lr must be > 0")
        self.z_lr = float(z_lr)
        self.log_z = nn.Parameter(
            torch.tensor(float(initial_log_z), device=self.device)
        )
        model_params = self._model_parameters()
        self.optimizer = torch.optim.Adam(
            [
                {"params": model_params, "lr": self.config.lr},
                {"params": [self.log_z], "lr": self.z_lr},
            ]
        )

    def _model_parameters(self) -> list[nn.Parameter]:
        return list(self.direction_policy.parameters()) + list(
            self.step_policy.parameters()
        )

    def _trajectory_log_pf_and_entropy(
        self, episodes: list[Episode]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not episodes or any(not episode.steps for episode in episodes):
            raise ValueError("episodes must contain at least one non-empty trajectory")

        steps = [step for episode in episodes for step in episode.steps]
        obs = torch.as_tensor(
            np.stack([step.obs for step in steps]),
            dtype=torch.float32,
            device=self.device,
        )
        direction_masks = torch.as_tensor(
            np.stack([step.direction_mask for step in steps]),
            dtype=torch.bool,
            device=self.device,
        )
        step_masks = torch.as_tensor(
            np.stack([step.step_mask for step in steps]),
            dtype=torch.bool,
            device=self.device,
        )
        directions = torch.tensor(
            [step.direction for step in steps],
            dtype=torch.long,
            device=self.device,
        )
        step_indices = torch.tensor(
            [step.step_index for step in steps],
            dtype=torch.long,
            device=self.device,
        )

        direction_dist, representation = self.direction_policy.dist_with_rep(
            obs, direction_masks
        )
        step_rep = (
            representation.detach()
            if self.config.detach_step_rep
            else representation
        )
        step_dist = self.step_policy.dist(step_rep, directions, step_masks)
        joint_log_prob = direction_dist.log_prob(directions) + step_dist.log_prob(
            step_indices
        )
        joint_entropy = direction_dist.entropy() + step_dist.entropy()

        lengths = [len(episode.steps) for episode in episodes]
        max_length = max(lengths)
        padded_log_prob = _pad_episode_values(joint_log_prob, lengths, max_length)
        padded_entropy = _pad_episode_values(joint_entropy, lengths, max_length)
        mask = torch.arange(max_length, device=self.device).unsqueeze(0) < torch.tensor(
            lengths, device=self.device
        ).unsqueeze(1)
        log_pf = (padded_log_prob * mask).sum(dim=1)
        mean_entropy = (
            (padded_entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        ).mean()
        return log_pf, mean_entropy

    def _loss(
        self, episodes: list[Episode]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        log_pf, mean_entropy = self._trajectory_log_pf_and_entropy(episodes)
        log_pb = torch.tensor(
            [
                uniform_backward_log_probability(
                    episode.trajectory, max_step=self.config.max_step
                )
                for episode in episodes
            ],
            dtype=torch.float32,
            device=self.device,
        )
        log_rewards = torch.tensor(
            [np.log(episode.reward) for episode in episodes],
            dtype=torch.float32,
            device=self.device,
        )
        loss, residual = trajectory_balance_loss(
            self.log_z, log_pf, log_pb, log_rewards
        )
        with torch.no_grad():
            stats = {
                "loss": float(loss.item()),
                "tb_residual_mean": float(residual.mean().item()),
                "tb_residual_std": float(residual.std(unbiased=False).item()),
                "tb_residual_abs_mean": float(residual.abs().mean().item()),
                "forward_log_probability_mean": float(log_pf.mean().item()),
                "backward_log_probability_mean": float(log_pb.mean().item()),
                "log_reward_mean": float(log_rewards.mean().item()),
                "entropy": float(mean_entropy.item()),
            }
        return loss, stats

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        totals: dict[str, float] = {}
        model_params = self._model_parameters()
        for _ in range(self.config.train_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss, stats = self._loss(episodes)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model_params, self.config.grad_clip_norm
            )
            z_grad = (
                float(self.log_z.grad.item())
                if self.log_z.grad is not None
                else 0.0
            )
            self.optimizer.step()
            stats.update(
                {
                    "grad_norm": float(grad_norm.item()),
                    "z_grad": z_grad,
                    "param_norm": float(
                        sum(
                            parameter.detach().norm().item() ** 2
                            for parameter in model_params
                        )
                        ** 0.5
                    ),
                    "log_z": float(self.log_z.detach().item()),
                }
            )
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
        return {
            key: value / self.config.train_epochs for key, value in totals.items()
        }

    def train(
        self,
        *,
        eval_every: int | None = None,
        eval_episodes: int = 10_000,
        checkpoint_every: int | None = None,
        checkpoint_dir: Path | str | None = None,
    ) -> list[dict[str, Any]]:
        if checkpoint_every is not None:
            if checkpoint_every < 1:
                raise ValueError("checkpoint_every must be >= 1")
            if checkpoint_dir is None:
                raise ValueError(
                    "checkpoint_dir is required when checkpoint_every is set"
                )
            checkpoint_dir = Path(checkpoint_dir)

        history: list[dict[str, Any]] = []
        for update_step in range(1, self.config.num_updates + 1):
            episodes: list[Episode] = []
            for _ in range(self.config.num_groups):
                episodes.extend(
                    self.rollout_batch(self.config.group_size, explore=True)
                )
            stats = self.update(episodes)
            batch_counts = Counter(episode.terminal for episode in episodes)
            self._seen_terminals.update(batch_counts)
            batch_size = len(episodes)
            row: dict[str, Any] = {
                "step": update_step,
                "mean_reward": float(np.mean([ep.reward for ep in episodes])),
                "mean_length": float(np.mean([len(ep.steps) for ep in episodes])),
                "unique_terminals": len(batch_counts),
                "global_unique_outcomes": float(len(self._seen_terminals)),
                "batch_outcome_counts": {
                    state.signature: int(batch_counts[state])
                    for state in self.terminals
                },
                "batch_outcome_probs": {
                    state.signature: float(batch_counts[state] / batch_size)
                    for state in self.terminals
                },
                **stats,
            }
            if eval_every and (update_step == 1 or update_step % eval_every == 0):
                row.update(self.evaluate(eval_episodes))
            history.append(row)
            if checkpoint_every and update_step % checkpoint_every == 0:
                assert checkpoint_dir is not None
                checkpoint_path = self.save(
                    checkpoint_dir / f"checkpoint_update_{update_step:06d}.pt",
                    update_step=update_step,
                )
                print(f"Checkpoint: {checkpoint_path}")
            if update_step == 1 or update_step % self.config.log_every == 0:
                print(
                    f"update {update_step:4d}  loss={row['loss']:.4f}  "
                    f"residual={row['tb_residual_abs_mean']:.3f}  "
                    f"logZ={row['log_z']:.3f}  "
                    f"reward={row['mean_reward']:.3f}  "
                    f"outcomes={row['unique_terminals']}  "
                    f"global_outcomes={row['global_unique_outcomes']:.0f}"
                    + (
                        f"  eval_TV={row['tv_reward_target']:.3f}"
                        if "tv_reward_target" in row
                        else ""
                    )
                )
        return history

    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "direction_policy": self.direction_policy.state_dict(),
                "step_policy": self.step_policy.state_dict(),
                "log_z": self.log_z.detach(),
                "optimizer": self.optimizer.state_dict(),
                "update_step": update_step,
                "algorithm": {
                    "name": "trajectory_balance_gflownet",
                    "objective": "(logZ + logPF - logR - logPB)^2",
                    "backward_policy": "uniform_over_valid_parents",
                    "z_lr": self.z_lr,
                },
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "TrajectoryBalanceGFlowNet":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        trainer = cls(
            payload["config"],
            device=device,
            z_lr=payload["algorithm"]["z_lr"],
            initial_log_z=float(payload["log_z"].item()),
        )
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        if "optimizer" in payload:
            trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer
