"""
GRPO on 4×4 grid — v2 with split per-model losses and trajectory-level IPS credit.

1,280 unique trajectories = 20 paths × 2^6 color sequences.
Model 1: obs → move + state_rep (+ position aux head)
Model 2: (state_rep, move) → color  [state_rep detached from color loss]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from config import TrainConfig
from grid_paths import GridEnv, make_env, moves_to_path_index, num_trajectories, trajectory_lookup
from model_input import model2_input_dim
from networks import ColorPolicyNet, PathPolicyNet

RED = 0
GREEN = 1


@dataclass
class StepRecord:
    obs: np.ndarray
    move_action: int
    color_action: int
    log_prob_path: float
    log_prob_color: float
    row: int
    col: int
    advantage_path: float = 0.0
    advantage_color: float = 0.0

    @property
    def log_prob_joint(self) -> float:
        return self.log_prob_path + self.log_prob_color


@dataclass
class Episode:
    steps: list[StepRecord] = field(default_factory=list)
    moves: tuple[int, ...] = ()
    colors: tuple[int, ...] = ()
    path_index: int = -1
    trajectory_index: int = -1
    return_: float = 0.0

    @property
    def outcome(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self.moves, self.colors)

    @property
    def log_prob_joint(self) -> float:
        return sum(s.log_prob_joint for s in self.steps)


def normalize(values: np.ndarray, eps: float) -> np.ndarray:
    mean = values.mean()
    std = values.std()
    if std < eps:
        return values - mean
    return (values - mean) / (std + eps)


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(w * w))
    if denom <= 0.0:
        return 0.0
    return float((w.sum() ** 2) / denom)


class GRPOTrainer:
    def __init__(self, config: TrainConfig | None = None, *, device: str = "cpu") -> None:
        self.config = config or TrainConfig()
        self.device = torch.device(device)
        torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)

        self.env = make_env(**self.config.profile_kwargs())
        self.obs_dim = self.env.obs_dim
        self._num_trajectories = num_trajectories(self.env)
        self._moves_lookup = moves_to_path_index(self.env)
        self._traj_lookup = trajectory_lookup(self.env)

        self.path_policy = PathPolicyNet(
            self.obs_dim,
            self.config.hidden_size,
            self.config.num_layers,
            grid_size=self.config.grid_size,
        ).to(self.device)
        color_in = model2_input_dim(self.config.hidden_size)
        self.color_policy = ColorPolicyNet(
            color_in, self.config.hidden_size, self.config.num_layers
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self._optimizer_params(), lr=self.config.lr)

    def _optimizer_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.config.trainable in ("both", "path_only"):
            params.extend(self.path_policy.parameters())
        if self.config.trainable in ("both", "color_only"):
            params.extend(self.color_policy.parameters())
        return params

    def _step_color_counterfactual(self, row: int, col: int, color_action: int) -> float:
        max_r = self.env._max_reward
        red_val = float(self.env.red_dist[row, col] / max_r)
        green_val = float(self.env.green_dist[row, col] / max_r)
        chosen = red_val if color_action == RED else green_val
        return chosen - 0.5 * (red_val + green_val)

    def _sample_color(self, state_rep: np.ndarray, move_action: int) -> tuple[int, float]:
        if self.config.trainable == "path_only":
            c = int(self.rng.integers(0, 2))
            return c, float(-np.log(2.0))
        return self.color_policy.sample(state_rep, move_action)

    def _select_step(self, obs: np.ndarray) -> tuple[int, int, float, float]:
        if self.config.trainable == "color_only":
            with torch.no_grad():
                move_action, log_p_path, state_rep = self.path_policy.sample_with_rep(obs)
        else:
            move_action, log_p_path, state_rep = self.path_policy.sample_with_rep(obs)
        color_action, log_p_color = self._sample_color(state_rep, move_action)
        return move_action, color_action, log_p_path, log_p_color

    def rollout_episode(self, env: GridEnv | None = None) -> Episode:
        env = env or self.env
        episode = Episode()
        obs, _, _ = env.reset()
        done = False
        moves: list[int] = []
        colors: list[int] = []

        while not done:
            move_action, color_action, log_p_path, log_p_color = self._select_step(obs)
            next_obs, reward, done, state = env.step(move_action, color_action)
            row, col = int(state[0]), int(state[1])
            episode.steps.append(
                StepRecord(
                    obs=obs.copy(),
                    move_action=move_action,
                    color_action=color_action,
                    log_prob_path=log_p_path,
                    log_prob_color=log_p_color,
                    row=row,
                    col=col,
                )
            )
            moves.append(move_action)
            colors.append(color_action)
            obs = next_obs

        episode.moves = tuple(moves)
        episode.colors = tuple(colors)
        episode.path_index = self._moves_lookup.get(episode.moves, -1)
        episode.trajectory_index = self._traj_lookup.get(episode.outcome, -1)
        episode.return_ = float(reward)
        return episode

    def _trajectory_weights(self, episodes: list[Episode]) -> np.ndarray:
        """Unit weights — overridden by IPSGRPOTrainer."""
        return np.ones(len(episodes), dtype=np.float64)

    def _group_advantages(self, episodes: list[Episode]) -> float:
        weights = self._trajectory_weights(episodes)
        snips = weights * len(weights) / max(weights.sum(), 1e-12)

        path_scaled = np.array([ep.return_ for ep in episodes], dtype=np.float64) * snips
        path_advs = normalize(path_scaled, self.config.advantage_eps)

        if self.config.color_credit == "counterfactual":
            color_raw: list[float] = []
            for ep, w in zip(episodes, snips):
                for step in ep.steps:
                    cf = self._step_color_counterfactual(step.row, step.col, step.color_action)
                    color_raw.append(cf * w)
            color_advs = normalize(np.asarray(color_raw, dtype=np.float64), self.config.advantage_eps)
            idx = 0
            for ep, path_adv in zip(episodes, path_advs):
                for step in ep.steps:
                    step.advantage_path = float(path_adv)
                    step.advantage_color = float(color_advs[idx])
                    idx += 1
        else:
            # "trajectory": both models share the IPS-weighted trajectory advantage
            # (matching fixed point), but keep separate losses/ratios.
            for ep, path_adv in zip(episodes, path_advs):
                for step in ep.steps:
                    step.advantage_path = float(path_adv)
                    step.advantage_color = float(path_adv)

        return effective_sample_size(weights)

    def _ppo_surrogate(
        self,
        log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: torch.Tensor,
    ) -> torch.Tensor:
        ratio = torch.exp(log_prob - old_log_prob)
        clip = self.config.clip_ratio
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
        return -torch.min(surr1, surr2).mean()

    def _policy_loss(self, steps: list[StepRecord]) -> tuple[torch.Tensor, dict[str, float]]:
        obs = torch.as_tensor(np.stack([s.obs for s in steps]), dtype=torch.float32, device=self.device)
        moves = torch.as_tensor([s.move_action for s in steps], dtype=torch.long, device=self.device)
        colors = torch.as_tensor([s.color_action for s in steps], dtype=torch.long, device=self.device)
        rows = torch.as_tensor([s.row for s in steps], dtype=torch.long, device=self.device)
        cols = torch.as_tensor([s.col for s in steps], dtype=torch.long, device=self.device)

        state_rep = self.path_policy.encode(obs)
        state_rep_color = state_rep.detach() if self.config.detach_color_rep else state_rep

        path_dist = self.path_policy.dist(obs)
        log_p_path = path_dist.log_prob(moves)
        log_p_color, color_ent = self.color_policy.log_prob_and_entropy(state_rep_color, moves, colors)

        old_log_p_path = torch.as_tensor(
            [s.log_prob_path for s in steps], dtype=torch.float32, device=self.device
        )
        old_log_p_color = torch.as_tensor(
            [s.log_prob_color for s in steps], dtype=torch.float32, device=self.device
        )
        adv_path = torch.as_tensor([s.advantage_path for s in steps], device=self.device)
        adv_color = torch.as_tensor([s.advantage_color for s in steps], device=self.device)

        path_loss = self._ppo_surrogate(log_p_path, old_log_p_path, adv_path)
        color_loss = self._ppo_surrogate(log_p_color, old_log_p_color, adv_color)
        policy_loss = path_loss + color_loss

        ent_terms: list[torch.Tensor] = []
        trainable = self.config.trainable
        if trainable in ("both", "path_only"):
            ent_terms.append(path_dist.entropy().mean())
        if trainable in ("both", "color_only"):
            ent_terms.append(color_ent.mean())
        entropy = sum(ent_terms) if ent_terms else torch.tensor(0.0, device=self.device)

        aux_loss = torch.tensor(0.0, device=self.device)
        if self.config.trainable in ("both", "path_only") and self.config.aux_pos_coef > 0:
            aux_loss = self.path_policy.position_aux_loss(state_rep, rows, cols)

        loss = (
            policy_loss
            - self.config.entropy_coef * entropy
            + self.config.aux_pos_coef * aux_loss
        )

        ratio_path = torch.exp(log_p_path - old_log_p_path)
        ratio_color = torch.exp(log_p_color - old_log_p_color)

        return loss, {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "path_loss": float(path_loss.item()),
            "color_loss": float(color_loss.item()),
            "aux_loss": float(aux_loss.item()),
            "entropy": float(entropy.item()) if ent_terms else 0.0,
            "ratio_path_mean": float(ratio_path.mean().item()),
            "ratio_color_mean": float(ratio_color.mean().item()),
        }

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        steps = [step for ep in episodes for step in ep.steps]
        if not steps:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "path_loss": 0.0,
                "color_loss": 0.0,
                "aux_loss": 0.0,
                "entropy": 0.0,
                "ratio_path_mean": 1.0,
                "ratio_color_mean": 1.0,
            }

        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "path_loss": 0.0,
            "color_loss": 0.0,
            "aux_loss": 0.0,
            "entropy": 0.0,
            "ratio_path_mean": 0.0,
            "ratio_color_mean": 0.0,
        }
        params = self._optimizer_params()
        for _ in range(self.config.train_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss, stats = self._policy_loss(steps)
            if params:
                loss.backward()
                if self.config.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(params, self.config.grad_clip_norm)
                self.optimizer.step()
            for key in totals:
                totals[key] += stats[key]
        n = float(self.config.train_epochs)
        return {key: value / n for key, value in totals.items()}

    def train(self, num_updates: int | None = None, log_every: int | None = None) -> list[dict[str, Any]]:
        num_updates = num_updates if num_updates is not None else self.config.num_updates
        log_every = log_every if log_every is not None else self.config.log_every
        history: list[dict[str, Any]] = []

        for step in range(1, num_updates + 1):
            all_episodes: list[Episode] = []
            mean_ess = 0.0
            for _ in range(self.config.num_groups):
                group = [self.rollout_episode() for _ in range(self.config.group_size)]
                mean_ess += self._group_advantages(group)
                all_episodes.extend(group)
            mean_ess /= self.config.num_groups

            opt_stats = self.update(all_episodes)
            returns = [ep.return_ for ep in all_episodes]
            uniq = len({ep.trajectory_index for ep in all_episodes if ep.trajectory_index >= 0})
            row = {
                "step": step,
                "mean_return": float(np.mean(returns)),
                "max_return": float(np.max(returns)),
                "unique_traj_in_batch": uniq,
                "mean_ess": float(mean_ess),
                **opt_stats,
            }
            history.append(row)

            if step == 1 or step % log_every == 0:
                print(
                    f"update {step:4d}  return={row['mean_return']:.3f}  "
                    f"max={row['max_return']:.3f}  uniq_traj={uniq}/{self._num_trajectories}  "
                    f"ESS={row['mean_ess']:.1f}  "
                    f"loss={row['loss']:.4f}  entropy={row['entropy']:.3f}"
                )

        return history

    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent_type": "grpo_v2",
                "path_policy": self.path_policy.state_dict(),
                "color_policy": self.color_policy.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "update_step": update_step,
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: str = "cpu",
        for_training: bool = False,
    ) -> "GRPOTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        trainer = cls(payload["config"], device=device)
        trainer.path_policy.load_state_dict(payload["path_policy"])
        trainer.color_policy.load_state_dict(payload["color_policy"])
        trainer._loaded_update_step = int(payload.get("update_step", 0))
        if for_training:
            trainer.path_policy.train()
            trainer.color_policy.train()
        else:
            trainer.path_policy.eval()
            trainer.color_policy.eval()
        return trainer

    @property
    def loaded_update_step(self) -> int:
        return getattr(self, "_loaded_update_step", 0)
