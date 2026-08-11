"""Group Relative Policy Optimization for the hierarchical color grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from grid_environment_2 import GridEnv
from hierarchical import HierarchicalAgent, HierarchicalStepInfo, HierarchicalAction, model2_input_dim
from policy import ColorPolicyNet, MovePolicyNet


@dataclass
class StepRecord:
    obs: np.ndarray
    move_action: int
    color_action: int
    log_prob_model1: float
    log_prob_model2: float
    advantage: float = 0.0

    @property
    def log_prob_joint(self) -> float:
        return self.log_prob_model1 + self.log_prob_model2


@dataclass
class Episode:
    steps: list[StepRecord] = field(default_factory=list)
    return_: float = 0.0
    outcome: tuple[tuple[int, ...], ...] | None = None


class GRPOAgent(HierarchicalAgent):
    def __init__(
        self,
        obs_dim: int,
        *,
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        clip_ratio: float = 0.2,
        hidden_size: int = 128,
        num_layers: int = 2,
        group_size: int = 16,
        num_groups: int = 4,
        train_epochs: int = 2,
        advantage_eps: float = 1e-8,
        grad_clip_norm: float = 1.0,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.move_net = MovePolicyNet(obs_dim, hidden_size, num_layers).to(self.device)
        color_input_dim = model2_input_dim(hidden_size)
        self.color_net = ColorPolicyNet(color_input_dim, hidden_size, num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.move_net.parameters()) + list(self.color_net.parameters()),
            lr=lr,
        )

        self.entropy_coef = entropy_coef
        self.clip_ratio = clip_ratio
        self.group_size = group_size
        self.num_groups = num_groups
        self.train_epochs = train_epochs
        self.advantage_eps = advantage_eps
        self.grad_clip_norm = grad_clip_norm

        super().__init__(self.move_net, self.color_net, rng=self.rng)
        self._obs_dim = obs_dim
        self._color_input_dim = color_input_dim
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._seed = seed

    def _select_actions(
        self, obs: np.ndarray, *, greedy: bool = False
    ) -> tuple[int, int, float, float]:
        if greedy:
            move_action, log_p1, move_rep = self.move_net._select_from_obs(obs, greedy=True)
        else:
            move_action, log_p1, move_rep = self.move_net.sample_with_rep(obs, self.rng)
        color_action, log_p2 = self.color_net.select_action(
            move_rep, move_action, greedy=greedy
        )
        return move_action, color_action, log_p1, log_p2

    def act(self, obs: np.ndarray) -> tuple[int, int, HierarchicalStepInfo]:
        move_action, color_action, log_p1, log_p2 = self._select_actions(obs)
        action = HierarchicalAction(move_action, color_action)
        info = HierarchicalStepInfo(action, log_p1, log_p2)
        return move_action, color_action, info

    def act_greedy(self, obs: np.ndarray) -> tuple[int, int, HierarchicalStepInfo]:
        move_action, color_action, log_p1, log_p2 = self._select_actions(obs, greedy=True)
        action = HierarchicalAction(move_action, color_action)
        info = HierarchicalStepInfo(action, log_p1, log_p2)
        return move_action, color_action, info

    def save_checkpoint(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "move_net": self.move_net.state_dict(),
                "color_net": self.color_net.state_dict(),
                "config": {
                    "obs_dim": self._obs_dim,
                    "color_input_dim": self._color_input_dim,
                    "hidden_size": self._hidden_size,
                    "num_layers": self._num_layers,
                    "group_size": self.group_size,
                    "num_groups": self.num_groups,
                    "train_epochs": self.train_epochs,
                    "entropy_coef": self.entropy_coef,
                    "clip_ratio": self.clip_ratio,
                    "advantage_eps": self.advantage_eps,
                    "grad_clip_norm": self.grad_clip_norm,
                    "seed": self._seed,
                },
            },
            path,
        )
        return path

    @classmethod
    def from_checkpoint(cls, path: Path | str, *, device: str = "cpu") -> "GRPOAgent":
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["config"]
        agent = cls(
            obs_dim=config["obs_dim"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            group_size=config.get("group_size", 16),
            num_groups=config.get("num_groups", 4),
            train_epochs=config.get("train_epochs", 2),
            entropy_coef=config.get("entropy_coef", 0.01),
            clip_ratio=config.get("clip_ratio", 0.2),
            advantage_eps=config.get("advantage_eps", 1e-8),
            grad_clip_norm=config.get("grad_clip_norm", 1.0),
            seed=config.get("seed", 0),
            device=device,
        )
        agent.move_net.load_state_dict(payload["move_net"])
        agent.color_net.load_state_dict(payload["color_net"])
        agent.move_net.eval()
        agent.color_net.eval()
        return agent

    def rollout_episode(self, env: GridEnv) -> Episode:
        episode = Episode()
        obs, _, _ = env.reset()
        done = False

        while not done:
            move_action, color_action, log_p1, log_p2 = self._select_actions(obs)
            next_obs, reward, done, _state = env.step(move_action, color_action)
            episode.steps.append(
                StepRecord(
                    obs=obs.copy(),
                    move_action=move_action,
                    color_action=color_action,
                    log_prob_model1=log_p1,
                    log_prob_model2=log_p2,
                )
            )
            obs = next_obs
            if done:
                episode.return_ = float(reward)

        return episode

    def _assign_group_advantages(self, episodes: list[Episode]) -> None:
        returns = np.asarray([ep.return_ for ep in episodes], dtype=np.float64)
        mean = returns.mean()
        std = returns.std()
        if std < self.advantage_eps:
            advantages = returns - mean
        else:
            advantages = (returns - mean) / (std + self.advantage_eps)

        for ep, adv in zip(episodes, advantages):
            for step in ep.steps:
                step.advantage = float(adv)

    def _policy_loss(self, steps: list[StepRecord]) -> tuple[torch.Tensor, dict[str, float]]:
        obs = torch.as_tensor(np.stack([s.obs for s in steps]), dtype=torch.float32, device=self.device)
        move = torch.as_tensor([s.move_action for s in steps], dtype=torch.long, device=self.device)
        color = torch.as_tensor([s.color_action for s in steps], dtype=torch.long, device=self.device)
        old_log_prob = torch.as_tensor(
            [s.log_prob_joint for s in steps], dtype=torch.float32, device=self.device
        )
        advantage = torch.as_tensor(
            [s.advantage for s in steps], dtype=torch.float32, device=self.device
        )

        move_rep = self.move_net.encode(obs)
        move_dist = self.move_net.dist(obs)
        log_p1 = move_dist.log_prob(move)
        ent1 = move_dist.entropy()

        log_p2, ent2 = self.color_net.log_prob_and_entropy(move_rep, move, color)
        log_prob = log_p1 + log_p2
        ratio = torch.exp(log_prob - old_log_prob)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy = (ent1 + ent2).mean()
        loss = policy_loss - self.entropy_coef * entropy

        stats = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()),
            "ratio_mean": float(ratio.mean().item()),
        }
        return loss, stats

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        steps = [step for ep in episodes for step in ep.steps]
        if not steps:
            return {"loss": 0.0, "policy_loss": 0.0, "entropy": 0.0, "ratio_mean": 1.0}

        totals = {"loss": 0.0, "policy_loss": 0.0, "entropy": 0.0, "ratio_mean": 0.0}
        for _ in range(self.train_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss, stats = self._policy_loss(steps)
            loss.backward()
            if self.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    list(self.move_net.parameters()) + list(self.color_net.parameters()),
                    self.grad_clip_norm,
                )
            self.optimizer.step()
            for key in totals:
                totals[key] += stats[key]

        n = float(self.train_epochs)
        return {key: value / n for key, value in totals.items()}

    def train(
        self,
        env: GridEnv,
        num_updates: int,
        log_every: int = 10,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []

        for update in range(1, num_updates + 1):
            episodes: list[Episode] = []
            for _ in range(self.num_groups):
                group_eps = [self.rollout_episode(env) for _ in range(self.group_size)]
                self._assign_group_advantages(group_eps)
                episodes.extend(group_eps)

            opt_stats = self.update(episodes)
            returns = [ep.return_ for ep in episodes]
            row: dict[str, Any] = {
                "step": update,
                "mean_return": float(np.mean(returns)),
                "max_return": float(np.max(returns)),
                "min_return": float(np.min(returns)),
                **opt_stats,
            }
            history.append(row)

            if update == 1 or update % log_every == 0:
                print(
                    f"update {update:4d}  "
                    f"return={row['mean_return']:.3f} "
                    f"(min={row['min_return']:.3f}, max={row['max_return']:.3f})  "
                    f"loss={row['loss']:.4f}  "
                    f"entropy={row['entropy']:.3f}  "
                    f"ratio={row['ratio_mean']:.3f}"
                )

        return history


def load_agent_from_checkpoint(path: Path | str, *, device: str = "cpu"):
    """Load GRPO or IPS-GRPO agent based on checkpoint metadata."""
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("agent_type") == "ips_grpo":
        from ips_grpo import IPSGRPOAgent

        return IPSGRPOAgent.from_checkpoint(path, device=device)
    return GRPOAgent.from_checkpoint(path, device=device)


def agent_label_from_checkpoint(path: Path | str) -> str:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    agent_type = payload.get("agent_type", "grpo")
    return agent_type.replace("_", "-").upper()
