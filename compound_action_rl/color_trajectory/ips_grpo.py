"""IPS-GRPO with group-based p_hat and advantage scaling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from grpo import Episode, GRPOAgent, StepRecord
from sampling_comparison import grid_tuple

AdvantageMode = Literal[
    "scale_reward_then_normalize",
    "normalize_reward_then_scale_advantage",
    "reward_only",
    "reward_over_phat",
]
PHatMode = Literal["group"]


class IPSGRPOAgent(GRPOAgent):
    """GRPO with inverse-probability-scaled trajectory advantages."""

    def __init__(
        self,
        obs_dim: int,
        *,
        p_eps: float = 1e-8,
        max_inverse_weight: float | None = None,
        advantage_mode: AdvantageMode = "scale_reward_then_normalize",
        p_hat_mode: PHatMode = "group",
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, **kwargs)
        self.p_eps = p_eps
        self.max_inverse_weight = max_inverse_weight
        self.advantage_mode = advantage_mode
        self.p_hat_mode = p_hat_mode

    def save_checkpoint(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent_type": "ips_grpo",
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
                    "p_eps": self.p_eps,
                    "max_inverse_weight": self.max_inverse_weight,
                    "advantage_mode": self.advantage_mode,
                    "p_hat_mode": self.p_hat_mode,
                },
            },
            path,
        )
        return path

    @classmethod
    def from_checkpoint(cls, path: Path | str, *, device: str = "cpu") -> "IPSGRPOAgent":
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["config"]
        agent = cls(
            obs_dim=config["obs_dim"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            group_size=config.get("group_size", 64),
            num_groups=config.get("num_groups", 4),
            train_epochs=config.get("train_epochs", 2),
            entropy_coef=config.get("entropy_coef", 0.01),
            clip_ratio=config.get("clip_ratio", 0.2),
            advantage_eps=config.get("advantage_eps", 1e-8),
            grad_clip_norm=config.get("grad_clip_norm", 1.0),
            seed=config.get("seed", 0),
            device=device,
            p_eps=config.get("p_eps", 1e-8),
            max_inverse_weight=config.get("max_inverse_weight"),
            advantage_mode=config.get("advantage_mode", "scale_reward_then_normalize"),
            p_hat_mode=config.get("p_hat_mode", "group"),
        )
        agent.move_net.load_state_dict(payload["move_net"])
        agent.color_net.load_state_dict(payload["color_net"])
        agent.move_net.eval()
        agent.color_net.eval()
        return agent

    def rollout_episode(self, env) -> Episode:
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
                episode.outcome = grid_tuple(env._colors)

        return episode

    def _inverse_weights(self, episodes: list[Episode]) -> np.ndarray:
        if self.p_hat_mode != "group":
            raise ValueError(f"Unsupported p_hat_mode: {self.p_hat_mode}")

        n = len(episodes)
        counts = Counter(ep.outcome for ep in episodes)
        weights = np.empty(n, dtype=np.float64)
        for i, ep in enumerate(episodes):
            p_hat = max(counts[ep.outcome] / n, self.p_eps)
            weight = 1.0 / p_hat
            if self.max_inverse_weight is not None:
                weight = min(weight, self.max_inverse_weight)
            weights[i] = weight
        return weights

    def _assign_group_advantages(self, episodes: list[Episode]) -> None:
        returns = np.asarray([ep.return_ for ep in episodes], dtype=np.float64)
        inverse_weights = self._inverse_weights(episodes)

        if self.advantage_mode == "reward_only":
            scaled = returns
        elif self.advantage_mode == "scale_reward_then_normalize":
            scaled = returns * inverse_weights
        elif self.advantage_mode == "normalize_reward_then_scale_advantage":
            mean = returns.mean()
            std = returns.std()
            if std < self.advantage_eps:
                normalized = returns - mean
            else:
                normalized = (returns - mean) / (std + self.advantage_eps)
            scaled = normalized * inverse_weights
        elif self.advantage_mode == "reward_over_phat":
            scaled = returns * inverse_weights
        else:
            raise ValueError(f"Unsupported advantage_mode: {self.advantage_mode}")

        if self.advantage_mode in {"scale_reward_then_normalize", "reward_only"}:
            mean = scaled.mean()
            std = scaled.std()
            if std < self.advantage_eps:
                advantages = scaled - mean
            else:
                advantages = (scaled - mean) / (std + self.advantage_eps)
        else:
            advantages = scaled

        for ep, adv in zip(episodes, advantages):
            for step in ep.steps:
                step.advantage = float(adv)

    def train(
        self,
        env,
        num_updates: int,
        log_every: int = 10,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []

        for update in range(1, num_updates + 1):
            episodes: list[Episode] = []
            mean_p_hat = 0.0
            mean_inv_weight = 0.0
            group_count = 0

            for _ in range(self.num_groups):
                group_eps = [self.rollout_episode(env) for _ in range(self.group_size)]
                counts = Counter(ep.outcome for ep in group_eps)
                for ep in group_eps:
                    mean_p_hat += counts[ep.outcome] / self.group_size
                    mean_inv_weight += 1.0 / max(counts[ep.outcome] / self.group_size, self.p_eps)
                    group_count += 1
                self._assign_group_advantages(group_eps)
                episodes.extend(group_eps)

            opt_stats = self.update(episodes)
            returns = [ep.return_ for ep in episodes]
            row: dict[str, Any] = {
                "step": update,
                "mean_return": float(np.mean(returns)),
                "max_return": float(np.max(returns)),
                "min_return": float(np.min(returns)),
                "mean_p_hat": float(mean_p_hat / max(group_count, 1)),
                "mean_inv_weight": float(mean_inv_weight / max(group_count, 1)),
                **opt_stats,
            }
            history.append(row)

            if update == 1 or update % log_every == 0:
                print(
                    f"update {update:4d}  "
                    f"return={row['mean_return']:.3f} "
                    f"(min={row['min_return']:.3f}, max={row['max_return']:.3f})  "
                    f"p_hat={row['mean_p_hat']:.3f}  "
                    f"inv_w={row['mean_inv_weight']:.2f}  "
                    f"loss={row['loss']:.4f}  "
                    f"entropy={row['entropy']:.3f}  "
                    f"ratio={row['ratio_mean']:.3f}"
                )

        return history
