"""Energy-model IPS with group-relative PPO on the compound-action DAG.

This is the DAG-toy counterpart of the Energy-IPS/GRPO setup:

    p_tilde_phi(o) = exp(-E_phi(o))
    scaled_reward  = R(o) / p_hat_phi(o)
    advantage      = group_normalize(scaled_reward)

The energy model is trained with noise-contrastive estimation (NCE). Positive
examples are terminal outcomes sampled by the current behavior policy and
negative examples are uniformly sampled from the terminal frontier. The
resulting density can be used either as an unnormalized score or normalized
over the outcomes in each rollout group.

The DAG policy remains hierarchical: one network selects direction and a
second selects the masked step size. The energy head can share the direction
policy's state encoder or use an independent encoder. PPO assigns the same
terminal group-relative advantage to every compound action in a trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from config import TrainConfig  # noqa: E402
from count_ips import CountIPSTrainer, Episode  # noqa: E402
from dag_env import State  # noqa: E402
from exact_probability_ips import _resolve_device  # noqa: E402
from run_count_ips import (  # noqa: E402
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)


ADVANTAGE_MODES = (
    "scale_reward_then_normalize",
    "normalize_reward_then_scale_advantage",
    "reward_only",
    "reward_over_phat",
)
P_HAT_MODES = ("normalised", "unnormalised")
P_HAT_TIMINGS = ("before_density_update", "after_density_update")
TRUNK_MODES = ("shared", "separate")
DENSITY_PARAM_SCOPES = ("all", "energy_trunk_and_head", "energy_head_only")


@dataclass(frozen=True)
class EnergyIPSConfig:
    """Energy-density and inverse-propensity settings."""

    density_lr: float | None = None
    density_loss_coef: float = 1.0
    p_eps: float = 1e-8
    max_inverse_weight: float | None = None
    advantage_mode: str = "scale_reward_then_normalize"
    p_hat_mode: str = "normalised"
    phat_timing: str = "before_density_update"
    trunk_mode: str = "shared"
    density_param_scope: str = "all"

    def validate(self) -> None:
        if self.density_lr is not None and self.density_lr <= 0.0:
            raise ValueError("density_lr must be > 0 when set")
        if self.density_loss_coef < 0.0:
            raise ValueError("density_loss_coef must be >= 0")
        if self.p_eps <= 0.0:
            raise ValueError("p_eps must be > 0")
        if (
            self.max_inverse_weight is not None
            and self.max_inverse_weight <= 0.0
        ):
            raise ValueError("max_inverse_weight must be > 0 when set")
        if self.advantage_mode not in ADVANTAGE_MODES:
            raise ValueError(f"unknown advantage_mode: {self.advantage_mode}")
        if self.p_hat_mode not in P_HAT_MODES:
            raise ValueError(f"unknown p_hat_mode: {self.p_hat_mode}")
        if self.phat_timing not in P_HAT_TIMINGS:
            raise ValueError(f"unknown phat_timing: {self.phat_timing}")
        if self.trunk_mode not in TRUNK_MODES:
            raise ValueError(f"unknown trunk_mode: {self.trunk_mode}")
        if self.density_param_scope not in DENSITY_PARAM_SCOPES:
            raise ValueError(
                f"unknown density_param_scope: {self.density_param_scope}"
            )


def _energy_trunk(
    input_dim: int, hidden_size: int, num_layers: int
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(num_layers):
        layers.extend((nn.Linear(width, hidden_size), nn.Tanh()))
        width = hidden_size
    return nn.Sequential(*layers)


class EnergyHead(nn.Module):
    """Map an encoded terminal state to one scalar energy."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.net(representation).squeeze(-1)


def energy_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    p_hat_unnormalised: Sequence[float],
    p_hat_normalised: Sequence[float],
    *,
    p_hat_mode: str = "normalised",
    advantage_mode: str = "scale_reward_then_normalize",
    p_eps: float = 1e-8,
    max_inverse_weight: float | None = None,
    normalize_group_advantages: bool = True,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute Energy-IPS advantages and diagnostics for one rollout group."""
    size = len(rewards)
    if (
        size == 0
        or len(outcome_ids) != size
        or len(p_hat_unnormalised) != size
        or len(p_hat_normalised) != size
    ):
        raise ValueError(
            "rewards, outcomes, and both p_hat arrays must have equal non-zero size"
        )
    if p_hat_mode not in P_HAT_MODES:
        raise ValueError(f"unknown p_hat_mode: {p_hat_mode}")
    if advantage_mode not in ADVANTAGE_MODES:
        raise ValueError(f"unknown advantage_mode: {advantage_mode}")
    if p_eps <= 0.0 or eps <= 0.0:
        raise ValueError("p_eps and eps must be > 0")
    if max_inverse_weight is not None and max_inverse_weight <= 0.0:
        raise ValueError("max_inverse_weight must be > 0 when set")

    reward_array = np.asarray(rewards, dtype=np.float64)
    unnormalised = np.asarray(p_hat_unnormalised, dtype=np.float64)
    normalised = np.asarray(p_hat_normalised, dtype=np.float64)
    if np.any(~np.isfinite(reward_array)):
        raise ValueError("rewards must be finite")
    if (
        np.any(~np.isfinite(unnormalised))
        or np.any(~np.isfinite(normalised))
        or np.any(unnormalised < 0.0)
        or np.any(normalised < 0.0)
    ):
        raise ValueError("p_hat values must be finite and non-negative")

    selected = normalised if p_hat_mode == "normalised" else unnormalised
    selected = np.maximum(selected, p_eps)
    inverse_weights = 1.0 / selected
    if max_inverse_weight is not None:
        inverse_weights = np.minimum(inverse_weights, max_inverse_weight)
    p_hat_for_scaling = 1.0 / inverse_weights
    scaled_rewards = reward_array * inverse_weights

    reward_std = float(reward_array.std())
    scaled_std = float(scaled_rewards.std())
    if advantage_mode == "scale_reward_then_normalize":
        if normalize_group_advantages:
            centered = scaled_rewards - scaled_rewards.mean()
            advantages = (
                centered
                if scaled_std < eps
                else centered / (scaled_std + eps)
            )
        else:
            advantages = scaled_rewards
    elif advantage_mode == "normalize_reward_then_scale_advantage":
        centered_reward = reward_array - reward_array.mean()
        normalized_reward = (
            centered_reward
            if reward_std < eps
            else centered_reward / (reward_std + eps)
        )
        advantages = normalized_reward * inverse_weights
    elif advantage_mode == "reward_only":
        advantages = reward_array.copy()
    else:
        advantages = scaled_rewards

    squared_sum = float(np.square(inverse_weights).sum())
    ess = float(
        inverse_weights.sum() ** 2 / max(squared_sum, np.finfo(float).tiny)
    )
    counts = Counter(outcome_ids)
    metrics = {
        "ips_prob_mean": float(p_hat_for_scaling.mean()),
        "ips_prob_min": float(p_hat_for_scaling.min()),
        "ips_prob_max": float(p_hat_for_scaling.max()),
        "ips_unique_outcomes": float(len(counts)),
        "ips_max_outcome_count": float(max(counts.values())),
        "ips_min_outcome_count": float(min(counts.values())),
        "ips_scaled_reward_mean": float(scaled_rewards.mean()),
        "ips_scaled_reward_std": scaled_std,
        "ips_inverse_weight_mean": float(inverse_weights.mean()),
        "ips_inverse_weight_max": float(inverse_weights.max()),
        "ips_ess": ess,
        "ips_ess_fraction": ess / size,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        "p_hat_unnormalised_mean": float(unnormalised.mean()),
        "p_hat_unnormalised_min": float(unnormalised.min()),
        "p_hat_unnormalised_max": float(unnormalised.max()),
        "p_hat_normalised_sum": float(normalised.sum()),
    }
    return advantages, metrics


def _unique_parameters(
    parameters: Sequence[nn.Parameter],
) -> list[nn.Parameter]:
    unique: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        identifier = id(parameter)
        if identifier not in seen:
            seen.add(identifier)
            unique.append(parameter)
    return unique


class EnergyIPSTrainer(CountIPSTrainer):
    """NCE terminal energy model plus group-relative compound-action PPO."""

    probability_label = "energy_p_hat"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        energy_config: EnergyIPSConfig | None = None,
    ) -> None:
        super().__init__(config, device=device)
        self.energy_config = energy_config or EnergyIPSConfig()
        self.energy_config.validate()

        self.energy_trunk: nn.Module | None
        if self.energy_config.trunk_mode == "separate":
            self.energy_trunk = _energy_trunk(
                self.env.obs_dim,
                self.config.hidden_size,
                self.config.num_layers,
            ).to(self.device)
        else:
            self.energy_trunk = None
        self.energy_head = EnergyHead(self.config.hidden_size).to(self.device)

        width = self.config.budget + 1
        terminal_x = torch.arange(width, device=self.device)
        terminal_y = self.config.budget - terminal_x
        terminal_remaining = torch.zeros_like(terminal_x)
        self._terminal_observations = torch.cat(
            (
                F.one_hot(terminal_x, num_classes=width),
                F.one_hot(terminal_y, num_classes=width),
                F.one_hot(terminal_remaining, num_classes=width),
            ),
            dim=-1,
        ).to(dtype=torch.float32)
        self._terminal_indices = {
            state: index for index, state in enumerate(self.terminals)
        }

        self._policy_parameters = _unique_parameters(
            list(self.direction_policy.parameters())
            + list(self.step_policy.parameters())
        )
        self._energy_only_parameters = _unique_parameters(
            (
                list(self.energy_trunk.parameters())
                if self.energy_trunk is not None
                else []
            )
            + list(self.energy_head.parameters())
        )
        self._all_parameters = _unique_parameters(
            self._policy_parameters + self._energy_only_parameters
        )
        density_lr = (
            self.config.lr
            if self.energy_config.density_lr is None
            else self.energy_config.density_lr
        )

        if self.energy_config.phat_timing == "before_density_update":
            # The pre-update mode performs one joint policy+density step.
            self.optimizer = torch.optim.Adam(
                self._all_parameters, lr=self.config.lr
            )
            self._density_parameters = self._all_parameters
            self.density_optimizer = self.optimizer
        else:
            self.optimizer = torch.optim.Adam(
                self._policy_parameters, lr=self.config.lr
            )
            self._density_parameters = self._resolve_density_parameters()
            self.density_optimizer = torch.optim.Adam(
                self._density_parameters, lr=density_lr
            )

    def _resolve_density_parameters(self) -> list[nn.Parameter]:
        scope = self.energy_config.density_param_scope
        if scope == "all":
            return self._all_parameters
        if scope == "energy_head_only":
            return list(self.energy_head.parameters())
        trunk = (
            self.direction_policy.trunk
            if self.energy_trunk is None
            else self.energy_trunk
        )
        return _unique_parameters(
            list(trunk.parameters()) + list(self.energy_head.parameters())
        )

    def _zero_all_gradients(self) -> None:
        self.direction_policy.zero_grad(set_to_none=True)
        self.step_policy.zero_grad(set_to_none=True)
        if self.energy_trunk is not None:
            self.energy_trunk.zero_grad(set_to_none=True)
        self.energy_head.zero_grad(set_to_none=True)

    def _outcome_indices(self, episodes: Sequence[Episode]) -> torch.Tensor:
        if not episodes:
            raise ValueError("episodes must be non-empty")
        try:
            indices = [
                self._terminal_indices[episode.terminal] for episode in episodes
            ]
        except KeyError as error:
            raise ValueError(
                "episode has an outcome outside the terminal frontier"
            ) from error
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _state_energy(self, observations: torch.Tensor) -> torch.Tensor:
        if self.energy_trunk is None:
            representation = self.direction_policy.encode(observations)
        else:
            representation = self.energy_trunk(observations)
        return self.energy_head(representation)

    def _compute_energy_loss(self, episodes: Sequence[Episode]) -> torch.Tensor:
        outcome_indices = self._outcome_indices(episodes)
        terminal_energy = self._state_energy(
            self._terminal_observations.index_select(0, outcome_indices)
        )
        noise_indices = torch.randint(
            len(self.terminals),
            (len(episodes),),
            dtype=torch.long,
            device=self.device,
        )
        noise_energy = self._state_energy(
            self._terminal_observations.index_select(0, noise_indices)
        )

        log_noise_probability = -math.log(float(len(self.terminals)))
        positive_logits = -terminal_energy - log_noise_probability
        negative_logits = -noise_energy - log_noise_probability
        positive_loss = F.softplus(-positive_logits).mean()
        negative_loss = F.softplus(negative_logits).mean()
        return positive_loss + negative_loss

    @torch.inference_mode()
    def _energy_diagnostics(
        self, episodes: Sequence[Episode]
    ) -> dict[str, float]:
        all_energies = self._state_energy(self._terminal_observations).float()
        log_probabilities = F.log_softmax(-all_energies, dim=0)
        probabilities = log_probabilities.exp()
        outcome_indices = self._outcome_indices(episodes)
        return {
            "density_nll_outcomes": float(
                -log_probabilities[outcome_indices].mean().item()
            ),
            "density_entropy": float(
                -(probabilities * log_probabilities).sum().item()
            ),
            "density_energy_mean": float(all_energies.mean().item()),
            "density_energy_min": float(all_energies.min().item()),
            "density_energy_max": float(all_energies.max().item()),
            "density_probability_min": float(probabilities.min().item()),
            "density_probability_max": float(probabilities.max().item()),
        }

    @torch.inference_mode()
    def terminal_density_distribution(self) -> dict[State, float]:
        """Return the globally normalized energy density for diagnostics."""
        energies = self._state_energy(self._terminal_observations).float()
        probabilities = torch.softmax(-energies, dim=0).cpu().tolist()
        return {
            state: float(probability)
            for state, probability in zip(self.terminals, probabilities)
        }

    @torch.inference_mode()
    def _energy_p_hat(
        self, episodes: Sequence[Episode]
    ) -> tuple[np.ndarray, np.ndarray]:
        outcome_indices = self._outcome_indices(episodes)
        energies = self._state_energy(
            self._terminal_observations.index_select(0, outcome_indices)
        ).float()
        unnormalised = torch.exp(-energies)
        normalised = unnormalised / unnormalised.sum().clamp_min(
            self.config.advantage_eps
        )
        return (
            unnormalised.cpu().numpy().astype(np.float64),
            normalised.cpu().numpy().astype(np.float64),
        )

    def _group_advantages(self, episodes: list[Episode]) -> float:
        unnormalised, normalised = self._energy_p_hat(episodes)
        advantages, metrics = energy_ips_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            unnormalised,
            normalised,
            p_hat_mode=self.energy_config.p_hat_mode,
            advantage_mode=self.energy_config.advantage_mode,
            p_eps=self.energy_config.p_eps,
            max_inverse_weight=self.energy_config.max_inverse_weight,
            eps=self.config.advantage_eps,
        )
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]

    def _density_update(
        self, episodes: list[Episode]
    ) -> dict[str, float]:
        density_loss_total = 0.0
        density_grad_norm_total = 0.0
        for _ in range(self.config.train_epochs):
            self._zero_all_gradients()
            energy_loss = self._compute_energy_loss(episodes)
            density_loss = self.energy_config.density_loss_coef * energy_loss
            density_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self._density_parameters, self.config.grad_clip_norm
            )
            self.density_optimizer.step()
            density_loss_total += float(energy_loss.item())
            density_grad_norm_total += float(grad_norm.item())
        metrics = {
            "density_loss": density_loss_total / self.config.train_epochs,
            "energy_loss": density_loss_total / self.config.train_epochs,
            "density_grad_norm": (
                density_grad_norm_total / self.config.train_epochs
            ),
        }
        metrics.update(self._energy_diagnostics(episodes))
        return metrics

    def _collect_training_groups(
        self,
    ) -> tuple[list[list[Episode]], list[dict[str, float]]]:
        # Every rollout comes from one frozen behavior policy. In the
        # after-density mode, the density phase then deliberately precedes
        # p_hat/advantage computation, matching the reference implementation.
        groups = [
            self.rollout_batch(self.config.group_size, explore=True)
            for _ in range(self._groups_for_update())
        ]
        all_episodes = [episode for group in groups for episode in group]
        density_metrics: dict[str, float] = {}
        if self.energy_config.phat_timing == "after_density_update":
            density_metrics = self._density_update(all_episodes)

        group_metrics: list[dict[str, float]] = []
        for group in groups:
            self._group_advantages(group)
            metrics = dict(self._last_ips_metrics)
            metrics.update(density_metrics)
            group_metrics.append(metrics)
        return groups, group_metrics

    def _joint_policy_density_update(
        self, episodes: list[Episode]
    ) -> dict[str, float]:
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "entropy": 0.0,
            "mean_importance_ratio": 0.0,
            "max_importance_ratio": 0.0,
            "min_importance_ratio": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "param_norm": 0.0,
            "density_loss": 0.0,
            "energy_loss": 0.0,
            "density_grad_norm": 0.0,
            "joint_loss": 0.0,
        }
        for _ in range(self.config.train_epochs):
            self._zero_all_gradients()
            policy_total_loss, policy_stats = self._joint_policy_loss(episodes)
            energy_loss = self._compute_energy_loss(episodes)
            joint_loss = (
                policy_total_loss
                + self.energy_config.density_loss_coef * energy_loss
            )
            joint_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self._all_parameters, self.config.grad_clip_norm
            )
            self.optimizer.step()

            policy_stats["grad_norm"] = float(grad_norm.item())
            policy_stats["param_norm"] = float(
                sum(
                    parameter.detach().norm().item() ** 2
                    for parameter in self._all_parameters
                )
                ** 0.5
            )
            policy_stats["density_loss"] = float(energy_loss.item())
            policy_stats["energy_loss"] = float(energy_loss.item())
            # In a joint backward pass this is the combined gradient norm.
            policy_stats["density_grad_norm"] = float(grad_norm.item())
            policy_stats["joint_loss"] = float(joint_loss.item())
            for key in totals:
                totals[key] += policy_stats[key]

        averaged = {
            key: value / self.config.train_epochs
            for key, value in totals.items()
        }
        averaged.update(self._energy_diagnostics(episodes))
        return averaged

    def _update_training_groups(
        self, groups: list[list[Episode]]
    ) -> dict[str, float]:
        episodes = [episode for group in groups for episode in group]
        if self.energy_config.phat_timing == "before_density_update":
            return self._joint_policy_density_update(episodes)
        # The density update already occurred in _collect_training_groups.
        return super()._update_training_groups(groups)

    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "direction_policy": self.direction_policy.state_dict(),
                "step_policy": self.step_policy.state_dict(),
                "energy_head": self.energy_head.state_dict(),
                "energy_trunk": (
                    self.energy_trunk.state_dict()
                    if self.energy_trunk is not None
                    else None
                ),
                "energy_config": asdict(self.energy_config),
                "update_step": update_step,
                "algorithm": {
                    "name": "energy_ips",
                    "raw_weight": "R(o) / p_hat_energy(o)",
                    "forward_loss": "group_relative_token_ppo",
                },
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "EnergyIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("algorithm", {}).get("name") != "energy_ips":
            raise ValueError("checkpoint is not an Energy-IPS run")
        energy_config_payload = payload["energy_config"]
        energy_config = (
            EnergyIPSConfig(**energy_config_payload)
            if isinstance(energy_config_payload, dict)
            else energy_config_payload
        )
        trainer = cls(
            payload["config"],
            device=device,
            energy_config=energy_config,
        )
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.energy_head.load_state_dict(payload["energy_head"])
        if trainer.energy_trunk is not None:
            if payload["energy_trunk"] is None:
                raise ValueError("checkpoint is missing the separate energy trunk")
            trainer.energy_trunk.load_state_dict(payload["energy_trunk"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        trainer.energy_head.eval()
        if trainer.energy_trunk is not None:
            trainer.energy_trunk.eval()
        return trainer


def _plot_energy_training(history: list[dict], *, output: Path) -> None:
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    axes[0, 0].plot(
        steps, [row["density_loss"] for row in history], label="NCE"
    )
    axes[0, 0].plot(
        steps,
        [row["density_nll_outcomes"] for row in history],
        label="global diagnostic NLL",
    )
    axes[0, 0].set_title("Energy-density losses")
    axes[0, 0].legend()

    axes[0, 1].plot(
        steps,
        [row["density_probability_min"] for row in history],
        label="minimum",
    )
    axes[0, 1].plot(
        steps,
        [row["density_probability_max"] for row in history],
        label="maximum",
    )
    axes[0, 1].set_title("Globally normalized energy density")
    axes[0, 1].legend()

    axes[1, 0].plot(
        steps,
        [row["ips_inverse_weight_mean"] for row in history],
        label="mean",
    )
    axes[1, 0].plot(
        steps,
        [row["ips_inverse_weight_max"] for row in history],
        label="maximum",
    )
    axes[1, 0].set_title("Inverse propensity weights")
    axes[1, 0].legend()

    axes[1, 1].plot(
        steps,
        [row["density_entropy"] for row in history],
        label="density entropy",
    )
    axes[1, 1].plot(
        steps,
        [row["ips_ess_fraction"] for row in history],
        label="IPS ESS / group",
    )
    axes[1, 1].set_title("Density spread and IPS stability")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Update")
        axis.grid(alpha=0.22)
    fig.suptitle("Energy-IPS diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--density-lr", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--density-loss-coef", type=float, default=1.0)
    parser.add_argument("--p-eps", type=float, default=1e-8)
    parser.add_argument("--max-inverse-weight", type=float, default=None)
    parser.add_argument(
        "--advantage-mode", choices=ADVANTAGE_MODES,
        default="scale_reward_then_normalize",
    )
    parser.add_argument(
        "--p-hat-mode", choices=P_HAT_MODES, default="normalised"
    )
    parser.add_argument(
        "--phat-timing",
        choices=P_HAT_TIMINGS,
        default="before_density_update",
    )
    parser.add_argument(
        "--trunk-mode", choices=TRUNK_MODES, default="shared"
    )
    parser.add_argument(
        "--density-param-scope",
        choices=DENSITY_PARAM_SCOPES,
        default="all",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--terminal-rewards", type=float, nargs="+", default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        budget=args.budget,
        max_step=args.max_step,
        terminal_rewards=(
            tuple(args.terminal_rewards)
            if args.terminal_rewards is not None
            else None
        ),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        group_size=args.group_size,
        num_groups=args.num_groups,
        num_updates=args.num_updates,
        train_epochs=args.train_epochs,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        clip_ratio=args.clip_ratio,
        seed=args.seed,
        log_every=args.log_every,
    )
    energy_config = EnergyIPSConfig(
        density_lr=args.density_lr,
        density_loss_coef=args.density_loss_coef,
        p_eps=args.p_eps,
        max_inverse_weight=args.max_inverse_weight,
        advantage_mode=args.advantage_mode,
        p_hat_mode=args.p_hat_mode,
        phat_timing=args.phat_timing,
        trunk_mode=args.trunk_mode,
        density_param_scope=args.density_param_scope,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "energy_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = EnergyIPSTrainer(
        config,
        device=_resolve_device(args.device),
        energy_config=energy_config,
    )
    checkpoint_every = args.checkpoint_every or None

    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Algorithm: Energy-NCE + R(o)/p_hat(o) group-relative PPO; "
        f"p_hat={energy_config.p_hat_mode}; "
        f"timing={energy_config.phat_timing}; "
        f"trunk={energy_config.trunk_mode}; "
        f"density_scope={energy_config.density_param_scope}"
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "energy_config": asdict(energy_config),
                "device": str(trainer.device),
                "algorithm": "energy_ips",
                "raw_weight": "R(o) / p_hat_energy(o)",
                "forward_loss": "group_relative_token_ppo",
                "checkpoint_every": checkpoint_every,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    history = trainer.train(
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=run_dir / "checkpoints",
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    trainer.save(run_dir / "checkpoint.pt", update_step=config.num_updates)

    training_plot = run_dir / "training_curves.png"
    _plot_training_curves(
        history,
        trainer,
        output=training_plot,
        propensity_title=f"Energy p_hat ({energy_config.p_hat_mode})",
        suptitle="Energy-IPS group-relative PPO training",
    )
    energy_plot = run_dir / "energy_diagnostics.png"
    _plot_energy_training(history, output=energy_plot)

    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Energy-IPS sampling vs ideal reward sampling",
    )
    trajectory_plot = run_dir / "trajectory_sampling.png"
    trajectories = _plot_trajectory_diagnostics(
        history,
        trainer,
        evaluation,
        samples=args.final_samples,
        output=trajectory_plot,
        subtitle=(
            "The energy model estimates terminal outcomes; paths are diagnostics"
        ),
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary: dict[str, Any] = {
        "environment": trainer.environment_summary(),
        "algorithm": {
            "name": "energy_ips",
            "raw_weight": "R(o) / p_hat_energy(o)",
            "forward_loss": "group_relative_token_ppo",
            **asdict(energy_config),
        },
        "final_energy_density": {
            state.signature: probability
            for state, probability in trainer.terminal_density_distribution().items()
        },
        "final_sampling": sampling,
        "trajectory_sampling": trajectories,
        "best_eval": (
            min(eval_rows, key=lambda row: row["tv_reward_target"])
            if eval_rows
            else None
        ),
        "final_training_diagnostics": {
            key: history[-1][key]
            for key in (
                "density_loss",
                "density_nll_outcomes",
                "density_entropy",
                "ips_ess_fraction",
                "ips_inverse_weight_max",
            )
        },
        "plots": {
            "training_curves": training_plot.name,
            "energy_diagnostics": energy_plot.name,
            "sampling_counts": "sampling_counts.png",
            "trajectory_sampling": trajectory_plot.name,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Final ideal-line R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(
        "Final energy diagnostics: "
        f"NCE={history[-1]['density_loss']:.3f}, "
        f"entropy={history[-1]['density_entropy']:.3f}, "
        f"IPS_ESS={history[-1]['ips_ess_fraction']:.3f}"
    )
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
