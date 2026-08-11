"""PPO with a value critic and an exact terminal-propensity oracle.

This experiment removes the small-group Monte Carlo estimate from Count-IPS.
For the frozen behavior policy used to collect an update, it computes

    p_q(x) = sum_{tau -> x} q(tau)

by forwarding probability mass through the DAG.  Every sampled terminal then
receives the normalized terminal importance target

    W(x) = R(x) / (Z_R * p_q(x)),       Z_R = sum_x R(x),

and a conventional state-value critic learns to predict W(x) from each state
visited on the trajectory.  PPO uses the Monte Carlo advantage

    A(s_t, a_t) = stop_gradient(W(x) - V(s_t)).

The optional log objective instead uses log R(x) - log p_q(x).  It is exposed
as a separate experiment because it has the same ideal terminal target but a
different optimization geometry from raw Count-IPS.

The PPO and propensity-provider boundary is environment independent:
``TerminalProbabilityProvider`` supplies terminal log probabilities.  The
``ExactToyDAGTerminalProbabilityProvider`` below is only the oracle adapter for
this enumerable toy DAG.  It accounts for all merging paths and for the actual
epsilon/temperature behavior distribution.  A learned terminal-density model
can later replace this provider without changing the PPO/value-critic code.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from config import TrainConfig  # noqa: E402
from count_ips import Episode, StepRecord, _pad_episode_values  # noqa: E402
from dag_env import RIGHT, UP, State  # noqa: E402
from epsilon_greedy_count_ips import (  # noqa: E402
    EpsilonGreedyCountIPSTrainer,
    ExplorationConfig,
    _plot_exploration,
    _resolve_device,
)
from run_count_ips import (  # noqa: E402
    _plot_final_counts,
    _plot_trajectory_diagnostics,
)


class TerminalProbabilityProvider(Protocol):
    """Interface consumed by PPO; implementations may be exact or learned."""

    def refresh(self) -> None:
        """Freeze/refresh probabilities for the next rollout and PPO update."""

    def log_probabilities(self, outcomes: Sequence[object]) -> np.ndarray:
        """Return behavior-policy terminal log probabilities in input order."""

    @property
    def behavior_distribution(self) -> Mapping[object, float]:
        """Complete cached behavior distribution when it is enumerable."""

    @property
    def evaluation_distribution(self) -> Mapping[object, float]:
        """Complete cached base-policy distribution when it is enumerable."""


@dataclass(frozen=True)
class OraclePPOConfig:
    """Value and importance-target settings independent of the DAG adapter."""

    objective: str = "raw"
    max_log_weight: float | None = 8.0
    normalize_advantages: bool = True
    value_hidden_size: int = 128
    value_num_layers: int = 2
    value_lr: float = 1e-3
    value_train_epochs: int = 4
    value_grad_clip_norm: float = 1.0

    def validate(self) -> None:
        if self.objective not in ("raw", "log"):
            raise ValueError("objective must be 'raw' or 'log'")
        if self.max_log_weight is not None and self.max_log_weight <= 0.0:
            raise ValueError("max_log_weight must be > 0 when set")
        if self.value_hidden_size < 1 or self.value_num_layers < 1:
            raise ValueError("value network dimensions must be >= 1")
        if self.value_lr <= 0.0:
            raise ValueError("value_lr must be > 0")
        if self.value_train_epochs < 1:
            raise ValueError("value_train_epochs must be >= 1")
        if self.value_grad_clip_norm <= 0.0:
            raise ValueError("value_grad_clip_norm must be > 0")


class ValueNetwork(nn.Module):
    """Environment-agnostic MLP state-value baseline."""

    def __init__(self, obs_dim: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = obs_dim
        for _ in range(num_layers):
            layers.extend((nn.Linear(width, hidden_size), nn.Tanh()))
            width = hidden_size
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


def terminal_importance_targets(
    rewards: Sequence[float],
    terminal_log_probabilities: Sequence[float],
    *,
    reward_normalizer: float,
    objective: str,
    max_log_weight: float | None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Construct stable raw-IPS or log-score critic targets.

    ``reward_normalizer`` is a single outcome-independent constant.  It only
    controls the numerical scale of raw weights and therefore does not alter
    their policy-gradient fixed point.
    """
    reward_array = np.asarray(rewards, dtype=np.float64)
    log_probability = np.asarray(
        terminal_log_probabilities, dtype=np.float64
    )
    if reward_array.size == 0 or reward_array.shape != log_probability.shape:
        raise ValueError("rewards and terminal probabilities must match")
    if np.any(~np.isfinite(reward_array)) or np.any(reward_array <= 0.0):
        raise ValueError("rewards must be finite and strictly positive")
    if np.any(~np.isfinite(log_probability)) or np.any(log_probability > 1e-7):
        raise ValueError("terminal log probabilities must be finite and <= 0")
    if reward_normalizer <= 0.0 or not math.isfinite(reward_normalizer):
        raise ValueError("reward_normalizer must be finite and > 0")
    if objective not in ("raw", "log"):
        raise ValueError("objective must be 'raw' or 'log'")

    unscaled_log_weight = np.log(reward_array) - log_probability
    normalized_log_weight = unscaled_log_weight - math.log(reward_normalizer)
    clipped_log_weight = normalized_log_weight.copy()
    if max_log_weight is not None:
        clipped_log_weight = np.minimum(clipped_log_weight, max_log_weight)
    clipped = clipped_log_weight < normalized_log_weight

    if objective == "raw":
        targets = np.exp(clipped_log_weight)
        ess_weights = targets
    else:
        # Removing log Z_R is an outcome-independent additive constant.  The
        # value critic absorbs it, while advantages preserve the log objective.
        targets = clipped_log_weight
        centered = targets - targets.max()
        ess_weights = np.exp(centered)

    square_sum = float(np.square(ess_weights).sum())
    ess = float(ess_weights.sum() ** 2 / max(square_sum, eps))
    probabilities = np.exp(np.clip(log_probability, -745.0, 0.0))
    return targets, {
        "ips_prob_mean": float(probabilities.mean()),
        "ips_prob_min": float(probabilities.min()),
        "ips_prob_max": float(probabilities.max()),
        "ips_scaled_reward_mean": float(targets.mean()),
        "ips_scaled_reward_std": float(targets.std()),
        "ips_ess": ess,
        "ips_ess_fraction": ess / targets.size,
        "terminal_log_probability_mean": float(log_probability.mean()),
        "terminal_log_probability_min": float(log_probability.min()),
        "terminal_log_probability_max": float(log_probability.max()),
        "log_importance_weight_mean": float(normalized_log_weight.mean()),
        "log_importance_weight_min": float(normalized_log_weight.min()),
        "log_importance_weight_max": float(normalized_log_weight.max()),
        "importance_target_mean": float(targets.mean()),
        "importance_target_min": float(targets.min()),
        "importance_target_max": float(targets.max()),
        "importance_weight_clip_fraction": float(clipped.mean()),
    }


def _distribution_metrics(
    probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    if probabilities.shape != target_probabilities.shape:
        raise ValueError("probability and target arrays must have equal shape")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative")
    positive = probabilities > 0.0
    entropy = float(
        -np.sum(probabilities[positive] * np.log(probabilities[positive]))
    )
    l1 = float(np.abs(probabilities - target_probabilities).sum())
    return {
        f"{prefix}_probability_mass": float(probabilities.sum()),
        f"{prefix}_min_probability": float(probabilities.min()),
        f"{prefix}_max_probability": float(probabilities.max()),
        f"{prefix}_entropy": entropy,
        f"{prefix}_tv_reward_target": 0.5 * l1,
        f"{prefix}_max_reward_target_error": float(
            np.abs(probabilities - target_probabilities).max()
        ),
    }


class ExactToyDAGTerminalProbabilityProvider:
    """Vectorized exact forward DP adapter for the toy's compound-action DAG.

    The provider is the only component that knows about ``State(x, y)`` or the
    hierarchical direction/step action.  PPO sees only terminal identities and
    their log probabilities.
    """

    def __init__(self, trainer: "OracleTerminalPropensityPPOTrainer") -> None:
        self.trainer = trainer
        self.device = trainer.device
        self.budget = trainer.config.budget
        self.max_step = trainer.config.max_step
        self._states = [
            State(x, depth - x)
            for depth in range(self.budget + 1)
            for x in range(depth + 1)
        ]
        self._state_indices = {
            state: index for index, state in enumerate(self._states)
        }
        self._nonterminal_count = len(self._states) - len(trainer.terminals)
        self._terminal_indices = torch.tensor(
            [self._state_indices[state] for state in trainer.terminals],
            dtype=torch.long,
            device=self.device,
        )
        self._observations, self._step_masks = self._build_policy_inputs()
        self._edges_by_depth = self._build_edges()
        self._behavior_distribution: dict[State, float] = {}
        self._evaluation_distribution: dict[State, float] = {}

    def _build_policy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        states = self._states[: self._nonterminal_count]
        width = self.budget + 1
        x = torch.tensor(
            [state.x for state in states], dtype=torch.long, device=self.device
        )
        y = torch.tensor(
            [state.y for state in states], dtype=torch.long, device=self.device
        )
        remaining = self.budget - x - y
        observations = torch.cat(
            (
                F.one_hot(x, num_classes=width),
                F.one_hot(y, num_classes=width),
                F.one_hot(remaining, num_classes=width),
            ),
            dim=-1,
        ).to(dtype=torch.float32)
        step_options = torch.arange(
            self.max_step, device=self.device
        ).unsqueeze(0)
        step_masks = step_options < remaining.clamp(
            max=self.max_step
        ).unsqueeze(1)
        return observations, step_masks

    def _build_edges(
        self,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        edges: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        for depth in range(self.budget):
            sources: list[int] = []
            destinations: list[int] = []
            directions: list[int] = []
            step_indices: list[int] = []
            remaining = self.budget - depth
            for x in range(depth + 1):
                state = State(x, depth - x)
                source = self._state_indices[state]
                for direction in (RIGHT, UP):
                    for length in range(
                        1, min(self.max_step, remaining) + 1
                    ):
                        child = (
                            State(state.x + length, state.y)
                            if direction == RIGHT
                            else State(state.x, state.y + length)
                        )
                        sources.append(source)
                        destinations.append(self._state_indices[child])
                        directions.append(direction)
                        step_indices.append(length - 1)
            edges.append(
                (
                    torch.tensor(
                        sources, dtype=torch.long, device=self.device
                    ),
                    torch.tensor(
                        destinations, dtype=torch.long, device=self.device
                    ),
                    torch.tensor(
                        directions, dtype=torch.long, device=self.device
                    ),
                    torch.tensor(
                        step_indices, dtype=torch.long, device=self.device
                    ),
                )
            )
        return edges

    @torch.inference_mode()
    def _joint_action_probabilities(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direction_mask = torch.ones(
            (self._nonterminal_count, 2),
            dtype=torch.bool,
            device=self.device,
        )
        base_direction, representation = (
            self.trainer.direction_policy.dist_with_rep(
                self._observations, direction_mask
            )
        )
        behavior_direction = self.trainer._action_distribution(
            base_direction, direction_mask, explore=True
        )
        base_joint = torch.zeros(
            (self._nonterminal_count, 2, self.max_step),
            dtype=torch.float64,
            device=self.device,
        )
        behavior_joint = torch.zeros_like(base_joint)
        for direction in (RIGHT, UP):
            chosen_direction = torch.full(
                (self._nonterminal_count,),
                direction,
                dtype=torch.long,
                device=self.device,
            )
            base_step = self.trainer.step_policy.dist(
                representation, chosen_direction, self._step_masks
            )
            behavior_step = self.trainer._action_distribution(
                base_step, self._step_masks, explore=True
            )
            base_joint[:, direction] = (
                base_direction.probs[:, direction, None].to(torch.float64)
                * base_step.probs.to(torch.float64)
            )
            behavior_joint[:, direction] = (
                behavior_direction.probs[:, direction, None].to(torch.float64)
                * behavior_step.probs.to(torch.float64)
            )

        # The policy distributions are evaluated in float32.  Their rows are
        # normalized individually, but multiplying the hierarchical direction
        # and step factors can leave O(1e-7) joint-mass errors.  At budget 128,
        # forwarding those errors through many merging paths can move terminal
        # mass outside a tight conservation check.  Renormalizing once in
        # float64 is only a roundoff correction; it does not change the
        # mathematical behavior distribution.
        base_normalizer = base_joint.sum(dim=(1, 2), keepdim=True)
        behavior_normalizer = behavior_joint.sum(
            dim=(1, 2), keepdim=True
        )
        if (
            torch.any(~torch.isfinite(base_normalizer))
            or torch.any(base_normalizer <= 0.0)
            or torch.any(~torch.isfinite(behavior_normalizer))
            or torch.any(behavior_normalizer <= 0.0)
        ):
            raise RuntimeError("joint action probabilities have invalid mass")
        base_joint = base_joint / base_normalizer
        behavior_joint = behavior_joint / behavior_normalizer

        base_mass = base_joint.sum(dim=(1, 2))
        behavior_mass = behavior_joint.sum(dim=(1, 2))
        if not torch.allclose(
            base_mass, torch.ones_like(base_mass), atol=1e-12, rtol=1e-12
        ):
            raise RuntimeError("base joint action probabilities are not normalized")
        if not torch.allclose(
            behavior_mass,
            torch.ones_like(behavior_mass),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise RuntimeError(
                "behavior joint action probabilities are not normalized"
            )
        return behavior_joint, base_joint

    @torch.inference_mode()
    def _forward_mass(self, joint: torch.Tensor) -> torch.Tensor:
        mass = torch.zeros(
            len(self._states), dtype=torch.float64, device=self.device
        )
        mass[0] = 1.0
        for sources, destinations, directions, step_indices in (
            self._edges_by_depth
        ):
            edge_mass = (
                mass[sources]
                * joint[sources, directions, step_indices]
            )
            mass.index_add_(0, destinations, edge_mass)
        terminals = mass[self._terminal_indices]
        terminal_mass = terminals.sum()
        if not torch.allclose(
            terminal_mass,
            torch.ones((), dtype=torch.float64, device=self.device),
            atol=1e-9,
            rtol=1e-9,
        ):
            raise RuntimeError(
                "exact forward DP terminal mass does not sum to one: "
                f"mass={terminal_mass.item():.17g}, "
                f"error={abs(terminal_mass.item() - 1.0):.3e}"
            )
        if torch.any(terminals <= 0.0):
            raise RuntimeError("exact behavior has a zero-probability terminal")
        return terminals

    @torch.inference_mode()
    def refresh(self) -> None:
        behavior_joint, base_joint = self._joint_action_probabilities()
        behavior = self._forward_mass(behavior_joint).cpu().numpy()
        evaluation = self._forward_mass(base_joint).cpu().numpy()
        self._behavior_distribution = {
            state: float(probability)
            for state, probability in zip(self.trainer.terminals, behavior)
        }
        self._evaluation_distribution = {
            state: float(probability)
            for state, probability in zip(self.trainer.terminals, evaluation)
        }

    def _require_refresh(self) -> None:
        if not self._behavior_distribution:
            raise RuntimeError("refresh() must be called before probability lookup")

    def log_probabilities(self, outcomes: Sequence[object]) -> np.ndarray:
        self._require_refresh()
        try:
            probabilities = np.asarray(
                [self._behavior_distribution[outcome] for outcome in outcomes],
                dtype=np.float64,
            )
        except KeyError as error:
            raise ValueError("outcome is outside the terminal frontier") from error
        return np.log(probabilities)

    @property
    def behavior_distribution(self) -> Mapping[object, float]:
        self._require_refresh()
        return self._behavior_distribution

    @property
    def evaluation_distribution(self) -> Mapping[object, float]:
        self._require_refresh()
        return self._evaluation_distribution


class OracleTerminalPropensityPPOTrainer(EpsilonGreedyCountIPSTrainer):
    """Terminal-marginal oracle, state-value critic, and token-level PPO."""

    probability_label = "p_q(x)"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        exploration: ExplorationConfig | None = None,
        oracle_ppo: OraclePPOConfig | None = None,
    ) -> None:
        super().__init__(config, device=device, exploration=exploration)
        self.oracle_ppo = oracle_ppo or OraclePPOConfig(
            value_hidden_size=self.config.hidden_size,
            value_num_layers=self.config.num_layers,
        )
        self.oracle_ppo.validate()
        self.value_network = ValueNetwork(
            self.env.obs_dim,
            self.oracle_ppo.value_hidden_size,
            self.oracle_ppo.value_num_layers,
        ).to(self.device)
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=self.oracle_ppo.value_lr
        )
        self.probability_provider: TerminalProbabilityProvider = (
            ExactToyDAGTerminalProbabilityProvider(self)
        )
        self._value_targets: dict[int, float] = {}
        self._current_update_step = 0
        self._reward_normalizer = float(sum(self.reward_by_terminal.values()))
        target = self.target_reward()
        self._target_probability_array = np.asarray(
            [target[state] for state in self.terminals], dtype=np.float64
        )

    def _on_update_start(self, update_step: int) -> None:
        super()._on_update_start(update_step)
        self._current_update_step = update_step

    def _collect_training_groups(
        self,
    ) -> tuple[list[list[Episode]], list[dict[str, float]]]:
        # This snapshot is computed before rollouts and remains fixed for every
        # PPO epoch, exactly like the old behavior-policy log probabilities.
        self.probability_provider.refresh()
        return super()._collect_training_groups()

    def _provider_distribution_metrics(self) -> dict[str, float]:
        behavior = np.asarray(
            [
                self.probability_provider.behavior_distribution[state]
                for state in self.terminals
            ],
            dtype=np.float64,
        )
        evaluation = np.asarray(
            [
                self.probability_provider.evaluation_distribution[state]
                for state in self.terminals
            ],
            dtype=np.float64,
        )
        metrics = _distribution_metrics(
            behavior,
            self._target_probability_array,
            prefix="oracle_behavior",
        )
        metrics.update(
            _distribution_metrics(
                evaluation,
                self._target_probability_array,
                prefix="oracle_policy",
            )
        )
        metrics["oracle_behavior_policy_tv"] = float(
            0.5 * np.abs(behavior - evaluation).sum()
        )
        return metrics

    def _group_advantages(self, episodes: list[Episode]) -> float:
        outcomes = [episode.terminal for episode in episodes]
        log_probabilities = self.probability_provider.log_probabilities(
            outcomes
        )
        targets, metrics = terminal_importance_targets(
            [episode.reward for episode in episodes],
            log_probabilities,
            reward_normalizer=self._reward_normalizer,
            objective=self.oracle_ppo.objective,
            max_log_weight=self.oracle_ppo.max_log_weight,
            eps=self.config.advantage_eps,
        )

        steps = [step for episode in episodes for step in episode.steps]
        observations = torch.as_tensor(
            np.stack([step.obs for step in steps]),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            values = self.value_network(observations).cpu().numpy()

        step_targets: list[float] = []
        for episode, target in zip(episodes, targets):
            for step in episode.steps:
                self._value_targets[id(step)] = float(target)
                step_targets.append(float(target))
        target_array = np.asarray(step_targets, dtype=np.float64)
        advantages = target_array - values.astype(np.float64)
        raw_advantages = advantages.copy()
        if self.oracle_ppo.normalize_advantages:
            advantage_std = float(advantages.std())
            advantages = advantages - advantages.mean()
            if advantage_std >= self.config.advantage_eps:
                advantages /= advantage_std + self.config.advantage_eps

        for step, advantage in zip(steps, advantages):
            step.advantage = float(advantage)

        target_variance = float(target_array.var())
        residual_variance = float((target_array - values).var())
        explained_variance = (
            1.0 - residual_variance / target_variance
            if target_variance > self.config.advantage_eps
            else 0.0
        )
        counts = Counter(outcomes)
        metrics.update(
            {
                "ips_unique_outcomes": float(len(counts)),
                "ips_max_outcome_count": float(max(counts.values())),
                "ips_min_outcome_count": float(min(counts.values())),
                "advantage_mean": float(advantages.mean()),
                "advantage_std": float(advantages.std()),
                "advantage_min": float(advantages.min()),
                "advantage_max": float(advantages.max()),
                "raw_advantage_mean": float(raw_advantages.mean()),
                "raw_advantage_std": float(raw_advantages.std()),
                "critic_value_mean": float(values.mean()),
                "critic_value_std": float(values.std()),
                "critic_preupdate_mse": float(
                    np.square(target_array - values).mean()
                ),
                "critic_preupdate_explained_variance": explained_variance,
                "exploration_epsilon": self.current_epsilon,
                "exploration_temperature": self.current_temperature,
                "oracle_objective_is_raw": float(
                    self.oracle_ppo.objective == "raw"
                ),
                "advantage_normalization_enabled": float(
                    self.oracle_ppo.normalize_advantages
                ),
            }
        )
        metrics.update(self._provider_distribution_metrics())
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]

    def _flat_step_tensors(
        self, episodes: list[Episode]
    ) -> tuple[
        list[StepRecord],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        steps = [step for episode in episodes for step in episode.steps]
        observations = torch.as_tensor(
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
        return (
            steps,
            observations,
            direction_masks,
            step_masks,
            directions,
            step_indices,
        )

    def _critic_loss(
        self, episodes: list[Episode]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        steps, observations, _, _, _, _ = self._flat_step_tensors(episodes)
        try:
            targets = torch.tensor(
                [self._value_targets[id(step)] for step in steps],
                dtype=torch.float32,
                device=self.device,
            )
        except KeyError as error:
            raise RuntimeError("a PPO step is missing its critic target") from error
        predictions = self.value_network(observations)
        loss = F.mse_loss(predictions, targets)
        with torch.no_grad():
            target_variance = targets.var(unbiased=False)
            residual_variance = (targets - predictions).var(unbiased=False)
            explained_variance = (
                1.0 - residual_variance / target_variance
                if target_variance > self.config.advantage_eps
                else predictions.new_zeros(())
            )
        return loss, {
            "value_loss": float(loss.item()),
            "critic_postupdate_explained_variance": float(
                explained_variance.item()
            ),
            "critic_prediction_mean": float(predictions.mean().item()),
            "critic_target_mean": float(targets.mean().item()),
        }

    def _joint_policy_loss(
        self, episodes: list[Episode]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        (
            steps,
            observations,
            direction_masks,
            step_masks,
            directions,
            step_indices,
        ) = self._flat_step_tensors(episodes)
        base_direction, representation = self.direction_policy.dist_with_rep(
            observations, direction_masks
        )
        direction_distribution = self._action_distribution(
            base_direction, direction_masks, explore=True
        )
        step_representation = (
            representation.detach()
            if self.config.detach_step_rep
            else representation
        )
        base_step = self.step_policy.dist(
            step_representation, directions, step_masks
        )
        step_distribution = self._action_distribution(
            base_step, step_masks, explore=True
        )
        new_joint_flat = (
            direction_distribution.log_prob(directions)
            + step_distribution.log_prob(step_indices)
        )
        old_joint_flat = torch.tensor(
            [step.log_prob_joint for step in steps],
            dtype=torch.float32,
            device=self.device,
        )
        entropy_flat = (
            direction_distribution.entropy() + step_distribution.entropy()
        )
        advantage_flat = torch.tensor(
            [step.advantage for step in steps],
            dtype=torch.float32,
            device=self.device,
        )

        lengths = [len(episode.steps) for episode in episodes]
        max_length = max(lengths)
        new_joint = _pad_episode_values(
            new_joint_flat, lengths, max_length
        )
        old_joint = _pad_episode_values(
            old_joint_flat, lengths, max_length
        )
        entropy = _pad_episode_values(entropy_flat, lengths, max_length)
        advantages = _pad_episode_values(
            advantage_flat, lengths, max_length
        )
        mask = (
            torch.arange(max_length, device=self.device).unsqueeze(0)
            < torch.tensor(lengths, device=self.device).unsqueeze(1)
        )
        numeric_mask = mask.to(dtype=new_joint.dtype)

        log_ratio = new_joint - old_joint.detach()
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio,
        )
        per_token = -torch.min(
            ratio * advantages, clipped_ratio * advantages
        )
        total_tokens = numeric_mask.sum().clamp(min=1.0)
        policy_loss = (per_token * numeric_mask).sum() / total_tokens
        mean_entropy = (entropy * numeric_mask).sum() / total_tokens
        loss = policy_loss - self.config.entropy_coef * mean_entropy
        with torch.no_grad():
            return loss, {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "entropy": float(mean_entropy.item()),
                "mean_importance_ratio": float(ratio[mask].mean().item()),
                "max_importance_ratio": float(ratio[mask].max().item()),
                "min_importance_ratio": float(ratio[mask].min().item()),
                "clip_fraction": float(
                    ((ratio != clipped_ratio) & mask).float().sum().item()
                    / mask.float().sum().item()
                ),
            }

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        policy_parameters = list(self.direction_policy.parameters()) + list(
            self.step_policy.parameters()
        )
        policy_totals: dict[str, float] = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "entropy": 0.0,
            "mean_importance_ratio": 0.0,
            "max_importance_ratio": 0.0,
            "min_importance_ratio": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "param_norm": 0.0,
        }
        for _ in range(self.config.train_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss, statistics = self._joint_policy_loss(episodes)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                policy_parameters, self.config.grad_clip_norm
            )
            self.optimizer.step()
            statistics["grad_norm"] = float(grad_norm.item())
            statistics["param_norm"] = float(
                sum(
                    parameter.detach().norm().item() ** 2
                    for parameter in policy_parameters
                )
                ** 0.5
            )
            for key in policy_totals:
                policy_totals[key] += statistics[key]

        value_totals = {
            "value_loss": 0.0,
            "critic_postupdate_explained_variance": 0.0,
            "critic_prediction_mean": 0.0,
            "critic_target_mean": 0.0,
            "critic_grad_norm": 0.0,
            "critic_param_norm": 0.0,
        }
        value_parameters = list(self.value_network.parameters())
        for _ in range(self.oracle_ppo.value_train_epochs):
            self.value_optimizer.zero_grad(set_to_none=True)
            value_loss, value_statistics = self._critic_loss(episodes)
            value_loss.backward()
            value_grad_norm = nn.utils.clip_grad_norm_(
                value_parameters, self.oracle_ppo.value_grad_clip_norm
            )
            self.value_optimizer.step()
            value_statistics["critic_grad_norm"] = float(
                value_grad_norm.item()
            )
            value_statistics["critic_param_norm"] = float(
                sum(
                    parameter.detach().norm().item() ** 2
                    for parameter in value_parameters
                )
                ** 0.5
            )
            for key in value_totals:
                value_totals[key] += value_statistics[key]

        self._value_targets.clear()
        output = {
            key: value / self.config.train_epochs
            for key, value in policy_totals.items()
        }
        output.update(
            {
                key: value / self.oracle_ppo.value_train_epochs
                for key, value in value_totals.items()
            }
        )
        return output

    def _update_training_groups(
        self, groups: list[list[Episode]]
    ) -> dict[str, float]:
        statistics = self.update(
            [episode for group in groups for episode in group]
        )
        if (
            self._current_update_step == 1
            or self._current_update_step % self.config.log_every == 0
        ):
            metrics = self._last_ips_metrics
            print(
                "  oracle "
                f"TV(q,target)={metrics['oracle_behavior_tv_reward_target']:.3f}  "
                f"TV(pi,target)={metrics['oracle_policy_tv_reward_target']:.3f}  "
                f"ESS={metrics['ips_ess_fraction']:.3f}  "
                f"w_max={metrics['importance_target_max']:.3f}  "
                f"w_clip={metrics['importance_weight_clip_fraction']:.3f}  "
                f"V_loss={statistics['value_loss']:.3f}  "
                "V_EV="
                f"{statistics['critic_postupdate_explained_variance']:.3f}"
            )
        return statistics

    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "direction_policy": self.direction_policy.state_dict(),
                "step_policy": self.step_policy.state_dict(),
                "value_network": self.value_network.state_dict(),
                "oracle_ppo_config": asdict(self.oracle_ppo),
                "exploration": asdict(self.exploration),
                "current_epsilon": self.current_epsilon,
                "current_temperature": self.current_temperature,
                "update_step": update_step,
                "algorithm": {
                    "name": "oracle_terminal_propensity_ppo",
                    "propensity": "exact_behavior_terminal_marginal",
                    "path_aggregation": "sum_over_all_paths_to_terminal",
                    "objective": self.oracle_ppo.objective,
                    "value_target": "terminal_importance_target",
                },
            },
            path,
        )
        return path


def _plot_oracle_training(
    history: list[dict[str, Any]], *, output: Path
) -> None:
    steps = [row["step"] for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    axes[0, 0].plot(
        steps,
        [row["oracle_behavior_tv_reward_target"] for row in history],
        label="behavior q",
    )
    axes[0, 0].plot(
        steps,
        [row["oracle_policy_tv_reward_target"] for row in history],
        label="base policy pi",
    )
    axes[0, 0].set_title("Exact terminal TV to reward target")
    axes[0, 0].legend()

    axes[0, 1].plot(
        steps,
        [row["ips_ess_fraction"] for row in history],
        label="weight ESS / batch",
    )
    axes[0, 1].plot(
        steps,
        [row["importance_weight_clip_fraction"] for row in history],
        label="weight clip fraction",
    )
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_title("Importance-weight stability")
    axes[0, 1].legend()

    axes[1, 0].plot(
        steps,
        [row["critic_preupdate_mse"] for row in history],
        label="pre-update MSE",
    )
    axes[1, 0].plot(
        steps,
        [row["value_loss"] for row in history],
        label="critic training loss",
    )
    axes[1, 0].set_yscale("symlog", linthresh=1e-4)
    axes[1, 0].set_title("Value-critic fit")
    axes[1, 0].legend()

    axes[1, 1].plot(
        steps,
        [row["oracle_behavior_entropy"] for row in history],
        label="behavior entropy",
    )
    axes[1, 1].plot(
        steps,
        [row["oracle_policy_entropy"] for row in history],
        label="base-policy entropy",
    )
    axes[1, 1].set_title("Exact terminal entropy")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Update")
        axis.grid(alpha=0.22)
    figure.suptitle(
        "Oracle terminal-propensity PPO and value-critic diagnostics"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=2_000)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--objective", choices=("raw", "log"), default="raw")
    parser.add_argument(
        "--max-log-weight",
        type=float,
        default=8.0,
        help=(
            "upper clip on log[R/(sum(R)*p_q)]; use a negative value to "
            "disable clipping"
        ),
    )
    parser.add_argument(
        "--no-normalize-advantages",
        action="store_true",
        help="use critic residuals without standardizing them in the batch",
    )
    parser.add_argument("--value-hidden-size", type=int, default=128)
    parser.add_argument("--value-num-layers", type=int, default=2)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--value-train-epochs", type=int, default=4)
    parser.add_argument("--value-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--temperature-start", type=float, default=2.0)
    parser.add_argument("--temperature-end", type=float, default=1.0)
    parser.add_argument("--anneal-updates", type=int, default=None)
    parser.add_argument(
        "--schedule", choices=("linear", "cosine"), default="cosine"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
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
    exploration = ExplorationConfig(
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        anneal_updates=args.anneal_updates,
        schedule=args.schedule,
    )
    oracle_ppo = OraclePPOConfig(
        objective=args.objective,
        max_log_weight=(
            None if args.max_log_weight < 0.0 else args.max_log_weight
        ),
        normalize_advantages=not args.no_normalize_advantages,
        value_hidden_size=args.value_hidden_size,
        value_num_layers=args.value_num_layers,
        value_lr=args.value_lr,
        value_train_epochs=args.value_train_epochs,
        value_grad_clip_norm=args.value_grad_clip_norm,
    )
    run_directory = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "oracle_terminal_propensity_ppo_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    trainer = OracleTerminalPropensityPPOTrainer(
        config,
        device=_resolve_device(args.device),
        exploration=exploration,
        oracle_ppo=oracle_ppo,
    )
    checkpoint_every = args.checkpoint_every or None

    print(f"Run directory: {run_directory}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Algorithm: exact behavior terminal marginal summed over all paths; "
        f"objective={oracle_ppo.objective}; separate state-value critic; "
        f"group_size={config.group_size}"
    )
    print(
        "Exploration: PPO and oracle both use the same annealed "
        "epsilon/temperature behavior distribution"
    )

    (run_directory / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "exploration": asdict(exploration),
                "oracle_ppo": asdict(oracle_ppo),
                "device": str(trainer.device),
                "algorithm": {
                    "name": "oracle_terminal_propensity_ppo",
                    "propensity": "exact_behavior_terminal_marginal",
                    "path_aggregation": "sum_over_all_paths_to_terminal",
                    "objective": oracle_ppo.objective,
                    "raw_weight_normalizer": "sum_of_terminal_rewards",
                    "value_target": "terminal_importance_target",
                    "ppo_advantage": "terminal_target_minus_state_value",
                },
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
        checkpoint_dir=run_directory / "checkpoints",
    )
    (run_directory / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    trainer.save(
        run_directory / "checkpoint.pt", update_step=config.num_updates
    )

    oracle_plot = run_directory / "oracle_critic_diagnostics.png"
    _plot_oracle_training(history, output=oracle_plot)
    exploration_plot = run_directory / "exploration_schedule.png"
    _plot_exploration(history, exploration_plot)

    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_directory / "sampling_counts.png",
        suptitle="Oracle terminal-propensity PPO vs ideal reward sampling",
    )
    trajectories = _plot_trajectory_diagnostics(
        history,
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_directory / "trajectory_sampling.png",
        subtitle=(
            "Terminal propensity sums over paths; paths are diagnostics only"
        ),
    )

    # Refresh after the final update so this distribution matches the saved
    # policy, rather than the behavior snapshot immediately before it.
    trainer.probability_provider.refresh()
    exact_final = trainer._provider_distribution_metrics()
    summary: dict[str, Any] = {
        "environment": trainer.environment_summary(),
        "algorithm": {
            "name": "oracle_terminal_propensity_ppo",
            "objective": oracle_ppo.objective,
            "propensity": "exact_behavior_terminal_marginal",
            "value_critic": True,
            "group_frequency_estimator": False,
        },
        "final_exact_distribution_metrics": exact_final,
        "final_exact_base_policy_probabilities": {
            state.signature: float(
                trainer.probability_provider.evaluation_distribution[state]
            )
            for state in trainer.terminals
        },
        "final_sampling": sampling,
        "trajectory_sampling": trajectories,
        "final_training_diagnostics": {
            key: history[-1][key]
            for key in (
                "ips_ess_fraction",
                "importance_weight_clip_fraction",
                "critic_preupdate_mse",
                "value_loss",
                "critic_postupdate_explained_variance",
                "oracle_behavior_tv_reward_target",
                "oracle_policy_tv_reward_target",
            )
        },
        "plots": {
            "oracle_critic_diagnostics": oracle_plot.name,
            "exploration_schedule": exploration_plot.name,
            "sampling_counts": "sampling_counts.png",
            "trajectory_sampling": "trajectory_sampling.png",
        },
    }
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Final sampled TV distance: {sampling['tv_reward_target']:.4f}")
    print(
        "Final exact base-policy TV distance: "
        f"{exact_final['oracle_policy_tv_reward_target']:.4f}"
    )
    print(f"Final R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Artifacts: {run_directory}")


if __name__ == "__main__":
    main()
