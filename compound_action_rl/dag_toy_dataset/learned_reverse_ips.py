"""Trajectory IPS with a learned, normalized reverse proposal.

For a forward trajectory ``tau`` ending at terminal ``x`` this trainer uses

    weight(tau) = R(x) * q_phi(tau | x) / P_F(tau).

``P_F`` is the exact frozen behavior-policy probability recorded at rollout
time.  ``q_phi`` is a terminal-conditioned policy over valid parent edges.
Because it is locally normalized and every reverse move decreases DAG depth,
its probability sums to one over all trajectories reaching each terminal.
Consequently the target trajectory mass also sums to ``R(x)`` at every
terminal, without enumerating paths or knowing their multiplicity.

The reverse policy starts uniform and is trained by maximum likelihood on
previous forward samples.  It is deliberately updated *after* the forward
policy update: the reverse proposal used to weight a batch is frozen before
that batch is observed.  As it approaches ``P_F(tau | x)``, the implicit
estimate ``P_F(tau) / q_phi(tau | x)`` approaches the terminal propensity
``P_F(x)`` and the trajectory weights become lower variance.
"""

from __future__ import annotations

import argparse
import json
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
from torch.distributions import Categorical  # noqa: E402

from config import TrainConfig
from count_ips import Episode
from dag_env import RIGHT, UP, State
from exact_probability_ips import _resolve_device
from run_count_ips import (
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)
from trajectory_ips import FullTrajectoryPPOTrainer

'''
normal count based IPS:
x -> terminal outcome

p_hat(x) = count(x) / G
scaled_i   = reward(x) / p_hat(x)
-------------------------
Exact propensity IPS:
tau -> trajectory taken
x -> terminal outcome
tau:x -> many to one mapping

weight(tau) = R(x) * q_phi(tau | x) / P_F(tau).
R(x)/P(x) 
'''

@dataclass(frozen=True)
class LearnedReverseConfig:
    hidden_size: int = 128
    num_layers: int = 2
    lr: float = 1e-3
    train_epochs: int = 4
    grad_clip_norm: float = 1.0

    def validate(self) -> None:
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("reverse hidden_size and num_layers must be >= 1")
        if self.lr <= 0:
            raise ValueError("reverse lr must be > 0")
        if self.train_epochs < 1:
            raise ValueError("reverse train_epochs must be >= 1")
        if self.grad_clip_norm <= 0:
            raise ValueError("reverse grad_clip_norm must be > 0")


@dataclass
class RunningLogWeightNormalizer:
    """Normalize IPS weights by a detached EMA RMS from previous batches.

    Both moments are stored in log space, so very large trajectory ratios do
    not overflow. The baseline and RMS used for a batch come from previous
    batches. Thus a batch containing a 100x error produces a larger correction
    than one containing a 2x error, up to the explicit advantage clip.
    """

    decay: float = 0.99
    advantage_clip: float = 10.0
    log_ratio_clip: float = 20.0
    log_first_moment: float | None = None
    log_second_moment: float | None = None
    updates: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("running scale decay must be in [0, 1)")
        if self.advantage_clip <= 0.0:
            raise ValueError("running advantage clip must be > 0")
        if self.log_ratio_clip <= 0.0:
            raise ValueError("running log-ratio clip must be > 0")

    @staticmethod
    def _log_mean_exp(values: np.ndarray) -> float:
        maximum = float(values.max())
        return maximum + float(np.log(np.exp(values - maximum).mean()))

    def normalize(
        self, log_weights: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        if log_weights.ndim != 1 or log_weights.size == 0:
            raise ValueError("log_weights must be a non-empty vector")
        if np.any(~np.isfinite(log_weights)):
            raise ValueError("log_weights must be finite")

        batch_log_first_moment = self._log_mean_exp(log_weights)
        batch_log_second_moment = self._log_mean_exp(2.0 * log_weights)
        bootstrap = (
            self.log_first_moment is None or self.log_second_moment is None
        )
        scale_log_first_moment = (
            batch_log_first_moment
            if bootstrap
            else float(self.log_first_moment)
        )
        scale_log_second_moment = (
            batch_log_second_moment
            if bootstrap
            else float(self.log_second_moment)
        )
        log_rms = 0.5 * scale_log_second_moment
        stabilized = np.exp(
            np.clip(
                log_weights - log_rms,
                -self.log_ratio_clip,
                self.log_ratio_clip,
            )
        )
        scaled_baseline = float(
            np.exp(
                np.clip(
                    scale_log_first_moment - log_rms,
                    -self.log_ratio_clip,
                    self.log_ratio_clip,
                )
            )
        )
        centered = stabilized - scaled_baseline
        clipped = np.clip(
            centered, -self.advantage_clip, self.advantage_clip
        )
        clip_fraction = float(np.mean(clipped != centered))

        if bootstrap:
            self.log_first_moment = batch_log_first_moment
            self.log_second_moment = batch_log_second_moment
        else:
            self.log_first_moment = float(
                np.logaddexp(
                    np.log(self.decay) + float(self.log_first_moment)
                    if self.decay > 0.0
                    else -np.inf,
                    np.log1p(-self.decay) + batch_log_first_moment,
                )
            )
            self.log_second_moment = float(
                np.logaddexp(
                    np.log(self.decay) + float(self.log_second_moment)
                    if self.decay > 0.0
                    else -np.inf,
                    np.log1p(-self.decay) + batch_log_second_moment,
                )
            )
        self.updates += 1
        return clipped, {
            "running_scale_bootstrap": float(bootstrap),
            "running_scale_updates": float(self.updates),
            "running_log_weight_rms": float(log_rms),
            "running_weight_rms": float(
                np.exp(np.clip(log_rms, -745.0, 700.0))
            ),
            "running_scaled_weight_baseline": scaled_baseline,
            "running_scaled_weight_mean": float(stabilized.mean()),
            "running_scaled_weight_std": float(stabilized.std()),
            "running_preclip_advantage_min": float(centered.min()),
            "running_preclip_advantage_max": float(centered.max()),
            "running_advantage_clip_fraction": clip_fraction,
        }

    def state_dict(self) -> dict[str, float | int | None]:
        return {
            "decay": self.decay,
            "advantage_clip": self.advantage_clip,
            "log_ratio_clip": self.log_ratio_clip,
            "log_first_moment": self.log_first_moment,
            "log_second_moment": self.log_second_moment,
            "updates": self.updates,
        }

    @classmethod
    def from_state_dict(
        cls, state: dict[str, float | int | None]
    ) -> "RunningLogWeightNormalizer":
        return cls(
            decay=float(state["decay"]),
            advantage_clip=float(state["advantage_clip"]),
            log_ratio_clip=float(state["log_ratio_clip"]),
            log_first_moment=(
                None
                if state.get("log_first_moment") is None
                else float(state["log_first_moment"])
            ),
            log_second_moment=(
                None
                if state["log_second_moment"] is None
                else float(state["log_second_moment"])
            ),
            updates=int(state["updates"]),
        )


@dataclass(frozen=True)
class ReverseBatch:
    contexts: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    episode_indices: torch.Tensor
    num_episodes: int


def reverse_context(
    child: State,
    terminal: State,
    *,
    budget: int,
) -> tuple[float, ...]:
    """Continuous child/terminal coordinates for a reverse decision."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if child.depth < 1 or child.depth > budget:
        raise ValueError("reverse child must be non-root and within the budget")
    if terminal.depth != budget:
        raise ValueError("terminal must lie on the terminal frontier")
    if child.x > terminal.x or child.y > terminal.y:
        raise ValueError("child must lie on a monotone path to terminal")
    scale = float(budget)
    return (
        child.x / scale,
        child.y / scale,
        terminal.x / scale,
        terminal.y / scale,
        (terminal.x - child.x) / scale,
        (terminal.y - child.y) / scale,
    )


def reverse_action_mask(child: State, *, max_step: int) -> tuple[bool, ...]:
    """Mask joint reverse actions ``(direction, incoming step length)``."""
    if max_step < 1:
        raise ValueError("max_step must be >= 1")
    if child.depth < 1:
        raise ValueError("the root has no reverse action")
    return tuple(
        length <= coordinate
        for coordinate in (child.x, child.y)
        for length in range(1, max_step + 1)
    )


def reverse_action_index(direction: int, length: int, *, max_step: int) -> int:
    if direction not in (RIGHT, UP):
        raise ValueError(f"unknown direction {direction}")
    if length < 1 or length > max_step:
        raise ValueError(f"length must be in [1, {max_step}]")
    return direction * max_step + length - 1


class LearnedReversePolicy(nn.Module):
    """Terminal-conditioned categorical policy over valid parent edges."""

    def __init__(self, max_step: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        if max_step < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("reverse network dimensions must be >= 1")
        layers: list[nn.Module] = []
        width = 6
        for _ in range(num_layers):
            layers.extend((nn.Linear(width, hidden_size), nn.Tanh()))
            width = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, 2 * max_step)
        # The initial policy is exactly uniform over the valid parents.  This
        # is the fixed reverse reference until data teaches a better proposal.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def dist(self, contexts: torch.Tensor, masks: torch.Tensor) -> Categorical:
        if contexts.ndim != 2 or contexts.shape[1] != 6:
            raise ValueError("reverse contexts must have shape (edges, 6)")
        if masks.ndim != 2 or masks.shape[0] != contexts.shape[0]:
            raise ValueError("reverse masks must have shape (edges, actions)")
        if torch.any(~masks.bool().any(dim=-1)):
            raise ValueError("every reverse state must have a valid parent")
        logits = self.head(self.trunk(contexts))
        logits = torch.where(
            masks.bool(), logits, torch.full_like(logits, -1e9)
        )
        return Categorical(logits=logits)


def learned_reverse_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    trajectory_ids: Sequence[object],
    forward_log_probabilities: Sequence[float],
    reverse_log_probabilities: Sequence[float],
    *,
    running_normalizer: RunningLogWeightNormalizer | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize ``R * q_phi(tau|x) / P_F(tau)`` in stable log space."""
    size = len(rewards)
    if (
        size == 0
        or len(outcome_ids) != size
        or len(trajectory_ids) != size
        or len(forward_log_probabilities) != size
        or len(reverse_log_probabilities) != size
    ):
        raise ValueError("all learned-reverse IPS inputs must have equal size")

    reward_array = np.asarray(rewards, dtype=np.float64)
    log_p_f = np.asarray(forward_log_probabilities, dtype=np.float64)
    log_q = np.asarray(reverse_log_probabilities, dtype=np.float64)
    if np.any(~np.isfinite(reward_array)) or np.any(reward_array <= 0.0):
        raise ValueError("rewards must be finite and strictly positive")
    if np.any(~np.isfinite(log_p_f)) or np.any(log_p_f > 1e-7):
        raise ValueError("forward log-probabilities must be finite and <= 0")
    if np.any(~np.isfinite(log_q)) or np.any(log_q > 1e-7):
        raise ValueError("reverse log-probabilities must be finite and <= 0")

    implied_terminal_log_probability = log_p_f - log_q
    log_weights = np.log(reward_array) - implied_terminal_log_probability
    # Batch z-scoring is invariant to a common positive multiplier, and ESS
    # depends only on relative weights. This max-shift keeps both calculations
    # stable; running normalization separately uses its detached EMA scale.
    log_shift = float(log_weights.max())
    scaled = np.exp(log_weights - log_shift)
    std = float(scaled.std())
    normalization_metrics: dict[str, float] = {}
    if running_normalizer is None:
        centered = scaled - scaled.mean()
        advantages = centered if std < eps else centered / (std + eps)
    else:
        advantages, normalization_metrics = running_normalizer.normalize(
            log_weights
        )

    squared_sum = float(np.square(scaled).sum())
    ess = float(scaled.sum() ** 2 / max(squared_sum, eps))
    outcome_counts = Counter(outcome_ids)
    trajectory_counts = Counter(trajectory_ids)
    implied_probability = np.exp(
        np.clip(implied_terminal_log_probability, -745.0, 700.0)
    )
    within_outcome_stds = [
        float(np.std(implied_terminal_log_probability[
            np.asarray([item == outcome for item in outcome_ids], dtype=bool)
        ]))
        for outcome, count in outcome_counts.items()
        if count > 1
    ]
    metrics = {
        "ips_prob_mean": float(implied_probability.mean()),
        "ips_prob_min": float(implied_probability.min()),
        "ips_prob_max": float(implied_probability.max()),
        "ips_unique_outcomes": float(len(outcome_counts)),
        "ips_max_outcome_count": float(max(outcome_counts.values())),
        "ips_min_outcome_count": float(min(outcome_counts.values())),
        # These are max-shifted weights. Their relative values, normalized
        # advantages, and ESS are identical to those of the raw IPS weights.
        "ips_scaled_reward_mean": float(scaled.mean()),
        "ips_scaled_reward_std": std,
        "ips_ess": ess,
        "ips_ess_fraction": ess / size,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        "ips_unique_trajectories": float(len(trajectory_counts)),
        "forward_log_probability_mean": float(log_p_f.mean()),
        "reverse_log_probability_mean": float(log_q.mean()),
        "implied_terminal_log_probability_mean": float(
            implied_terminal_log_probability.mean()
        ),
        "implied_terminal_log_probability_std": float(
            implied_terminal_log_probability.std()
        ),
        "implied_terminal_within_outcome_std": float(
            np.mean(within_outcome_stds) if within_outcome_stds else 0.0
        ),
        "log_importance_weight_mean": float(log_weights.mean()),
        "log_importance_weight_min": float(log_weights.min()),
        "log_importance_weight_max": float(log_weights.max()),
        "log_weight_shift": log_shift,
        "advantage_normalization_is_running": float(
            running_normalizer is not None
        ),
    }
    metrics.update(normalization_metrics)
    return advantages, metrics


class LearnedReverseIPSTrainer(FullTrajectoryPPOTrainer):
    """Full-path PPO using a learned normalized reverse IPS proposal."""

    probability_label = "p_tau/q"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        reverse_config: LearnedReverseConfig | None = None,
        forward_lr_decay_after: int | None = None,
        forward_lr_after_decay: float | None = None,
        advantage_normalization: str = "batch",
        running_scale_decay: float = 0.99,
        running_advantage_clip: float = 10.0,
        running_log_ratio_clip: float = 20.0,
    ) -> None:
        super().__init__(config, device=device)
        self.reverse_config = reverse_config or LearnedReverseConfig(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
        )
        self.reverse_config.validate()
        self.reverse_policy = LearnedReversePolicy(
            self.config.max_step,
            self.reverse_config.hidden_size,
            self.reverse_config.num_layers,
        ).to(self.device)
        self.reverse_optimizer = torch.optim.Adam(
            self.reverse_policy.parameters(), lr=self.reverse_config.lr
        )
        if (forward_lr_decay_after is None) != (forward_lr_after_decay is None):
            raise ValueError(
                "forward_lr_decay_after and forward_lr_after_decay must be set together"
            )
        if forward_lr_decay_after is not None and forward_lr_decay_after < 1:
            raise ValueError("forward_lr_decay_after must be >= 1")
        if forward_lr_after_decay is not None and forward_lr_after_decay <= 0:
            raise ValueError("forward_lr_after_decay must be > 0")
        self.forward_lr_decay_after = forward_lr_decay_after
        self.forward_lr_after_decay = forward_lr_after_decay
        self.current_forward_lr = float(self.config.lr)
        if advantage_normalization not in ("batch", "running"):
            raise ValueError(
                "advantage_normalization must be 'batch' or 'running'"
            )
        self.advantage_normalization = advantage_normalization
        self.running_weight_normalizer = (
            RunningLogWeightNormalizer(
                decay=running_scale_decay,
                advantage_clip=running_advantage_clip,
                log_ratio_clip=running_log_ratio_clip,
            )
            if advantage_normalization == "running"
            else None
        )

    def _on_update_start(self, update_step: int) -> None:
        super()._on_update_start(update_step)
        learning_rate = float(self.config.lr)
        if (
            self.forward_lr_decay_after is not None
            and update_step > self.forward_lr_decay_after
        ):
            assert self.forward_lr_after_decay is not None
            learning_rate = float(self.forward_lr_after_decay)
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        self.current_forward_lr = learning_rate

    def _reverse_batch(self, episodes: list[Episode]) -> ReverseBatch:
        contexts: list[tuple[float, ...]] = []
        masks: list[tuple[bool, ...]] = []
        actions: list[int] = []
        episode_indices: list[int] = []
        for episode_index, episode in enumerate(episodes):
            child = State(0, 0)
            for direction, length in episode.trajectory:
                child = (
                    State(child.x + length, child.y)
                    if direction == RIGHT
                    else State(child.x, child.y + length)
                )
                contexts.append(
                    reverse_context(child, episode.terminal, budget=self.config.budget)
                )
                masks.append(
                    reverse_action_mask(child, max_step=self.config.max_step)
                )
                actions.append(
                    reverse_action_index(
                        direction, length, max_step=self.config.max_step
                    )
                )
                episode_indices.append(episode_index)
            if child != episode.terminal:
                raise ValueError("episode trajectory does not reach its terminal")
        if not contexts:
            raise ValueError("episodes must contain at least one transition")
        return ReverseBatch(
            contexts=torch.tensor(
                contexts, dtype=torch.float32, device=self.device
            ),
            masks=torch.tensor(masks, dtype=torch.bool, device=self.device),
            actions=torch.tensor(actions, dtype=torch.long, device=self.device),
            episode_indices=torch.tensor(
                episode_indices, dtype=torch.long, device=self.device
            ),
            num_episodes=len(episodes),
        )

    def _reverse_path_log_probabilities_tensor(
        self, batch: ReverseBatch
    ) -> tuple[torch.Tensor, Categorical]:
        distribution = self.reverse_policy.dist(batch.contexts, batch.masks)
        edge_log_probabilities = distribution.log_prob(batch.actions)
        path_log_probabilities = torch.zeros(
            batch.num_episodes, dtype=torch.float32, device=self.device
        )
        path_log_probabilities.scatter_add_(
            0, batch.episode_indices, edge_log_probabilities
        )
        return path_log_probabilities, distribution

    @torch.inference_mode()
    def reverse_path_log_probabilities(
        self, episodes: list[Episode]
    ) -> np.ndarray:
        batch = self._reverse_batch(episodes)
        path_log_probabilities, _ = self._reverse_path_log_probabilities_tensor(batch)
        return path_log_probabilities.cpu().numpy().astype(np.float64)

    def _group_advantages(self, episodes: list[Episode]) -> float:
        forward_log_probabilities = [
            sum(step.log_prob_joint for step in episode.steps)
            for episode in episodes
        ]
        reverse_log_probabilities = self.reverse_path_log_probabilities(episodes)
        advantages, metrics = learned_reverse_ips_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            [episode.trajectory for episode in episodes],
            forward_log_probabilities,
            reverse_log_probabilities,
            running_normalizer=self.running_weight_normalizer,
            eps=self.config.advantage_eps,
        )
        metrics["forward_lr"] = self.current_forward_lr
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]

    def evaluate(
        self, episodes: int = 10_000, *, batch_size: int | None = None
    ) -> dict[str, Any]:
        evaluation = super().evaluate(episodes, batch_size=batch_size)
        boundary_states = _boundary_states(self.config.budget)
        target = self.target_reward()
        evaluation["eval_boundary_outcome_probs"] = {
            state.signature: evaluation["eval_outcome_probs"][state.signature]
            for state in boundary_states
        }
        evaluation["target_boundary_outcome_probs"] = {
            state.signature: target[state] for state in boundary_states
        }
        return evaluation

    def _update_reverse_policy(self, episodes: list[Episode]) -> dict[str, float]:
        batch = self._reverse_batch(episodes)
        grad_norm_total = 0.0
        for _ in range(self.reverse_config.train_epochs):
            self.reverse_optimizer.zero_grad(set_to_none=True)
            path_log_probabilities, _ = (
                self._reverse_path_log_probabilities_tensor(batch)
            )
            loss = -path_log_probabilities.mean()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.reverse_policy.parameters(),
                self.reverse_config.grad_clip_norm,
            )
            self.reverse_optimizer.step()
            grad_norm_total += float(grad_norm.item())

        with torch.inference_mode():
            path_log_probabilities, distribution = (
                self._reverse_path_log_probabilities_tensor(batch)
            )
            predictions = distribution.logits.argmax(dim=-1)
            parameter_norm = sum(
                parameter.detach().norm().item() ** 2
                for parameter in self.reverse_policy.parameters()
            ) ** 0.5
            return {
                "reverse_loss": float(-path_log_probabilities.mean().item()),
                "reverse_edge_accuracy": float(
                    (predictions == batch.actions).float().mean().item()
                ),
                "reverse_edge_entropy": float(
                    distribution.entropy().mean().item()
                ),
                "reverse_grad_norm": (
                    grad_norm_total / self.reverse_config.train_epochs
                ),
                "reverse_param_norm": float(parameter_norm),
            }

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        # Advantages were computed with the reverse policy frozen before this
        # batch. Update the forward policy first, then fit q for the next batch.
        forward_metrics = super().update(episodes)
        reverse_metrics = self._update_reverse_policy(episodes)
        return {**forward_metrics, **reverse_metrics}

    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "direction_policy": self.direction_policy.state_dict(),
                "step_policy": self.step_policy.state_dict(),
                "reverse_policy": self.reverse_policy.state_dict(),
                # Store plain data instead of pickling a script-local dataclass.
                # This keeps checkpoints loadable when this file was launched
                # as ``python learned_reverse_ips.py`` and therefore __main__.
                "reverse_config": asdict(self.reverse_config),
                "running_weight_normalizer": (
                    self.running_weight_normalizer.state_dict()
                    if self.running_weight_normalizer is not None
                    else None
                ),
                "update_step": update_step,
                "algorithm": {
                    "name": "learned_reverse_ips",
                    "raw_weight": "R(x) * q_phi(tau|x) / P_F(tau)",
                    "forward_loss": "full_trajectory_ppo",
                    "reverse_update_order": "after_forward_update",
                    "forward_lr_decay_after": self.forward_lr_decay_after,
                    "forward_lr_after_decay": self.forward_lr_after_decay,
                    "advantage_normalization": self.advantage_normalization,
                },
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "LearnedReverseIPSTrainer":
        # Older checkpoints pickled LearnedReverseConfig as __main__ when the
        # runner was invoked as a script. Supplying that symbol lets those
        # existing files load; newly saved checkpoints use a plain dictionary.
        import __main__

        missing = object()
        previous = getattr(__main__, "LearnedReverseConfig", missing)
        setattr(__main__, "LearnedReverseConfig", LearnedReverseConfig)
        try:
            payload = torch.load(Path(path), map_location=device, weights_only=False)
        finally:
            if previous is missing:
                delattr(__main__, "LearnedReverseConfig")
            else:
                setattr(__main__, "LearnedReverseConfig", previous)
        if payload.get("algorithm", {}).get("name") != "learned_reverse_ips":
            raise ValueError("checkpoint is not a learned-reverse IPS run")
        reverse_config_payload = payload["reverse_config"]
        reverse_config = (
            LearnedReverseConfig(**reverse_config_payload)
            if isinstance(reverse_config_payload, dict)
            else reverse_config_payload
        )
        algorithm = payload.get("algorithm", {})
        running_state = payload.get("running_weight_normalizer")
        trainer = cls(
            payload["config"],
            device=device,
            reverse_config=reverse_config,
            forward_lr_decay_after=algorithm.get("forward_lr_decay_after"),
            forward_lr_after_decay=algorithm.get("forward_lr_after_decay"),
            advantage_normalization=algorithm.get(
                "advantage_normalization", "batch"
            ),
            running_scale_decay=(
                float(running_state["decay"])
                if running_state is not None
                else 0.99
            ),
            running_advantage_clip=(
                float(running_state["advantage_clip"])
                if running_state is not None
                else 10.0
            ),
            running_log_ratio_clip=(
                float(running_state["log_ratio_clip"])
                if running_state is not None
                else 20.0
            ),
        )
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.reverse_policy.load_state_dict(payload["reverse_policy"])
        if running_state is not None:
            trainer.running_weight_normalizer = (
                RunningLogWeightNormalizer.from_state_dict(running_state)
            )
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        trainer.reverse_policy.eval()
        return trainer


def _boundary_states(budget: int) -> tuple[State, ...]:
    """The two terminal states at each edge of the frontier."""
    x_coordinates = dict.fromkeys((0, min(1, budget), max(0, budget - 1), budget))
    return tuple(State(x, budget - x) for x in x_coordinates)


def _plot_reverse_training(history: list[dict], *, output: Path) -> None:
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    axes[0, 0].plot(steps, [row["reverse_loss"] for row in history])
    axes[0, 0].set_title("Reverse trajectory NLL")
    axes[0, 0].set_ylabel("NLL")

    axes[0, 1].plot(
        steps,
        [row["reverse_edge_accuracy"] for row in history],
        label="edge accuracy",
    )
    axes[0, 1].plot(
        steps,
        [row["ips_ess_fraction"] for row in history],
        label="IPS ESS / batch",
    )
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_title("Reverse fit and IPS stability")
    axes[0, 1].legend()

    axes[1, 0].plot(
        steps,
        [row["forward_log_probability_mean"] for row in history],
        label="log P_F(tau)",
    )
    axes[1, 0].plot(
        steps,
        [row["reverse_log_probability_mean"] for row in history],
        label="log q(tau|x)",
    )
    axes[1, 0].set_title("Forward and reverse path probabilities")
    axes[1, 0].legend()

    axes[1, 1].plot(
        steps,
        [row["implied_terminal_within_outcome_std"] for row in history],
        label="within-outcome std",
    )
    axes[1, 1].plot(
        steps,
        [row["implied_terminal_log_probability_std"] for row in history],
        label="overall std",
        alpha=0.75,
    )
    axes[1, 1].set_title("Dispersion of log[P_F(tau) / q(tau|x)]")
    axes[1, 1].set_ylabel("Log-probability standard deviation")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Update")
        axis.grid(alpha=0.22)
    fig.suptitle("Learned reverse-proposal diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_boundary_training(
    history: list[dict],
    trainer: LearnedReverseIPSTrainer,
    *,
    output: Path,
) -> None:
    """Track the endpoint modes and their immediately adjacent outcomes."""
    eval_rows = [row for row in history if "eval_boundary_outcome_probs" in row]
    states = _boundary_states(trainer.config.budget)
    target = trainer.target_reward()
    colors = plt.colormaps["viridis"](
        np.linspace(0.08, 0.92, len(states))
    )
    fig, (probability_axis, ratio_axis) = plt.subplots(1, 2, figsize=(13, 5))
    if eval_rows:
        steps = [row["step"] for row in eval_rows]
        for state, color in zip(states, colors):
            probabilities = [
                row["eval_boundary_outcome_probs"][state.signature]
                for row in eval_rows
            ]
            probability_axis.plot(
                steps, probabilities, "o-", color=color, label=state.signature
            )
            probability_axis.axhline(
                target[state], color=color, linestyle="--", alpha=0.55
            )
            ratio_axis.plot(
                steps,
                [probability / target[state] for probability in probabilities],
                "o-",
                color=color,
                label=state.signature,
            )
    probability_axis.set_title("Boundary probabilities")
    probability_axis.set_ylabel("P(x); dashed = reward target")
    probability_axis.set_ylim(bottom=0.0)
    probability_axis.legend()
    ratio_axis.axhline(1.0, color="#2d3436", linestyle="--", label="ideal ratio")
    ratio_axis.set_title("Boundary actual / ideal ratio")
    ratio_axis.set_ylabel("P(x) / P*(x)")
    ratio_axis.set_ylim(bottom=0.0)
    ratio_axis.legend()
    for axis in (probability_axis, ratio_axis):
        axis.set_xlabel("Update")
        axis.grid(alpha=0.22)
    fig.suptitle("Extreme-terminal diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_running_normalization(history: list[dict], *, output: Path) -> None:
    """Show the scale and clipping behavior of running IPS normalization."""
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    axes[0, 0].plot(
        steps, [row["running_log_weight_rms"] for row in history]
    )
    axes[0, 0].set_title("EMA log RMS of raw IPS weights")

    axes[0, 1].plot(
        steps,
        [row["running_advantage_clip_fraction"] for row in history],
    )
    axes[0, 1].set_ylim(-0.01, 1.01)
    axes[0, 1].set_title("Fraction explicitly clipped")

    axes[1, 0].plot(
        steps,
        [row["running_preclip_advantage_max"] for row in history],
        label="maximum",
    )
    axes[1, 0].plot(
        steps,
        [row["running_preclip_advantage_min"] for row in history],
        label="minimum",
    )
    axes[1, 0].set_title("Advantages before explicit clipping")
    axes[1, 0].legend()

    axes[1, 1].plot(
        steps,
        [row["advantage_std"] for row in history],
        label="advantage std",
    )
    axes[1, 1].plot(
        steps,
        [row["ips_ess_fraction"] for row in history],
        label="IPS ESS / batch",
    )
    axes[1, 1].set_title("Update magnitude and weight stability")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Update")
        axis.grid(alpha=0.22)
    fig.suptitle("Running IPS normalization diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=2_000)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-decay-after",
        type=int,
        default=None,
        help="keep --lr through this update, then switch to --lr-after-decay",
    )
    parser.add_argument("--lr-after-decay", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--reverse-lr", type=float, default=1e-3)
    parser.add_argument("--reverse-hidden-size", type=int, default=128)
    parser.add_argument("--reverse-num-layers", type=int, default=2)
    parser.add_argument("--reverse-train-epochs", type=int, default=4)
    parser.add_argument("--reverse-grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--advantage-normalization",
        choices=("batch", "running"),
        default="batch",
    )
    parser.add_argument("--running-scale-decay", type=float, default=0.99)
    parser.add_argument("--running-advantage-clip", type=float, default=10.0)
    parser.add_argument("--running-log-ratio-clip", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--terminal-rewards", type=float, nargs="+", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        budget=args.budget,
        max_step=args.max_step,
        terminal_rewards=(
            tuple(args.terminal_rewards) if args.terminal_rewards is not None else None
        ),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        group_size=args.group_size,
        num_groups=args.num_groups,
        num_updates=args.num_updates,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        log_every=args.log_every,
    )
    reverse_config = LearnedReverseConfig(
        hidden_size=args.reverse_hidden_size,
        num_layers=args.reverse_num_layers,
        lr=args.reverse_lr,
        train_epochs=args.reverse_train_epochs,
        grad_clip_norm=args.reverse_grad_clip_norm,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "learned_reverse_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = LearnedReverseIPSTrainer(
        config,
        device=_resolve_device(args.device),
        reverse_config=reverse_config,
        forward_lr_decay_after=args.lr_decay_after,
        forward_lr_after_decay=args.lr_after_decay,
        advantage_normalization=args.advantage_normalization,
        running_scale_decay=args.running_scale_decay,
        running_advantage_clip=args.running_advantage_clip,
        running_log_ratio_clip=args.running_log_ratio_clip,
    )
    checkpoint_every = args.checkpoint_every or None
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Algorithm: R(x) * q_phi(tau|x) / P_F(tau); "
        "terminal-conditioned q is learned after each forward update; "
        f"advantage_normalization={args.advantage_normalization}"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "reverse_config": asdict(reverse_config),
                "device": str(trainer.device),
                "algorithm": "learned_reverse_ips",
                "raw_weight": "R(x) * q_phi(tau|x) / P_F(tau)",
                "implicit_terminal_propensity": "P_F(tau) / q_phi(tau|x)",
                "forward_loss": "full_trajectory_ppo",
                "reverse_update_order": "after_forward_update",
                "forward_lr_decay_after": args.lr_decay_after,
                "forward_lr_after_decay": args.lr_after_decay,
                "advantage_normalization": args.advantage_normalization,
                "running_normalization": {
                    "scale_decay": args.running_scale_decay,
                    "advantage_clip": args.running_advantage_clip,
                    "log_ratio_clip": args.running_log_ratio_clip,
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
        propensity_title="Implicit terminal propensity P_F(tau) / q_phi(tau|x)",
        suptitle="Learned reverse-proposal IPS training",
    )
    reverse_plot = run_dir / "reverse_diagnostics.png"
    _plot_reverse_training(history, output=reverse_plot)
    boundary_plot = run_dir / "boundary_diagnostics.png"
    _plot_boundary_training(history, trainer, output=boundary_plot)
    running_plot = None
    if args.advantage_normalization == "running":
        running_plot = run_dir / "running_normalization_diagnostics.png"
        _plot_running_normalization(history, output=running_plot)
    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Learned reverse-proposal IPS vs ideal reward sampling",
    )
    trajectory_plot = run_dir / "trajectory_sampling.png"
    trajectories = _plot_trajectory_diagnostics(
        history,
        trainer,
        evaluation,
        samples=args.final_samples,
        output=trajectory_plot,
        subtitle="Conditional paths are represented by the learned reverse policy",
    )
    target = trainer.target_reward()
    outcome_index = {
        signature: index
        for index, signature in enumerate(sampling["outcomes"])
    }
    boundary_sampling = {}
    for state in _boundary_states(config.budget):
        index = outcome_index[state.signature]
        ideal_probability = float(sampling["ideal_probabilities"][index])
        actual_probability = float(sampling["actual_probabilities"][index])
        boundary_sampling[state.signature] = {
            "reward": trainer.reward_by_terminal[state],
            "target_probability": target[state],
            "ideal_probability": ideal_probability,
            "actual_probability": actual_probability,
            "actual_to_ideal_ratio": actual_probability / ideal_probability,
        }
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary = {
        "environment": trainer.environment_summary(),
        "algorithm": {
            "name": "learned_reverse_ips",
            "raw_weight": "R(x) * q_phi(tau|x) / P_F(tau)",
            "implicit_terminal_propensity": "P_F(tau) / q_phi(tau|x)",
            "forward_loss": "full_trajectory_ppo",
            "reverse_update_order": "after_forward_update",
        },
        "reverse_config": asdict(reverse_config),
        "forward_lr_schedule": {
            "initial_lr": config.lr,
            "decay_after_update": args.lr_decay_after,
            "lr_after_decay": args.lr_after_decay,
        },
        "advantage_normalization": {
            "mode": args.advantage_normalization,
            "running_scale_decay": args.running_scale_decay,
            "running_advantage_clip": args.running_advantage_clip,
            "running_log_ratio_clip": args.running_log_ratio_clip,
        },
        "final_sampling": sampling,
        "boundary_sampling": boundary_sampling,
        "trajectory_sampling": trajectories,
        "best_eval": (
            min(eval_rows, key=lambda row: row["tv_reward_target"])
            if eval_rows
            else None
        ),
        "final_training_diagnostics": {
            key: history[-1][key]
            for key in (
                "ips_ess_fraction",
                "reverse_loss",
                "reverse_edge_accuracy",
                "reverse_edge_entropy",
                "implied_terminal_within_outcome_std",
            )
        },
        "plots": {
            "training_curves": training_plot.name,
            "reverse_diagnostics": reverse_plot.name,
            "boundary_diagnostics": boundary_plot.name,
            "sampling_counts": "sampling_counts.png",
            "trajectory_sampling": trajectory_plot.name,
        },
    }
    if running_plot is not None:
        summary["plots"]["running_normalization_diagnostics"] = (
            running_plot.name
        )
        summary["final_training_diagnostics"].update(
            {
                key: history[-1][key]
                for key in (
                    "running_log_weight_rms",
                    "running_advantage_clip_fraction",
                    "running_preclip_advantage_min",
                    "running_preclip_advantage_max",
                )
            }
        )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Final ideal-line R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(
        "Final reverse diagnostics: "
        f"loss={history[-1]['reverse_loss']:.3f}, "
        f"edge_accuracy={history[-1]['reverse_edge_accuracy']:.3f}, "
        f"IPS_ESS={history[-1]['ips_ess_fraction']:.3f}"
    )
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
