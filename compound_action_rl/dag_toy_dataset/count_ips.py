"""Simple count-based IPS RL for the direction/step DAG.

For each independently sampled group of G episodes:

    p_hat(x_i) = count(x_i) / G
    scaled_i   = reward(x_i) / p_hat(x_i)
    advantage  = normalize(scaled)

The advantages feed the same token-level PPO clipped surrogate used by
``grpo_experiments/core/loss.py``. A token here is one compound DAG action and
its joint log-probability is log pi(direction) + log pi(step | direction).
There is no backward policy and no exact trajectory propensity correction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import TrainConfig
from dag_env import (
    DAGEnv,
    RIGHT,
    UP,
    RewardModel,
    State,
    Trajectory,
    find_default_terminal_states,
    reward_per_terminal_state,
    trajectory_signature,
)
from networks import DirectionPolicy, StepPolicy


@dataclass
class StepRecord:
    obs: np.ndarray
    direction_mask: np.ndarray
    step_mask: np.ndarray
    direction: int
    step_index: int
    log_prob_direction: float
    log_prob_step: float
    advantage: float = 0.0

    @property
    def log_prob_joint(self) -> float:
        return self.log_prob_direction + self.log_prob_step


@dataclass
class Episode:
    steps: list[StepRecord] = field(default_factory=list)
    terminal: State = State(0, 0)
    signature: str = ""
    trajectory: Trajectory = ()
    reward: float = 0.0


def _r2_against(target: np.ndarray, observed: np.ndarray) -> float:
    slope, intercept = np.polyfit(target, observed, 1)
    prediction = slope * target + intercept
    residual = float(np.sum((observed - prediction) ** 2))
    total = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - residual / total if total else 1.0


def count_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute normalized ``reward / batch outcome frequency`` advantages."""
    if len(rewards) == 0 or len(rewards) != len(outcome_ids):
        raise ValueError("rewards and outcome_ids must have the same non-zero length")
    reward_array = np.asarray(rewards, dtype=np.float64)
    counts = Counter(outcome_ids)
    group_size = len(outcome_ids)
    p_hat = np.array([counts[outcome] / group_size for outcome in outcome_ids])
    scaled = reward_array / p_hat
    centered = scaled - scaled.mean()
    std = float(scaled.std())
    advantages = centered if std < eps else centered / (std + eps)
    inverse = 1.0 / p_hat
    ess = float(inverse.sum() ** 2 / np.square(inverse).sum())
    metrics = {
        "ips_prob_mean": float(p_hat.mean()),
        "ips_prob_min": float(p_hat.min()),
        "ips_prob_max": float(p_hat.max()),
        "ips_unique_outcomes": float(len(counts)),
        "ips_max_outcome_count": float(max(counts.values())),
        "ips_min_outcome_count": float(min(counts.values())),
        "ips_scaled_reward_mean": float(scaled.mean()),
        "ips_scaled_reward_std": std,
        "ips_ess": ess,
        "ips_ess_fraction": ess / group_size,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
    }
    return advantages, metrics


def ppo_token_loss(
    log_paths_pf: torch.Tensor,
    advantages: torch.Tensor,
    *,
    log_paths_pf_old: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Minimal local equivalent of ``core/loss.py::compute_grpo_policy_loss``."""
    if log_paths_pf.ndim != 2 or log_paths_pf_old.shape != log_paths_pf.shape:
        raise ValueError("new and old log_paths_pf must have shape (batch, time)")
    if advantages.shape != (log_paths_pf.shape[0],):
        raise ValueError("advantages must have shape (batch,)")
    mask = mask.to(dtype=log_paths_pf.dtype)
    log_ratio = log_paths_pf - log_paths_pf_old.detach()
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    advantage = advantages.unsqueeze(1)
    per_token = -torch.min(ratio * advantage, clipped_ratio * advantage)
    token_counts = mask.sum(dim=-1).clamp(min=1.0)
    loss = ((per_token * mask).sum(dim=-1) / token_counts).mean()
    with torch.no_grad():
        valid = mask.bool()
        metrics = {
            "mean_importance_ratio": float(ratio[valid].mean().item()),
            "max_importance_ratio": float(ratio[valid].max().item()),
            "min_importance_ratio": float(ratio[valid].min().item()),
            "clip_fraction": float(((ratio != clipped_ratio) & valid).float().sum().item() / valid.float().sum().item()),
        }
    return loss, metrics


def _pad_episode_values(
    flat: torch.Tensor, lengths: list[int], max_length: int
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    offset = 0
    for length in lengths:
        values = flat[offset : offset + length]
        offset += length
        if length < max_length:
            values = torch.cat((values, values.new_zeros(max_length - length)))
        rows.append(values)
    return torch.stack(rows)


class CountIPSTrainer:
    """Count-IPS advantages plus joint token-level PPO policy updates."""

    probability_label = "p_hat"

    def __init__(self, config: TrainConfig | None = None, *, device: str = "cpu") -> None:
        self.config = config or TrainConfig()
        self.config.validate()
        self.device = torch.device(device)
        torch.manual_seed(self.config.seed)
        self.reward_model = RewardModel(
            budget=self.config.budget, values=self.config.terminal_rewards
        )
        self.env = DAGEnv(
            budget=self.config.budget,
            max_step=self.config.max_step,
            reward_model=self.reward_model,
        )
        self.terminals = list(find_default_terminal_states(self.config.budget))
        self.reward_by_terminal = reward_per_terminal_state(
            self.config.budget, self.config.terminal_rewards
        )
        self.direction_policy = DirectionPolicy(
            self.env.obs_dim, self.config.hidden_size, self.config.num_layers
        ).to(self.device)
        self.step_policy = StepPolicy(
            self.config.hidden_size,
            self.config.max_step,
            self.config.hidden_size,
            self.config.num_layers,
        ).to(self.device)
        params = list(self.direction_policy.parameters()) + list(self.step_policy.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.config.lr)
        self._last_ips_metrics: dict[str, float] = {}
        self._seen_terminals: set[State] = set()
        self._training_history: list[dict[str, Any]] = []
        self._completed_updates = 0
        self._cumulative_rollouts = 0

    @property
    def training_history(self) -> list[dict[str, Any]]:
        """Return the accumulated history, including restored updates."""
        return self._training_history

    @property
    def completed_updates(self) -> int:
        """Return the last globally completed update number."""
        return self._completed_updates

    def target_reward(self) -> dict[State, float]:
        total = float(sum(self.reward_by_terminal.values()))
        return {
            state: self.reward_by_terminal[state] / total for state in self.terminals
        }

    def environment_summary(self) -> dict[str, int]:
        return {
            "budget": self.config.budget,
            "max_step": self.config.max_step,
            "num_states": (self.config.budget + 1) * (self.config.budget + 2) // 2,
            "num_terminals": len(self.terminals),
        }

    def _on_update_start(self, update_step: int) -> None:
        """Hook for algorithms with update-dependent coefficients."""

    def _action_distribution(
        self,
        distribution: Categorical,
        mask: torch.Tensor,
        *,
        explore: bool,
    ) -> Categorical:
        """Hook for variants that alter the rollout/training distribution.

        The base trainer always returns the learned policy unchanged.  An
        exploration variant can override this method, while ``explore=False``
        keeps evaluation on the unmodified learned policy.
        """
        return distribution

    def rollout_episode(self) -> Episode:
        """Sample one evaluation episode through the vectorized rollout path."""
        return self.rollout_batch(1, explore=False)[0]

    @torch.inference_mode()
    def rollout_batch(
        self, batch_size: int, *, explore: bool = False
    ) -> list[Episode]:
        """Sample a complete group with batched policy calls on ``self.device``.

        Environment state and action sampling stay on the accelerator for the
        complete rollout. The finished batch is copied to CPU once and adapted
        to the existing ``Episode`` representation so the IPS and PPO code does
        not change semantics.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        budget = self.config.budget
        max_step = self.config.max_step
        obs_width = budget + 1
        obs_dim = self.env.obs_dim
        device = self.device

        x = torch.zeros(batch_size, dtype=torch.long, device=device)
        y = torch.zeros_like(x)

        observations = torch.zeros(
            (batch_size, budget, obs_dim), dtype=torch.float32, device=device
        )
        step_masks = torch.zeros(
            (batch_size, budget, max_step), dtype=torch.bool, device=device
        )
        directions = torch.zeros(
            (batch_size, budget), dtype=torch.long, device=device
        )
        step_indices = torch.zeros_like(directions)
        direction_log_probs = torch.zeros(
            (batch_size, budget), dtype=torch.float32, device=device
        )
        step_log_probs = torch.zeros_like(direction_log_probs)
        valid_steps = torch.zeros(
            (batch_size, budget), dtype=torch.bool, device=device
        )
        step_options = torch.arange(max_step, device=device).unsqueeze(0)

        for time_index in range(budget):
            remaining = budget - x - y
            active_ids = torch.nonzero(remaining > 0, as_tuple=True)[0]
            if active_ids.numel() == 0:
                break

            active_x = x[active_ids]
            active_y = y[active_ids]
            active_remaining = remaining[active_ids]
            obs = torch.cat(
                (
                    F.one_hot(active_x, num_classes=obs_width),
                    F.one_hot(active_y, num_classes=obs_width),
                    F.one_hot(active_remaining, num_classes=obs_width),
                ),
                dim=-1,
            ).to(dtype=torch.float32)

            direction_mask = torch.ones(
                (active_ids.numel(), 2), dtype=torch.bool, device=device
            )
            base_direction_dist, representation = (
                self.direction_policy.dist_with_rep(obs, direction_mask)
            )
            direction_dist = self._action_distribution(
                base_direction_dist, direction_mask, explore=explore
            )
            direction = direction_dist.sample()

            valid_step_count = active_remaining.clamp(max=max_step)
            step_mask = step_options < valid_step_count.unsqueeze(1)
            base_step_dist = self.step_policy.dist(
                representation, direction, step_mask
            )
            step_dist = self._action_distribution(
                base_step_dist, step_mask, explore=explore
            )
            step_index = step_dist.sample()
            physical_step = step_index + 1

            x[active_ids] = active_x + physical_step * (direction == RIGHT)
            y[active_ids] = active_y + physical_step * (direction == UP)

            observations[active_ids, time_index] = obs
            step_masks[active_ids, time_index] = step_mask
            directions[active_ids, time_index] = direction
            step_indices[active_ids, time_index] = step_index
            direction_log_probs[active_ids, time_index] = (
                direction_dist.log_prob(direction)
            )
            step_log_probs[active_ids, time_index] = step_dist.log_prob(step_index)
            valid_steps[active_ids, time_index] = True

        if torch.any(x + y != budget):
            raise RuntimeError("vectorized rollout failed to reach the terminal frontier")

        # One transfer per completed rollout tensor, rather than transfers and
        # scalar synchronizations after every compound action.
        observations_np = observations.cpu().numpy()
        step_masks_np = step_masks.cpu().numpy()
        directions_np = directions.cpu().numpy()
        step_indices_np = step_indices.cpu().numpy()
        direction_log_probs_np = direction_log_probs.cpu().numpy()
        step_log_probs_np = step_log_probs.cpu().numpy()
        lengths = valid_steps.sum(dim=1).cpu().tolist()
        terminal_x = x.cpu().tolist()
        terminal_y = y.cpu().tolist()
        direction_mask_np = np.ones(2, dtype=bool)

        episodes: list[Episode] = []
        for episode_index in range(batch_size):
            length = int(lengths[episode_index])
            steps = [
                StepRecord(
                    obs=observations_np[episode_index, time_index],
                    direction_mask=direction_mask_np,
                    step_mask=step_masks_np[episode_index, time_index],
                    direction=int(directions_np[episode_index, time_index]),
                    step_index=int(step_indices_np[episode_index, time_index]),
                    log_prob_direction=float(
                        direction_log_probs_np[episode_index, time_index]
                    ),
                    log_prob_step=float(
                        step_log_probs_np[episode_index, time_index]
                    ),
                )
                for time_index in range(length)
            ]
            terminal = State(
                int(terminal_x[episode_index]), int(terminal_y[episode_index])
            )
            episodes.append(
                Episode(
                    steps=steps,
                    terminal=terminal,
                    signature=terminal.signature,
                    trajectory=tuple(
                        (step.direction, step.step_index + 1) for step in steps
                    ),
                    reward=float(self.reward_by_terminal[terminal]),
                )
            )
        return episodes

    def summarize_rollouts(self, samples: Iterable[Episode]) -> dict[str, Any]:
        """Stream terminal and observed-path statistics without storing rollouts."""
        counts: Counter[State] = Counter()
        trajectory_counts: Counter[Trajectory] = Counter()
        trajectory_terminal: dict[Trajectory, State] = {}
        episodes = 0
        for sample in samples:
            episodes += 1
            counts[sample.terminal] += 1
            trajectory_counts[sample.trajectory] += 1
            trajectory_terminal[sample.trajectory] = sample.terminal
        if episodes == 0:
            raise ValueError("samples must be non-empty")
        observed = np.array(
            [counts[state] / episodes for state in self.terminals]
        )
        target = self.target_reward()
        expected = np.array([target[state] for state in self.terminals])
        l1 = float(np.abs(expected - observed).sum())
        outcome_counts = {
            state.signature: int(counts[state]) for state in self.terminals
        }
        outcome_probs = {
            state.signature: float(observed[index])
            for index, state in enumerate(self.terminals)
        }
        trajectory_count_metrics: dict[str, int] = {}
        trajectory_prob_metrics: dict[str, float] = {}
        trajectory_terminal_metrics: dict[str, str] = {}
        conditional_prob_metrics: dict[str, dict[str, float]] = {}
        coverage_metrics: dict[str, int] = {}
        entropy_metrics: dict[str, float] = {}
        normalized_entropy_metrics: dict[str, float] = {}
        effective_trajectory_metrics: dict[str, float] = {}
        max_share_metrics: dict[str, float] = {}
        details_truncated = len(trajectory_counts) > 200
        for state in self.terminals:
            state_trajectories = [
                trajectory
                for trajectory in trajectory_counts
                if trajectory_terminal[trajectory] == state
            ]
            terminal_total = counts[state]
            conditional: dict[str, float] = {}
            shares: list[float] = []
            for trajectory in state_trajectories:
                count = int(trajectory_counts[trajectory])
                share = count / terminal_total if terminal_total else 0.0
                shares.append(share)
            reported_trajectories = (
                state_trajectories
                if not details_truncated
                else sorted(
                    state_trajectories,
                    key=lambda trajectory: trajectory_counts[trajectory],
                    reverse=True,
                )[:5]
            )
            for trajectory in reported_trajectories:
                signature = trajectory_signature(trajectory)
                count = int(trajectory_counts[trajectory])
                share = count / terminal_total if terminal_total else 0.0
                trajectory_count_metrics[signature] = count
                trajectory_prob_metrics[signature] = count / episodes
                trajectory_terminal_metrics[signature] = state.signature
                conditional[signature] = share
            positive_shares = np.asarray([share for share in shares if share > 0])
            entropy = float(
                -np.sum(positive_shares * np.log(positive_shares))
            ) if positive_shares.size else 0.0
            max_entropy = (
                float(np.log(len(state_trajectories)))
                if state_trajectories
                else 0.0
            )
            conditional_prob_metrics[state.signature] = conditional
            coverage_metrics[state.signature] = sum(count > 0 for count in (
                trajectory_counts[trajectory] for trajectory in state_trajectories
            ))
            entropy_metrics[state.signature] = entropy
            normalized_entropy_metrics[state.signature] = (
                entropy / max_entropy
                if max_entropy > 0
                else float(terminal_total > 0)
            )
            effective_trajectory_metrics[state.signature] = float(np.exp(entropy))
            max_share_metrics[state.signature] = max(shares, default=0.0)
        return {
            "eval_episodes": episodes,
            "terminals_hit": len(counts),
            "r2_reward_target": _r2_against(expected, observed),
            "l1_reward_target": l1,
            "tv_reward_target": 0.5 * l1,
            "max_abs_prob_error": float(np.abs(expected - observed).max()),
            "eval_mean_reward": float(
                sum(
                    observed[i] * self.reward_by_terminal[state]
                    for i, state in enumerate(self.terminals)
                )
            ),
            "eval_outcome_counts": outcome_counts,
            "eval_outcome_probs": outcome_probs,
            "eval_unique_trajectories": len(trajectory_counts),
            "eval_reported_trajectories": len(trajectory_count_metrics),
            "eval_trajectory_details_truncated": details_truncated,
            "eval_trajectory_counts": trajectory_count_metrics,
            "eval_trajectory_probs": trajectory_prob_metrics,
            "eval_trajectory_terminal": trajectory_terminal_metrics,
            "eval_conditional_trajectory_probs": conditional_prob_metrics,
            "eval_trajectory_coverage": coverage_metrics,
            "eval_conditional_trajectory_entropy": entropy_metrics,
            "eval_normalized_trajectory_entropy": normalized_entropy_metrics,
            "eval_effective_trajectories": effective_trajectory_metrics,
            "eval_max_conditional_trajectory_share": max_share_metrics,
            "trajectory_entropy_reference": "uniform_over_observed_trajectories",
        }

    def evaluate(
        self, episodes: int = 10_000, *, batch_size: int | None = None
    ) -> dict[str, Any]:
        """Evaluate in rollout batches instead of sampling episodes serially."""
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        if batch_size is None:
            batch_size = min(episodes, max(self.config.group_size, 4_096))
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        def samples() -> Iterable[Episode]:
            remaining = episodes
            while remaining:
                current_batch_size = min(batch_size, remaining)
                yield from self.rollout_batch(current_batch_size)
                remaining -= current_batch_size

        return self.summarize_rollouts(samples())

    def _group_advantages(self, episodes: list[Episode]) -> float:
        advantages, metrics = count_ips_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            eps=self.config.advantage_eps,
        )
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]

    def _joint_policy_loss(
        self, episodes: list[Episode]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        steps: list[StepRecord] = [step for episode in episodes for step in episode.steps]
        obs = torch.as_tensor(
            np.stack([step.obs for step in steps]), dtype=torch.float32, device=self.device
        )
        direction_masks = torch.as_tensor(
            np.stack([step.direction_mask for step in steps]),
            dtype=torch.bool,
            device=self.device,
        )
        step_masks = torch.as_tensor(
            np.stack([step.step_mask for step in steps]), dtype=torch.bool, device=self.device
        )
        directions = torch.tensor(
            [step.direction for step in steps], dtype=torch.long, device=self.device
        )
        step_indices = torch.tensor(
            [step.step_index for step in steps], dtype=torch.long, device=self.device
        )

        base_direction_dist, representation = self.direction_policy.dist_with_rep(
            obs, direction_masks
        )
        direction_dist = self._action_distribution(
            base_direction_dist, direction_masks, explore=True
        )
        step_rep = representation.detach() if self.config.detach_step_rep else representation
        base_step_dist = self.step_policy.dist(step_rep, directions, step_masks)
        step_dist = self._action_distribution(
            base_step_dist, step_masks, explore=True
        )
        new_joint_flat = direction_dist.log_prob(directions) + step_dist.log_prob(step_indices)
        old_joint_flat = torch.tensor(
            [step.log_prob_joint for step in steps], dtype=torch.float32, device=self.device
        )
        entropy_flat = direction_dist.entropy() + step_dist.entropy()

        lengths = [len(episode.steps) for episode in episodes]
        max_length = max(lengths)
        new_joint = _pad_episode_values(new_joint_flat, lengths, max_length)
        old_joint = _pad_episode_values(old_joint_flat, lengths, max_length)
        entropy = _pad_episode_values(entropy_flat, lengths, max_length)
        mask = torch.arange(max_length, device=self.device).unsqueeze(0) < torch.tensor(
            lengths, device=self.device
        ).unsqueeze(1)
        advantages = torch.tensor(
            [episode.steps[0].advantage for episode in episodes],
            dtype=torch.float32,
            device=self.device,
        )

        policy_loss, ratio_metrics = ppo_token_loss(
            new_joint,
            advantages,
            log_paths_pf_old=old_joint,
            mask=mask,
            clip_eps=self.config.clip_ratio,
        )
        token_counts = mask.sum(dim=-1).clamp(min=1)
        mean_entropy = ((entropy * mask).sum(dim=-1) / token_counts).mean()
        loss = policy_loss - self.config.entropy_coef * mean_entropy
        return loss, {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(mean_entropy.item()),
            **ratio_metrics,
        }

    def update(self, episodes: list[Episode]) -> dict[str, float]:
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
        }
        params = list(self.direction_policy.parameters()) + list(self.step_policy.parameters())
        for _ in range(self.config.train_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss, stats = self._joint_policy_loss(episodes)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(params, self.config.grad_clip_norm)
            self.optimizer.step()
            stats["grad_norm"] = float(grad_norm.item())
            stats["param_norm"] = float(
                sum(parameter.detach().norm().item() ** 2 for parameter in params) ** 0.5
            )
            for key in totals:
                totals[key] += stats[key]
        return {
            key: value / self.config.train_epochs for key, value in totals.items()
        }

    def _groups_for_update(self) -> int:
        """Return the number of independently normalized rollout groups."""
        return self.config.num_groups

    def _collect_training_groups(
        self,
    ) -> tuple[list[list[Episode]], list[dict[str, float]]]:
        """Collect one frozen-policy pool, partitioned into advantage groups.

        Subclasses can override this hook to schedule the number of groups or
        to estimate propensities from a larger shared pool. No optimizer step
        occurs here, so every returned episode comes from the same behavior
        policy version.
        """
        groups: list[list[Episode]] = []
        group_metrics: list[dict[str, float]] = []
        for _ in range(self._groups_for_update()):
            group = self.rollout_batch(self.config.group_size, explore=True)
            self._group_advantages(group)
            groups.append(group)
            group_metrics.append(dict(self._last_ips_metrics))
        return groups, group_metrics

    def _update_training_groups(
        self, groups: list[list[Episode]]
    ) -> dict[str, float]:
        """Perform one policy update using all collected advantage groups."""
        return self.update([episode for group in groups for episode in group])

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

        history = self._training_history
        if self._completed_updates > self.config.num_updates:
            raise ValueError(
                "completed updates exceed the configured target: "
                f"{self._completed_updates} > {self.config.num_updates}"
            )
        for update_step in range(
            self._completed_updates + 1,
            self.config.num_updates + 1,
        ):
            self._on_update_start(update_step)
            groups, group_metrics = self._collect_training_groups()
            if not groups or not group_metrics:
                raise RuntimeError("training collection returned no rollout groups")
            all_episodes = [episode for group in groups for episode in group]
            stats = self._update_training_groups(groups)
            batch_counts = Counter(ep.terminal for ep in all_episodes)
            batch_size = len(all_episodes)
            self._cumulative_rollouts += batch_size
            self._seen_terminals.update(batch_counts)
            row: dict[str, Any] = {
                "step": update_step,
                "rollout_groups": len(groups),
                "rollouts_per_update": batch_size,
                "cumulative_rollouts": self._cumulative_rollouts,
                "mean_reward": float(np.mean([ep.reward for ep in all_episodes])),
                "mean_length": float(np.mean([len(ep.steps) for ep in all_episodes])),
                "unique_terminals": len({ep.terminal for ep in all_episodes}),
                "global_unique_outcomes": float(len(self._seen_terminals)),
                "batch_outcome_counts": {
                    state.signature: int(batch_counts[state])
                    for state in self.terminals
                },
                "batch_outcome_probs": {
                    state.signature: float(batch_counts[state] / batch_size)
                    for state in self.terminals
                },
                **{
                    key: float(np.mean([metrics[key] for metrics in group_metrics]))
                    for key in group_metrics[0]
                },
                **stats,
            }
            if eval_every and (update_step == 1 or update_step % eval_every == 0):
                row.update(self.evaluate(eval_episodes))
            history.append(row)
            self._completed_updates = update_step
            if checkpoint_every and update_step % checkpoint_every == 0:
                assert checkpoint_dir is not None
                checkpoint_path = self.save(
                    checkpoint_dir / f"checkpoint_update_{update_step:06d}.pt",
                    update_step=update_step,
                )
                print(f"Checkpoint: {checkpoint_path}")
            if update_step == 1 or update_step % self.config.log_every == 0:
                print(
                    f"update {update_step:4d}  reward={row['mean_reward']:.3f}  "
                    f"groups={row['rollout_groups']}  "
                    f"rollouts={row['rollouts_per_update']}  "
                    f"outcomes={row['ips_unique_outcomes']:.1f}  "
                    f"global_outcomes={row['global_unique_outcomes']:.0f}  "
                    f"{self.probability_label}={row['ips_prob_mean']:.3f}  "
                    f"grad={row['grad_norm']:.3f}  entropy={row['entropy']:.3f}"
                    + (
                        f"  path_lambda={row['path_coefficient']:.2f}"
                        if "path_coefficient" in row
                        else ""
                    )
                    + (
                        f"  eps={row['exploration_epsilon']:.3f}"
                        f"  temp={row['exploration_temperature']:.3f}"
                        if "exploration_epsilon" in row
                        else ""
                    )
                    + (
                        f"  beta={row['reward_beta']:.3f}"
                        if "reward_beta" in row
                        else ""
                    )
                    + (
                        f"  cov={row['coverage_fraction']:.2f}"
                        if "coverage_fraction" in row
                        else ""
                    )
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
                "update_step": update_step,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str, *, device: str = "cpu") -> "CountIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        trainer = cls(payload["config"], device=device)
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer
