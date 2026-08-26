from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
import torch

_LOG_REWARD_EPS = 1e-8
RewardTarget = Literal["likelihood", "shifted_linear"]


@dataclass
class RunningLogWeightNormalizer:
    """Detached EMA scale for log-space importance weights."""

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
            raise ValueError("running advantage clip must be positive")
        if self.log_ratio_clip <= 0.0:
            raise ValueError("running log-ratio clip must be positive")

    @staticmethod
    def _log_mean_exp(values: np.ndarray) -> float:
        maximum = float(values.max())
        return maximum + float(np.log(np.exp(values - maximum).mean()))

    def normalize(self, log_weights: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        if log_weights.ndim != 1 or log_weights.size == 0:
            raise ValueError("log_weights must be a non-empty vector")
        if np.any(~np.isfinite(log_weights)):
            raise ValueError("log_weights must be finite")

        batch_log_first = self._log_mean_exp(log_weights)
        batch_log_second = self._log_mean_exp(2.0 * log_weights)
        bootstrap = self.log_first_moment is None or self.log_second_moment is None
        scale_log_first = batch_log_first if bootstrap else float(self.log_first_moment)
        scale_log_second = batch_log_second if bootstrap else float(self.log_second_moment)
        log_rms = 0.5 * scale_log_second

        stabilized = np.exp(
            np.clip(log_weights - log_rms, -self.log_ratio_clip, self.log_ratio_clip)
        )
        baseline = float(
            np.exp(
                np.clip(
                    scale_log_first - log_rms,
                    -self.log_ratio_clip,
                    self.log_ratio_clip,
                )
            )
        )
        centered = stabilized - baseline
        advantages = np.clip(centered, -self.advantage_clip, self.advantage_clip)

        if bootstrap:
            self.log_first_moment = batch_log_first
            self.log_second_moment = batch_log_second
        else:
            log_decay = math.log(self.decay) if self.decay > 0.0 else -math.inf
            log_new = math.log1p(-self.decay)
            self.log_first_moment = float(
                np.logaddexp(
                    log_decay + float(self.log_first_moment),
                    log_new + batch_log_first,
                )
            )
            self.log_second_moment = float(
                np.logaddexp(
                    log_decay + float(self.log_second_moment),
                    log_new + batch_log_second,
                )
            )
        self.updates += 1
        return advantages, {
            "running_scale_bootstrap": float(bootstrap),
            "running_scale_updates": float(self.updates),
            "running_log_weight_rms": float(log_rms),
            "running_scaled_weight_baseline": baseline,
            "running_scaled_weight_mean": float(stabilized.mean()),
            "running_scaled_weight_std": float(stabilized.std()),
            "running_preclip_advantage_min": float(centered.min()),
            "running_preclip_advantage_max": float(centered.max()),
            "running_advantage_clip_fraction": float(np.mean(advantages != centered)),
        }

    def state_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def terminal_log_rewards_from_scores(
    log_scores: torch.Tensor,
    *,
    reward_target: RewardTarget,
    reward_c: float,
    reward_scale: float,
) -> torch.Tensor:
    """Return log R for the selected phylogenetic target."""
    if reward_scale == 0.0:
        raise ValueError("reward_scale must be non-zero")
    linear_score = (reward_c + log_scores.detach()) / reward_scale
    if reward_target == "likelihood":
        return linear_score
    if reward_target == "shifted_linear":
        positive = linear_score.clamp(min=_LOG_REWARD_EPS)
        if bool(torch.any(linear_score <= 0.0)):
            n_bad = int(torch.sum(linear_score <= 0.0).item())
            print(
                f"warning: clamping {n_bad}/{linear_score.numel()} "
                "shifted_linear scores to keep log R finite",
                flush=True,
            )
        return positive.log()
    raise ValueError(f"unknown reward target: {reward_target!r}")


def learned_reverse_advantages(
    log_scores: torch.Tensor,
    forward_log_probabilities: torch.Tensor,
    reverse_log_probabilities: torch.Tensor,
    *,
    reward_target: RewardTarget,
    reward_c: float,
    reward_scale: float,
    normalizer: RunningLogWeightNormalizer | None,
    advantage_eps: float,
    terminal_ids: Sequence[str],
    trajectory_indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages from log R + log q_phi - log P_F."""
    if (
        log_scores.ndim != 1
        or forward_log_probabilities.shape != log_scores.shape
        or reverse_log_probabilities.shape != log_scores.shape
    ):
        raise ValueError("scores and path log-probabilities must all have shape (B,)")

    log_rewards = terminal_log_rewards_from_scores(
        log_scores,
        reward_target=reward_target,
        reward_c=reward_c,
        reward_scale=reward_scale,
    )
    log_weights = log_rewards + reverse_log_probabilities.detach() - forward_log_probabilities.detach()
    log_weights_np = log_weights.double().cpu().numpy()

    maximum = float(log_weights_np.max())
    scaled = np.exp(log_weights_np - maximum)
    if normalizer is None:
        centered = scaled - scaled.mean()
        std = float(scaled.std())
        advantages_np = centered if std < advantage_eps else centered / (std + advantage_eps)
        normalization_metrics: dict[str, float] = {}
    else:
        advantages_np, normalization_metrics = normalizer.normalize(log_weights_np)

    squared_sum = float(np.square(scaled).sum())
    ess = float(scaled.sum() ** 2 / max(squared_sum, advantage_eps))
    implied_log_probability = (
        forward_log_probabilities.detach() - reverse_log_probabilities.detach()
    ).double().cpu().numpy()
    by_terminal: dict[str, list[float]] = defaultdict(list)
    for terminal_id, value in zip(terminal_ids, implied_log_probability):
        by_terminal[str(terminal_id)].append(float(value))
    within_terminal_stds = [
        float(np.std(values)) for values in by_terminal.values() if len(values) > 1
    ]

    metrics = {
        "ips_ess": ess,
        "ips_ess_fraction": ess / max(len(scaled), 1),
        "ips_unique_outcomes": float(len(set(terminal_ids))),
        "ips_unique_trajectories": float(trajectory_indices.unique().numel()),
        "forward_log_probability_mean": float(forward_log_probabilities.mean().item()),
        "reverse_log_probability_mean": float(reverse_log_probabilities.mean().item()),
        "implied_terminal_log_probability_mean": float(implied_log_probability.mean()),
        "implied_terminal_log_probability_std": float(implied_log_probability.std()),
        "implied_terminal_within_outcome_std": float(
            np.mean(within_terminal_stds) if within_terminal_stds else 0.0
        ),
        "log_importance_weight_mean": float(log_weights_np.mean()),
        "log_importance_weight_min": float(log_weights_np.min()),
        "log_importance_weight_max": float(log_weights_np.max()),
        "target_log_reward_mean": float(log_rewards.mean().item()),
        "target_log_reward_min": float(log_rewards.min().item()),
        "target_log_reward_max": float(log_rewards.max().item()),
        "advantage_mean": float(advantages_np.mean()),
        "advantage_std": float(advantages_np.std()),
        "advantage_min": float(advantages_np.min()),
        "advantage_max": float(advantages_np.max()),
        "advantage_normalization_is_running": float(normalizer is not None),
    }
    metrics.update(normalization_metrics)
    advantages = torch.tensor(
        advantages_np,
        dtype=log_scores.dtype,
        device=log_scores.device,
    )
    return advantages, metrics
