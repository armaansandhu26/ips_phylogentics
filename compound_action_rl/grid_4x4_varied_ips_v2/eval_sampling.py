"""Sampling metrics for convergence checks (enumeration-free log-log diagnostics)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from grid_grpo import GRPOTrainer

LOG_EPS = 1e-8


@dataclass(frozen=True)
class SamplingMetrics:
    episodes: int
    trajectories_hit: int
    mean_return: float
    max_return: float
    r2: float
    slope: float


@dataclass(frozen=True)
class LogLogMetrics:
    episodes: int
    log_r2: float
    log_slope: float
    log_intercept: float


def sample_trajectory_indices(trainer: GRPOTrainer, n: int) -> list[tuple[int, float]]:
    lookup = trainer._traj_lookup
    rows: list[tuple[int, float]] = []
    for _ in range(n):
        ep = trainer.rollout_episode()
        rows.append((lookup.get(ep.outcome, -1), ep.return_))
    return rows


def sample_log_prob_reward(trainer: GRPOTrainer, n: int) -> list[tuple[float, float]]:
    """Return (log p_theta(tau), log R(tau)) for each rollout."""
    rows: list[tuple[float, float]] = []
    for _ in range(n):
        ep = trainer.rollout_episode()
        log_p = ep.log_prob_joint
        log_r = float(np.log(max(ep.return_, LOG_EPS)))
        rows.append((log_p, log_r))
    return rows


def _fit_log_log(log_p: np.ndarray, log_r: np.ndarray) -> tuple[float, float, float]:
    if log_p.size < 2:
        return 0.0, 0.0, 0.0
    slope, intercept = np.polyfit(log_r, log_p, 1)
    pred = slope * log_r + intercept
    ss_res = float(np.sum((log_p - pred) ** 2))
    ss_tot = float(np.sum((log_p - log_p.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return float(r2), float(slope), float(intercept)


def fit_sampling_metrics(
    trainer: GRPOTrainer,
    *,
    episodes: int,
    reward_by_index: dict[int, float],
    ideal_density: dict[int, float],
) -> SamplingMetrics:
    rows = sample_trajectory_indices(trainer, episodes)
    counts = Counter(idx for idx, _ in rows if idx >= 0)
    returns = [r for _, r in rows]

    x_s = np.array([reward_by_index[i] for i in sorted(counts)], dtype=np.float64)
    y_s = np.array([counts[i] / episodes for i in sorted(counts)], dtype=np.float64)

    if x_s.size >= 2:
        slope, intercept = np.polyfit(x_s, y_s, 1)
        pred = slope * x_s + intercept
        ss_res = float(np.sum((y_s - pred) ** 2))
        ss_tot = float(np.sum((y_s - y_s.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    else:
        slope = 0.0
        r2 = 0.0

    return SamplingMetrics(
        episodes=episodes,
        trajectories_hit=len(counts),
        mean_return=float(np.mean(returns)),
        max_return=float(np.max(returns)),
        r2=float(r2),
        slope=float(slope),
    )


def fit_log_log_metrics(trainer: GRPOTrainer, *, episodes: int) -> LogLogMetrics:
    rows = sample_log_prob_reward(trainer, episodes)
    log_p = np.array([r[0] for r in rows], dtype=np.float64)
    log_r = np.array([r[1] for r in rows], dtype=np.float64)
    log_r2, log_slope, log_intercept = _fit_log_log(log_p, log_r)
    return LogLogMetrics(
        episodes=episodes,
        log_r2=log_r2,
        log_slope=log_slope,
        log_intercept=log_intercept,
    )
