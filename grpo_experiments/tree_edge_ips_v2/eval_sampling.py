from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SamplingMetrics:
    sampled_unique: int
    catalog_size: int | None
    hit_fraction: float | None
    density_r2: float | None
    log_r2: float | None
    log_slope: float | None
    mean_return: float


def _ols_r2_and_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    ss_res = np.square(y - pred).sum()
    ss_tot = np.square(y - y.mean()).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(r2), float(slope)


def ideal_density(rewards_by_outcome: Mapping[Hashable, float]) -> dict[Hashable, float]:
    total = float(sum(max(0.0, reward) for reward in rewards_by_outcome.values()))
    if total <= 0.0:
        raise ValueError("At least one catalog reward must be positive.")
    return {outcome: max(0.0, reward) / total for outcome, reward in rewards_by_outcome.items()}


def sampled_density(outcomes: Sequence[Hashable]) -> dict[Hashable, float]:
    counts = Counter(outcomes)
    n = float(len(outcomes))
    if n == 0:
        raise ValueError("outcomes must be non-empty.")
    return {outcome: count / n for outcome, count in counts.items()}


def compute_sampling_metrics(
    outcomes: Sequence[Hashable],
    returns: Sequence[float],
    *,
    rewards_by_outcome: Mapping[Hashable, float] | None = None,
    eps: float = 1e-12,
) -> SamplingMetrics:
    """Evaluate sampled density against the ideal R(tau) / sum R when available."""

    p_hat = sampled_density(outcomes)
    sampled_unique = len(p_hat)
    mean_return = float(np.mean(np.asarray(returns, dtype=np.float64))) if returns else float("nan")

    if rewards_by_outcome is None:
        return SamplingMetrics(
            sampled_unique=sampled_unique,
            catalog_size=None,
            hit_fraction=None,
            density_r2=None,
            log_r2=None,
            log_slope=None,
            mean_return=mean_return,
        )

    target = ideal_density(rewards_by_outcome)
    catalog_outcomes = list(target)
    target_density = np.asarray([target[outcome] for outcome in catalog_outcomes], dtype=np.float64)
    sampled = np.asarray([p_hat.get(outcome, 0.0) for outcome in catalog_outcomes], dtype=np.float64)
    density_r2, _ = _ols_r2_and_slope(target_density, sampled)

    shared = [outcome for outcome in catalog_outcomes if outcome in p_hat and rewards_by_outcome[outcome] > 0]
    if shared:
        log_reward = np.log(np.asarray([rewards_by_outcome[outcome] for outcome in shared], dtype=np.float64) + eps)
        log_prob = np.log(np.asarray([p_hat[outcome] for outcome in shared], dtype=np.float64) + eps)
        log_r2, log_slope = _ols_r2_and_slope(log_reward, log_prob)
    else:
        log_r2, log_slope = float("nan"), float("nan")

    return SamplingMetrics(
        sampled_unique=sampled_unique,
        catalog_size=len(target),
        hit_fraction=float(sampled_unique / max(len(target), 1)),
        density_r2=density_r2,
        log_r2=log_r2,
        log_slope=log_slope,
        mean_return=mean_return,
    )
