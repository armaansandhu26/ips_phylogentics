"""
Sampling metrics against exact ground truth.

The key diagnostic: does the learned sampler match the *marginal* posterior
pi*(x) ∝ R(x), or the *biased* trajectory-IPS fixed point pi_ips(x) ∝ m(x) R(x)?
We fit both and report R² for each, plus the slope of log q̂(x) vs log R(x)
(the merge-toy analogue of the `signature_qhat_vs_loglikelihood` phylo plot).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

LOG_EPS = 1e-12


@dataclass(frozen=True)
class SamplingMetrics:
    episodes: int
    signatures_hit: int
    num_topologies: int
    r2_marginal: float
    r2_ips: float
    logq_slope: float
    logq_r2: float
    mean_log_reward: float


def sample_signatures(trainer, n: int) -> Counter:
    counts: Counter = Counter()
    trainer.policy.eval()
    for _ in range(n):
        ep = trainer.rollout_episode()
        counts[ep.signature] += 1
    return counts


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 1.0


def _r2_against(x: np.ndarray, y: np.ndarray) -> float:
    """R² of the best line y ~ a*x + b (proportionality check)."""
    if x.size < 2:
        return 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return _r2(y, slope * x + intercept)


def fit_sampling_metrics(trainer, *, episodes: int) -> SamplingMetrics:
    catalog = trainer.catalog
    counts = sample_signatures(trainer, episodes)
    hit = len(counts)

    sigs = [s for s in catalog.signatures if counts.get(s, 0) > 0]
    qhat = np.array([counts[s] / episodes for s in sigs], dtype=np.float64)

    target_marginal = catalog.target_marginal()
    target_ips = catalog.target_ips()
    pm = np.array([target_marginal[s] for s in sigs], dtype=np.float64)
    pi = np.array([target_ips[s] for s in sigs], dtype=np.float64)

    r2_marginal = _r2_against(pm, qhat)
    r2_ips = _r2_against(pi, qhat)

    log_r = np.array([catalog.log_reward[s] for s in sigs], dtype=np.float64)
    log_q = np.log(np.maximum(qhat, LOG_EPS))
    if log_r.size >= 2:
        slope, intercept = np.polyfit(log_r, log_q, 1)
        logq_r2 = _r2(log_q, slope * log_r + intercept)
    else:
        slope, logq_r2 = 0.0, 0.0

    mean_log_reward = float(
        np.sum([counts[s] * catalog.log_reward[s] for s in sigs]) / max(episodes, 1)
    )

    return SamplingMetrics(
        episodes=episodes,
        signatures_hit=hit,
        num_topologies=len(catalog.signatures),
        r2_marginal=float(r2_marginal),
        r2_ips=float(r2_ips),
        logq_slope=float(slope),
        logq_r2=float(logq_r2),
        mean_log_reward=mean_log_reward,
    )
