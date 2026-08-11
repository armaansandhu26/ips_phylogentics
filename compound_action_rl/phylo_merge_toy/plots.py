"""Plots: training curves, signature sampling vs exact targets, log q̂ vs log R."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval_sampling import sample_signatures  # noqa: E402

LOG_EPS = 1e-12


@dataclass
class SamplingPlotResult:
    episodes: int
    signatures_hit: int
    num_topologies: int
    r2_marginal: float
    r2_ips: float
    logq_slope: float
    logq_r2: float

    def to_dict(self) -> dict:
        return asdict(self)


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return float(slope), float(intercept), float(r2)


def plot_signature_sampling(trainer, *, out_path: Path, episodes: int = 5000, title: str | None = None) -> SamplingPlotResult:
    """Scatter empirical q̂(x) against both exact targets: marginal pi∝R and biased pi∝mR."""
    catalog = trainer.catalog
    counts = sample_signatures(trainer, episodes)
    hit = len(counts)

    sigs = [s for s in catalog.signatures if counts.get(s, 0) > 0]
    qhat = np.array([counts[s] / episodes for s in sigs], dtype=np.float64)
    tm = catalog.target_marginal()
    ti = catalog.target_ips()
    pm = np.array([tm[s] for s in sigs], dtype=np.float64)
    pi = np.array([ti[s] for s in sigs], dtype=np.float64)

    sm, _, r2_m = _ols(pm, qhat) if pm.size >= 2 else (0.0, 0.0, 0.0)
    si, _, r2_i = _ols(pi, qhat) if pi.size >= 2 else (0.0, 0.0, 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    lim = max(qhat.max(), pm.max(), pi.max()) * 1.05 if qhat.size else 1.0

    ax1.scatter(pm, qhat, s=45, color="#0984e3", edgecolors="white", label=f"vs marginal ∝R  (R²={r2_m:.3f})")
    ax1.plot([0, lim], [0, lim], "-", color="#00b894", linewidth=2, label="y=x (perfect)")
    ax1.set_xlabel("exact marginal target  π*(x) ∝ R(x)")
    ax1.set_ylabel("empirical sampling  q̂(x)")
    ax1.set_title("Unbiased target")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.25)

    ax2.scatter(pi, qhat, s=45, color="#e17055", edgecolors="white", label=f"vs biased ∝mR  (R²={r2_i:.3f})")
    ax2.plot([0, lim], [0, lim], "-", color="#00b894", linewidth=2, label="y=x (perfect)")
    ax2.set_xlabel("trajectory-IPS target  π(x) ∝ m(x)·R(x)")
    ax2.set_ylabel("empirical sampling  q̂(x)")
    ax2.set_title("Multiplicity-biased target")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.25)

    if title is None:
        title = f"Signature sampling — {hit}/{len(catalog.signatures)} topologies hit ({episodes} eps)"
    fig.suptitle(title)

    # log q̂ vs log R panel result
    log_r = np.array([catalog.log_reward[s] for s in sigs], dtype=np.float64)
    log_q = np.log(np.maximum(qhat, LOG_EPS))
    logq_slope, _, logq_r2 = _ols(log_r, log_q) if log_r.size >= 2 else (0.0, 0.0, 0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return SamplingPlotResult(
        episodes=episodes,
        signatures_hit=hit,
        num_topologies=len(catalog.signatures),
        r2_marginal=float(r2_m),
        r2_ips=float(r2_i),
        logq_slope=float(logq_slope),
        logq_r2=float(logq_r2),
    )


def plot_qhat_vs_logreward(trainer, *, out_path: Path, episodes: int = 5000, title: str | None = None) -> tuple[float, float]:
    """log q̂(x) vs log R(x) — merge-toy analogue of signature_qhat_vs_loglikelihood.

    Overlays the two ideal lines: marginal (log q̂ = log R + c) and biased
    (log q̂ = log R + log m + c). Perfect marginal sampling lies on the marginal
    line with slope 1.
    """
    catalog = trainer.catalog
    counts = sample_signatures(trainer, episodes)
    sigs = [s for s in catalog.signatures if counts.get(s, 0) > 0]
    qhat = np.array([counts[s] / episodes for s in sigs], dtype=np.float64)
    log_r = np.array([catalog.log_reward[s] for s in sigs], dtype=np.float64)
    log_m = np.array([np.log(catalog.multiplicity[s]) for s in sigs], dtype=np.float64)
    log_q = np.log(np.maximum(qhat, LOG_EPS))

    slope, intercept, r2 = _ols(log_r, log_q) if log_r.size >= 2 else (0.0, 0.0, 0.0)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    sc = ax.scatter(log_r, log_q, c=log_m, cmap="viridis", s=55, edgecolors="white", label="topologies (color=log m)")
    fig.colorbar(sc, ax=ax, label="log multiplicity  log m(x)")

    if log_r.size >= 2:
        xl = np.linspace(float(log_r.min()), float(log_r.max()), 100)
        ax.plot(xl, slope * xl + intercept, ":", color="#2d3436", linewidth=2, label=f"OLS slope={slope:.3f} R²={r2:.3f}")
        # marginal ideal: log q = log R - logZ, slope 1 through centroid of the marginal-consistent points
        c_marg = float(log_q.mean() - log_r.mean())
        ax.plot(xl, xl + c_marg, "-", color="#00b894", linewidth=2, label="ideal marginal (slope 1)")

    ax.set_xlabel("log R(x)  (topology log-reward ~ 'log-likelihood')")
    ax.set_ylabel("log q̂(x)  (empirical sampling log-prob)")
    if title is None:
        title = f"log q̂ vs log R — {trainer.config.reward_profile}/{trainer.config.propensity_mode} ({episodes} eps)"
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return float(r2), float(slope)


def plot_training_from_history(history_path: Path, out_path: Path, *, title: str, smooth_window: int = 50):
    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    steps = [h["step"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    y = np.array([h["mean_log_reward"] for h in history], dtype=np.float64)
    ax.plot(steps, y, color="#0984e3", alpha=0.4, label="mean log R")
    if len(y) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        ax.plot(steps[smooth_window - 1:], np.convolve(y, kernel, "valid"), color="#0984e3", linewidth=2, label=f"smoothed ({smooth_window})")
    ax.set_title("mean log reward"); ax.set_xlabel("update"); ax.grid(alpha=0.25); ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, [h["mean_ess"] for h in history], color="#6c5ce7")
    ax.set_title("mean effective sample size"); ax.set_xlabel("update"); ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(steps, [h["entropy"] for h in history], color="#00b894")
    ax.set_title("policy entropy"); ax.set_xlabel("update"); ax.grid(alpha=0.25)

    evals = [h for h in history if "eval_r2_marginal" in h]
    ax = axes[1, 1]
    if evals:
        es = [h["step"] for h in evals]
        ax.plot(es, [h["eval_r2_marginal"] for h in evals], "o-", color="#0984e3", label="R² vs marginal ∝R")
        ax.plot(es, [h["eval_r2_ips"] for h in evals], "o-", color="#e17055", label="R² vs biased ∝mR")
        ax.set_ylim(-0.05, 1.05); ax.legend()
    ax.set_title("eval R² vs exact targets"); ax.set_xlabel("update"); ax.grid(alpha=0.25)

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return history, evals
