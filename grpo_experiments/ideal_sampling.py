"""Reference curves for reward-proportional signature sampling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SERIES_MARKERS = ("o", "^", "s", "D", "v", "P", "X", "*")

IDEAL_SAMPLING_LABEL = "ideal q* ∝ R(x)"
DEFAULT_REWARD_LINEAR_OFFSET = 2000.0
IDEAL_SAMPLING_STYLE = {
    "linestyle": "--",
    "color": "0.25",
    "linewidth": 1.8,
    "alpha": 0.95,
    "zorder": 20,
}


def build_signature_reward_catalog_from_trees(
    trees: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """One row per signature with terminal log_score and reward R = exp(log_reward)."""
    catalog: dict[str, dict[str, float]] = {}
    for tree in trees:
        signature = str(tree["signature"])
        if signature in catalog:
            continue
        log_score = float(tree["log_score"])
        log_reward = float(tree["log_reward"])
        catalog[signature] = {
            "log_score": log_score,
            "log_reward": log_reward,
            "reward": float(np.exp(log_reward)),
        }
    return catalog


def proportional_integer_counts(
    rewards: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Largest-remainder allocation so counts sum exactly to n_samples."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.size == 0:
        return np.array([], dtype=np.int64)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    total = float(rewards.sum())
    if total <= 0.0:
        raise ValueError("reward sum must be positive")
    exact = rewards / total * float(n_samples)
    counts = np.floor(exact).astype(np.int64)
    remainder = int(n_samples - counts.sum())
    if remainder > 0:
        fractional = exact - counts
        for idx in np.argsort(-fractional)[:remainder]:
            counts[idx] += 1
    return counts


def compute_ideal_signature_sampling_table(
    catalog: dict[str, dict[str, float]],
    *,
    n_samples: int = 100_000,
) -> list[dict[str, Any]]:
    """Reward-proportional target counts over a fixed signature catalog."""
    if not catalog:
        return []
    signatures = sorted(catalog)
    log_scores = np.asarray([catalog[s]["log_score"] for s in signatures], dtype=np.float64)
    log_rewards = np.asarray([catalog[s]["log_reward"] for s in signatures], dtype=np.float64)
    rewards = np.asarray([catalog[s]["reward"] for s in signatures], dtype=np.float64)
    counts = proportional_integer_counts(rewards, n_samples)
    q_hat = counts.astype(np.float64) / float(n_samples)
    rows: list[dict[str, Any]] = []
    for idx, signature in enumerate(signatures):
        rows.append(
            {
                "signature": signature,
                "log_score": float(log_scores[idx]),
                "log_reward": float(log_rewards[idx]),
                "reward": float(rewards[idx]),
                "count": int(counts[idx]),
                "q_hat": float(q_hat[idx]),
            }
        )
    return rows


def ideal_sampling_table_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    counts = np.asarray([row["count"] for row in rows], dtype=np.int64)
    q_hat = counts.astype(np.float64) / float(counts.sum())
    reward_sum = float(rewards.sum())
    slope_q_vs_reward = float(1.0 / reward_sum) if reward_sum > 0.0 else float("nan")
    slope_count_vs_reward = float(counts.sum() / reward_sum) if reward_sum > 0.0 else float("nan")
    log_scores = np.asarray([row["log_score"] for row in rows], dtype=np.float64)
    slope_q_vs_log_score, intercept_q_vs_log_score = np.polyfit(log_scores, q_hat, 1)
    return {
        "n_signatures": len(rows),
        "n_samples": int(counts.sum()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "count_min": int(counts.min()),
        "count_max": int(counts.max()),
        "q_hat_min": float(q_hat.min()),
        "q_hat_max": float(q_hat.max()),
        "q_hat_vs_reward_slope": slope_q_vs_reward,
        "count_vs_reward_slope": slope_count_vs_reward,
        "q_hat_vs_log_score_slope": float(slope_q_vs_log_score),
        "q_hat_vs_log_score_intercept": float(intercept_q_vs_log_score),
        "count_top_bottom_ratio": float(counts.max() / max(int(counts.min()), 1)),
    }


def save_ideal_signature_sampling_table(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    from grpo_experiments.eval_utils import save_json

    payload = {
        "metadata": metadata or {},
        "summary": ideal_sampling_table_summary(rows),
        "signatures": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, payload)


def plot_ideal_qhat_vs_log_score(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    n_samples: int,
    title: str | None = None,
) -> None:
    log_scores = np.asarray([row["log_score"] for row in rows], dtype=np.float64)
    q_hat = np.asarray([row["q_hat"] for row in rows], dtype=np.float64)
    order = np.argsort(log_scores)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=220, constrained_layout=True)
    ax.scatter(
        log_scores,
        q_hat,
        s=28.0,
        alpha=0.9,
        color="0.25",
        marker="o",
        label="ideal q* ∝ R(x)",
        zorder=2,
    )
    ax.plot(
        log_scores[order],
        q_hat[order],
        linestyle="--",
        color="0.25",
        linewidth=1.8,
        alpha=0.95,
        zorder=1,
    )
    ax.set_xlabel("terminal log likelihood")
    ax.set_ylabel("q_hat")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best")
    if title is None:
        title = f"Ideal reward-proportional sampling ({n_samples:,} samples, {len(rows)} signatures)"
    ax.set_title(title, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def empirical_qhat_by_signature(
    trees: list[dict[str, Any]],
    *,
    n_samples: int | None = None,
) -> dict[str, float]:
    counts = Counter(str(tree["signature"]) for tree in trees)
    total = int(n_samples if n_samples is not None else len(trees))
    if total <= 0:
        raise ValueError("n_samples must be positive")
    return {signature: float(count) / float(total) for signature, count in counts.items()}


def prepare_qhat_vs_reward_points(
    catalog_rows: list[dict[str, Any]],
    q_hat_by_signature: dict[str, float],
    *,
    include_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a signature-level q_hat dict onto the canonical reward catalog."""
    rewards: list[float] = []
    q_hat: list[float] = []
    for row in catalog_rows:
        signature = str(row["signature"])
        if signature not in q_hat_by_signature:
            if not include_missing:
                continue
            q_value = 0.0
        else:
            q_value = float(q_hat_by_signature[signature])
        rewards.append(float(row["reward"]))
        q_hat.append(q_value)
    return np.asarray(rewards, dtype=np.float64), np.asarray(q_hat, dtype=np.float64)


def _fit_qhat_vs_reward_line(
    rewards: np.ndarray,
    q_hat: np.ndarray,
) -> tuple[float, float] | None:
    finite = np.isfinite(rewards) & np.isfinite(q_hat)
    rewards = rewards[finite]
    q_hat = q_hat[finite]
    if rewards.size < 2 or float(np.std(rewards)) == 0.0:
        return None
    slope, intercept = np.polyfit(rewards, q_hat, 1)
    return float(slope), float(intercept)


def plot_qhat_vs_reward_comparison(
    catalog_rows: list[dict[str, Any]],
    empirical_runs: list[tuple[str, dict[str, float]]],
    output_path: Path,
    *,
    n_samples: int,
    with_fit: bool = True,
    title: str | None = None,
) -> None:
    """Overlay ideal and empirical q_hat against reward R(x)=exp(log_reward)."""
    from grpo_experiments.final_eval_experiment.eval_signature_mass_scatter import (
        configure_qhat_y_axis,
    )

    ideal_rewards = np.asarray([row["reward"] for row in catalog_rows], dtype=np.float64)
    ideal_q_hat = np.asarray([row["q_hat"] for row in catalog_rows], dtype=np.float64)
    reward_sum = float(ideal_rewards.sum())
    fit_x = np.linspace(float(ideal_rewards.min()), float(ideal_rewards.max()), 200)
    fit_y = fit_x / reward_sum

    fig, ax = plt.subplots(figsize=(11, 6), dpi=220, constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    for idx, (label, q_hat_by_signature) in enumerate(empirical_runs):
        rewards, q_hat = prepare_qhat_vs_reward_points(
            catalog_rows,
            q_hat_by_signature,
            include_missing=True,
        )
        observed = q_hat > 0.0
        n_observed = int(observed.sum())
        color = cmap(idx % 10)
        marker = SERIES_MARKERS[idx % len(SERIES_MARKERS)]
        scatter_label = label if not with_fit else "_nolegend_"
        ax.scatter(
            rewards[observed],
            q_hat[observed],
            s=22.0,
            alpha=0.8,
            color=color,
            marker=marker,
            label=f"{label} ({n_observed} signatures)",
            zorder=2 + idx,
            edgecolors="none",
            linewidths=0.0,
        )
        if with_fit:
            fit = _fit_qhat_vs_reward_line(rewards, q_hat)
            if fit is not None:
                slope, intercept = fit
                fit_line_x = np.linspace(float(rewards.min()), float(rewards.max()), 200)
                fit_line_y = slope * fit_line_x + intercept
                ax.plot(
                    fit_line_x,
                    fit_line_y,
                    color=color,
                    linewidth=1.6,
                    alpha=0.9,
                    zorder=10 + idx,
                    label=f"{label}: slope={slope:.4g}, b={intercept:.4g}",
                )

    ax.plot(
        fit_x,
        fit_y,
        linestyle="--",
        color="0.25",
        linewidth=1.8,
        alpha=0.95,
        label=f"ideal q* ∝ R(x), slope={1.0 / reward_sum:.4g}",
        zorder=1,
    )
    ax.scatter(
        ideal_rewards,
        ideal_q_hat,
        s=16.0,
        alpha=0.35,
        color="0.25",
        marker="o",
        zorder=1,
        label="_nolegend_",
    )

    ax.set_xlabel("reward R(x) = exp(log_reward)")
    ax.set_ylabel("q_hat")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best")
    configure_qhat_y_axis(ax, samples=n_samples)
    if title is None:
        title = (
            f"Per-signature empirical mass vs reward "
            f"({n_samples:,} samples/run, {len(catalog_rows)} signatures)"
        )
    ax.set_title(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ideal_qhat_vs_reward(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    n_samples: int,
    title: str | None = None,
) -> None:
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    q_hat = np.asarray([row["q_hat"] for row in rows], dtype=np.float64)
    reward_sum = float(rewards.sum())
    fit_x = np.linspace(float(rewards.min()), float(rewards.max()), 200)
    fit_y = fit_x / reward_sum

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=220, constrained_layout=True)
    ax.scatter(
        rewards,
        q_hat,
        s=28.0,
        alpha=0.9,
        color="0.25",
        marker="o",
        label="ideal q* ∝ R(x)",
        zorder=2,
    )
    ax.plot(
        fit_x,
        fit_y,
        linestyle="--",
        color="0.25",
        linewidth=1.8,
        alpha=0.95,
        label=f"q_hat = R / sum(R), slope={1.0 / reward_sum:.6g}",
        zorder=1,
    )
    ax.set_xlabel("reward R(x) = exp(log_reward)")
    ax.set_ylabel("q_hat")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best")
    if title is None:
        title = f"Ideal q_hat vs reward ({n_samples:,} samples, {len(rows)} signatures)"
    ax.set_title(title, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_signature_log_score_catalog(
    summaries: list[dict[str, Any]],
) -> dict[str, float]:
    """Union of signature -> terminal log_score across sampled_trees summaries."""
    catalog: dict[str, float] = {}
    for row in summaries:
        for signature, log_score in zip(row["signatures"], row["log_scores"], strict=True):
            signature = str(signature)
            if signature not in catalog:
                catalog[signature] = float(log_score)
    return catalog


def compute_ideal_signature_mass(
    catalog: dict[str, float],
    *,
    reward_linear_offset: float = DEFAULT_REWARD_LINEAR_OFFSET,
) -> tuple[np.ndarray, np.ndarray]:
    """Reward-proportional target q*(x) over known signatures, with R(x) affine in log_score."""
    if not catalog:
        return np.array([]), np.array([])
    log_scores = np.asarray(list(catalog.values()), dtype=np.float64)
    rewards = np.maximum(reward_linear_offset + log_scores, 1e-300)
    q_star = rewards / rewards.sum()
    order = np.argsort(log_scores)
    return log_scores[order], q_star[order]


def ideal_sampling_xy(
    log_scores: np.ndarray,
    q_star: np.ndarray,
    *,
    x_axis: str,
    y_axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    if log_scores.size == 0:
        return log_scores, q_star
    if x_axis == "log_score":
        xs = log_scores
    elif x_axis == "likelihood":
        xs = np.exp(log_scores)
    else:
        raise ValueError(f"Unknown x_axis: {x_axis!r}")
    if y_axis == "qhat":
        ys = q_star
    elif y_axis == "log_qhat":
        ys = np.log(np.maximum(q_star, 1e-300))
    else:
        raise ValueError(f"Unknown y_axis: {y_axis!r}")
    order = np.argsort(xs)
    return xs[order], ys[order]


def compute_ideal_log_score_density(
    catalog: dict[str, float],
    bin_edges: np.ndarray,
    *,
    reward_linear_offset: float = DEFAULT_REWARD_LINEAR_OFFSET,
) -> tuple[np.ndarray, np.ndarray]:
    """Marginal log-score density under ideal signature-level reward-proportional sampling."""
    log_scores, q_star = compute_ideal_signature_mass(
        catalog,
        reward_linear_offset=reward_linear_offset,
    )
    if log_scores.size == 0:
        return np.array([]), np.array([])

    density = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    for log_score, weight in zip(log_scores, q_star, strict=True):
        idx = int(np.searchsorted(bin_edges, log_score, side="right") - 1)
        idx = min(max(idx, 0), len(density) - 1)
        density[idx] += float(weight)

    bin_widths = np.diff(bin_edges)
    total = density.sum()
    if total > 0:
        density = density / (total * bin_widths)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, density


def plot_ideal_sampling_reference(
    ax: plt.Axes,
    summaries: list[dict[str, Any]],
    *,
    x_axis: str,
    y_axis: str,
    group_by: str = "signature",
    label: str = IDEAL_SAMPLING_LABEL,
    xlim: tuple[float, float] | None = None,
    reward_linear_offset: float = DEFAULT_REWARD_LINEAR_OFFSET,
    catalog_rows: list[dict[str, Any]] | None = None,
) -> None:
    if group_by != "signature":
        return
    if catalog_rows is not None:
        log_scores = np.asarray([row["log_score"] for row in catalog_rows], dtype=np.float64)
        q_star = np.asarray([row["q_hat"] for row in catalog_rows], dtype=np.float64)
    else:
        catalog = build_signature_log_score_catalog(summaries)
        log_scores, q_star = compute_ideal_signature_mass(
            catalog,
            reward_linear_offset=reward_linear_offset,
        )
    xs, ys = ideal_sampling_xy(log_scores, q_star, x_axis=x_axis, y_axis=y_axis)
    if xlim is not None:
        mask = (xs >= xlim[0]) & (xs <= xlim[1])
        xs = xs[mask]
        ys = ys[mask]
    if xs.size < 2:
        return
    ax.plot(xs, ys, label=label, **IDEAL_SAMPLING_STYLE)
