"""Binned log-log diagnostics for reward-proportional sampling.

Raw signature/topology scatter plots show horizontal bands because q_hat comes from
integer counts. This module averages ln q_hat within equal log-score bins so the
relationship between reward and empirical mass is easier to read and fit.

Target for proportional sampling: ln q_hat vs ln R with slope ~ 1 (or 1/tau if
tempered), where R(x) = exp(log_reward) from the sampled tree catalog.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from grpo_experiments.eval_utils import save_json
from grpo_experiments.ideal_sampling import (
    IDEAL_SAMPLING_LABEL,
    IDEAL_SAMPLING_STYLE,
    build_signature_reward_catalog_from_trees,
    compute_ideal_signature_sampling_table,
)


def _equal_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    if values.size == 0:
        raise ValueError("cannot bin an empty array")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = max(abs(lo), 1.0) * 1e-3
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, n_bins + 1)


def _assign_bins(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bin_edges, values, side="right") - 1
    return np.clip(idx, 0, len(bin_edges) - 2).astype(np.int64)


def collect_group_stats(
    summary: dict[str, Any],
    *,
    group_by: str,
    min_count: int,
) -> dict[str, np.ndarray]:
    """Per-group log_score, ln reward, ln q_hat, and raw counts."""
    counts: dict[str, int] = defaultdict(int)
    signature_totals = defaultdict(float)
    signature_counts = defaultdict(int)
    signature_ln_reward = {}
    topology_signatures: dict[str, set[str]] = defaultdict(set)

    signatures = summary["signatures"]
    topology_ids = summary["topology_ids"]
    log_scores = summary["log_scores"]
    log_rewards = summary["log_rewards"]
    n_samples = int(summary["samples"])
    log_total = float(np.log(n_samples))

    for signature, topology_id, log_score, log_reward in zip(
        signatures,
        topology_ids,
        log_scores,
        log_rewards,
        strict=True,
    ):
        signature = str(signature)
        topology_id = str(topology_id)
        log_score = float(log_score)
        log_reward = float(log_reward)

        signature_totals[signature] += log_score
        signature_counts[signature] += 1
        if signature not in signature_ln_reward:
            signature_ln_reward[signature] = log_reward
        topology_signatures[topology_id].add(signature)

        group_id = signature if group_by == "signature" else topology_id
        counts[group_id] += 1

    xs: list[float] = []
    ln_rewards: list[float] = []
    ln_qhats: list[float] = []
    group_counts: list[int] = []

    for group_id, count in counts.items():
        if count < min_count:
            continue
        if group_by == "signature":
            x_value = signature_totals[group_id] / signature_counts[group_id]
            ln_reward = signature_ln_reward[group_id]
        else:
            observed_signature_log_scores = np.asarray(
                [
                    signature_totals[sig] / signature_counts[sig]
                    for sig in topology_signatures[group_id]
                ],
                dtype=np.float64,
            )
            observed_ln_rewards = np.asarray(
                [signature_ln_reward[sig] for sig in topology_signatures[group_id]],
                dtype=np.float64,
            )
            x_value = float(np.logaddexp.reduce(observed_signature_log_scores))
            ln_reward = float(np.logaddexp.reduce(observed_ln_rewards))

        xs.append(x_value)
        ln_rewards.append(ln_reward)
        ln_qhats.append(float(np.log(count) - log_total))
        group_counts.append(count)

    order = np.argsort(np.asarray(xs, dtype=np.float64))
    return {
        "log_scores": np.asarray(xs, dtype=np.float64)[order],
        "ln_rewards": np.asarray(ln_rewards, dtype=np.float64)[order],
        "ln_qhat": np.asarray(ln_qhats, dtype=np.float64)[order],
        "counts": np.asarray(group_counts, dtype=np.int64)[order],
    }


def aggregate_binned_ln_qhat(
    log_scores: np.ndarray,
    ln_rewards: np.ndarray,
    ln_qhat: np.ndarray,
    counts: np.ndarray,
    bin_edges: np.ndarray,
    *,
    min_groups: int = 1,
) -> dict[str, np.ndarray]:
    """Average ln q_hat within equal log-score bins."""
    n_bins = len(bin_edges) - 1
    bin_idx = _assign_bins(log_scores, bin_edges)
    centers_log_score = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mean_ln_qhat = np.full(n_bins, np.nan, dtype=np.float64)
    std_ln_qhat = np.full(n_bins, np.nan, dtype=np.float64)
    sem_ln_qhat = np.full(n_bins, np.nan, dtype=np.float64)
    centers_ln_reward = np.full(n_bins, np.nan, dtype=np.float64)
    n_groups = np.zeros(n_bins, dtype=np.int64)
    total_draws = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n < min_groups:
            continue
        vals = ln_qhat[mask]
        mean_ln_qhat[b] = float(np.mean(vals))
        std_ln_qhat[b] = float(np.std(vals))
        sem_ln_qhat[b] = float(np.std(vals) / np.sqrt(max(n, 1)))
        centers_ln_reward[b] = float(np.mean(ln_rewards[mask]))
        n_groups[b] = n
        total_draws[b] = int(np.sum(counts[mask]))

    valid = np.isfinite(mean_ln_qhat) & np.isfinite(centers_ln_reward)
    return {
        "bin_edges": bin_edges,
        "bin_centers_log_score": centers_log_score[valid],
        "bin_centers_ln_reward": centers_ln_reward[valid],
        "mean_ln_qhat": mean_ln_qhat[valid],
        "std_ln_qhat": std_ln_qhat[valid],
        "sem_ln_qhat": sem_ln_qhat[valid],
        "n_groups": n_groups[valid],
        "total_draws": total_draws[valid],
    }


def _fit_loglog(xs: np.ndarray, ys: np.ndarray) -> dict[str, float] | None:
    if xs.size < 2 or float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return None
    slope, intercept = np.polyfit(xs, ys, deg=1)
    yhat = slope * xs + intercept
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((ys - yhat) ** 2)))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "rmse": float(rmse),
        "pearson_r": float(np.corrcoef(xs, ys)[0, 1]),
    }


def _ideal_binned_series(
    summary: dict[str, Any],
    bin_edges: np.ndarray,
    *,
    min_groups: int,
) -> dict[str, np.ndarray] | None:
    trees = [{"signature": s, "log_score": ls, "log_reward": lr} for s, ls, lr in zip(
        summary["signatures"],
        summary["log_scores"],
        summary["log_rewards"],
        strict=True,
    )]
    catalog = build_signature_reward_catalog_from_trees(trees)
    if not catalog:
        return None
    ideal_rows = compute_ideal_signature_sampling_table(catalog, n_samples=int(summary["samples"]))
    log_scores = np.asarray([row["log_score"] for row in ideal_rows], dtype=np.float64)
    ln_rewards = np.asarray([row["log_reward"] for row in ideal_rows], dtype=np.float64)
    ln_qhat = np.log(np.maximum([row["q_hat"] for row in ideal_rows], 1e-300))
    counts = np.asarray([row["count"] for row in ideal_rows], dtype=np.int64)
    return aggregate_binned_ln_qhat(
        log_scores,
        ln_rewards,
        ln_qhat,
        counts,
        bin_edges,
        min_groups=min_groups,
    )


def plot_binned_loglog_ln_qhat_vs_ln_reward(
    summaries: list[dict[str, Any]],
    output_path: Path,
    *,
    group_by: str,
    samples: int,
    n_bins: int,
    min_count: int,
    min_bin_groups: int,
    with_fit: bool,
    show_ideal: bool = True,
) -> list[dict[str, Any]]:
    """Plot binned mean ln q_hat vs ln R with optional OLS fit (target slope ~ 1)."""
    cmap = plt.get_cmap("tab10")
    combined = np.concatenate(
        [collect_group_stats(row, group_by=group_by, min_count=min_count)["log_scores"] for row in summaries]
    )
    bin_edges = _equal_bin_edges(combined, n_bins)

    title_target = "Per-signature" if group_by == "signature" else "Per-topology"
    fig, ax = plt.subplots(figsize=(11, 6), dpi=220, constrained_layout=True)
    fit_rows: list[dict[str, Any]] = []

    for idx, summary in enumerate(summaries):
        groups = collect_group_stats(summary, group_by=group_by, min_count=min_count)
        binned = aggregate_binned_ln_qhat(
            groups["log_scores"],
            groups["ln_rewards"],
            groups["ln_qhat"],
            groups["counts"],
            bin_edges,
            min_groups=min_bin_groups,
        )
        xs = binned["bin_centers_ln_reward"]
        ys = binned["mean_ln_qhat"]
        yerr = binned["sem_ln_qhat"]
        color = cmap(idx % 10)

        if xs.size == 0:
            fit_rows.append({"label": summary["label"], "n_bins_used": 0})
            continue

        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="o",
            ms=6.5,
            capsize=3.0,
            color=color,
            alpha=0.95,
            label=f"{summary['label']} ({int(binned['n_groups'].sum())} groups in {xs.size} bins)",
            zorder=3 + idx,
        )

        fit = None
        if with_fit:
            fit = _fit_loglog(xs, ys)
            if fit is not None:
                xfit = np.linspace(float(xs.min()), float(xs.max()), 200)
                yfit = fit["slope"] * xfit + fit["intercept"]
                ax.plot(
                    xfit,
                    yfit,
                    color=color,
                    linewidth=1.8,
                    alpha=0.95,
                    zorder=10 + idx,
                    label=(
                        f"slope={fit['slope']:.3f}, R²={fit['r2']:.3f}"
                        if len(summaries) == 1
                        else f"{summary['label']} fit (slope={fit['slope']:.3f})"
                    ),
                )

        fit_rows.append(
            {
                "label": summary["label"],
                "group_by": group_by,
                "n_bins_requested": n_bins,
                "n_bins_used": int(xs.size),
                "min_count": min_count,
                "min_bin_groups": min_bin_groups,
                "fit": fit,
            }
        )

    if show_ideal and summaries:
        ideal = _ideal_binned_series(
            summaries[0],
            bin_edges,
            min_groups=min_bin_groups,
        )
        if ideal is not None and ideal["bin_centers_ln_reward"].size > 0:
            ax.plot(
                ideal["bin_centers_ln_reward"],
                ideal["mean_ln_qhat"],
                label=IDEAL_SAMPLING_LABEL,
                **IDEAL_SAMPLING_STYLE,
            )

    ax.axline((0, 0), slope=1.0, color="0.55", linestyle=":", linewidth=1.2, label="slope = 1 reference")
    ax.set_xlabel("ln R(x) bin center  [R = exp(log_reward)]")
    ax.set_ylabel("mean ln q_hat(x) within log-score bin")
    ax.set_title(
        f"{title_target} binned log-log: mean ln q_hat vs ln reward "
        f"({samples:,} samples, {n_bins} log-score bins)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return fit_rows


def generate_binned_loglog_plots(
    summaries: list[dict[str, Any]],
    *,
    output_dir: Path,
    group_by: str,
    samples: int,
    sample_tag: str,
    n_bins: int,
    min_count: int,
    min_bin_groups: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    stem = f"{group_by}_binned_loglog_lnqhat_vs_lnr_{sample_tag}"
    if min_count > 1:
        stem += f"_mincount{min_count}"
    paths = [
        output_dir / f"{stem}.png",
        output_dir / f"{stem}_fit.png",
    ]
    fit_rows_plain = plot_binned_loglog_ln_qhat_vs_ln_reward(
        summaries,
        paths[0],
        group_by=group_by,
        samples=samples,
        n_bins=n_bins,
        min_count=min_count,
        min_bin_groups=min_bin_groups,
        with_fit=False,
        show_ideal=True,
    )
    fit_rows = plot_binned_loglog_ln_qhat_vs_ln_reward(
        summaries,
        paths[1],
        group_by=group_by,
        samples=samples,
        n_bins=n_bins,
        min_count=min_count,
        min_bin_groups=min_bin_groups,
        with_fit=True,
        show_ideal=True,
    )
    save_json(
        output_dir / f"binned_loglog_fits_{group_by}_{sample_tag}.json",
        {"runs": fit_rows, "runs_no_fit": fit_rows_plain},
    )
    return [p.name for p in paths], fit_rows
