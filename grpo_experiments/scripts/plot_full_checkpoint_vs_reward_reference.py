#!/usr/bin/env python3
"""Compare full-model samples with an IPS reward reference on sampled support.

For trajectories tau ~ P_F, the normalized weights

    w(tau) = R(x) q_phi(tau | x) / P_F(tau)

give a self-normalized Monte Carlo reference for q*(x) proportional to R(x).
The reverse proposal is exactly normalized over all structural histories, so
this remains valid even when it is not the model's exact reverse posterior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from reward_probability_plot_reference import (  # noqa: E402
    ideal_line_label,
    linear_probability_limits,
    log_probability_limits,
    merge_reward_axis_bounds,
    pearson_vs_ideal_sampling,
    pearson_vs_ideal_sampling_linear,
    shared_reward_axis_bounds,
)


GRAY = "#777777"
BLUE = "#1976d2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-bins", type=int, default=100)
    parser.add_argument("--signature-top-k", type=int, default=500)
    parser.add_argument(
        "--scatter-points",
        type=int,
        default=0,
        help="Maximum displayed trajectories; 0 displays all samples.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot-method",
        choices=("learned-reverse", "ppo"),
        default="learned-reverse",
        help="Controls method-specific plot titles and legend estimator.",
    )
    parser.add_argument(
        "--reference-run-dir",
        type=Path,
        default=None,
        help=(
            "Reference run used for shared reward-axis limits when "
            "--shared-reference is enabled."
        ),
    )
    parser.add_argument(
        "--shared-reference",
        action="store_true",
        help=(
            "Use a shared reward-axis range across methods. Each method still "
            "uses its own estimated log partition for the ideal line and auto "
            "probability-axis limits."
        ),
    )
    parser.add_argument(
        "--reference-log-partition",
        type=float,
        default=None,
        help=(
            "Optional fixed log partition for the ideal line. Default: each "
            "method's own estimated log partition."
        ),
    )
    return parser.parse_args()


def load_terminal_log_likelihood(
    payload: np.lib.npyio.NpzFile,
    *,
    shift: float,
) -> np.ndarray:
    if "raw_log_likelihood" in payload:
        return payload["raw_log_likelihood"].astype(np.float64)
    log_score = payload["log_score"].astype(np.float64)
    return log_score - shift


def normalized_reference_weights(
    log_score: np.ndarray,
    log_pf: np.ndarray,
    log_q_reverse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if np.any(log_score <= 0.0):
        raise ValueError("shifted-linear scores must be positive")
    log_weights = (
        np.log(log_score.astype(np.float64))
        + log_q_reverse.astype(np.float64)
        - log_pf.astype(np.float64)
    )
    shifted = np.exp(log_weights - log_weights.max())
    weights = shifted / shifted.sum()
    ess = float(1.0 / np.square(weights).sum())
    return weights, log_weights, ess


def aggregate(
    group_index: np.ndarray,
    groups: int,
    reference_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(group_index, minlength=groups)
    model_frequency = counts.astype(np.float64) / len(group_index)
    reference_frequency = np.bincount(
        group_index,
        weights=reference_weights,
        minlength=groups,
    )
    return counts, model_frequency, reference_frequency


def save_rank_plot(
    output: Path,
    *,
    model_frequency: np.ndarray,
    reference_frequency: np.ndarray,
    title: str,
    xlabel: str,
    log_y: bool = False,
) -> np.ndarray:
    order = np.argsort(-reference_frequency)
    ranks = np.arange(1, len(order) + 1)
    model = model_frequency[order]
    reference = reference_frequency[order]

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=220, constrained_layout=True)
    if log_y:
        positive = np.concatenate([model[model > 0], reference[reference > 0]])
        floor = max(float(positive.min()) / 2.0, 1e-12)
        model = np.maximum(model, floor)
        reference = np.maximum(reference, floor)
        ax.set_yscale("log")
    ax.scatter(
        ranks,
        reference,
        marker="o",
        s=34,
        color=GRAY,
        alpha=0.85,
        label="ideal reward reference (IPS/SNIS)",
        zorder=2,
    )
    ax.scatter(
        ranks,
        model,
        marker="x",
        s=46,
        linewidths=1.5,
        color=BLUE,
        label="learned checkpoint samples",
        zorder=3,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability mass")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return order


def save_score_plot(
    output: Path,
    *,
    original_log_likelihood: np.ndarray,
    reference_weights: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(
        float(original_log_likelihood.min()),
        float(original_log_likelihood.max()),
        bins + 1,
    )
    model_counts, _ = np.histogram(original_log_likelihood, bins=edges)
    reference_mass, _ = np.histogram(
        original_log_likelihood,
        bins=edges,
        weights=reference_weights,
    )
    model_mass = model_counts.astype(np.float64) / len(original_log_likelihood)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(12, 7), dpi=220, constrained_layout=True)
    ax.scatter(
        centers,
        reference_mass,
        marker="o",
        s=38,
        color=GRAY,
        alpha=0.85,
        label="ideal reward reference (IPS/SNIS)",
        zorder=2,
    )
    ax.scatter(
        centers,
        model_mass,
        marker="x",
        s=48,
        linewidths=1.5,
        color=BLUE,
        label="learned checkpoint samples",
        zorder=3,
    )
    ax.set_xlabel("Terminal-tree log likelihood")
    ax.set_ylabel("Probability mass per likelihood bin")
    ax.set_title("Full model: checkpoint versus shifted-linear reward reference")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return edges, centers, model_mass, reference_mass


def save_signature_likelihood_plot(
    output: Path,
    *,
    signature_log_likelihood: np.ndarray,
    model_frequency: np.ndarray,
    reference_frequency: np.ndarray,
    samples: int,
    log_y: bool,
) -> None:
    """Og-style per-signature mass versus terminal log likelihood."""
    order = np.argsort(signature_log_likelihood)
    x = signature_log_likelihood[order]
    model = model_frequency[order]
    reference = reference_frequency[order]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=220, constrained_layout=True)
    if log_y:
        floor = 0.5 / samples
        model = np.maximum(model, floor)
        reference = np.maximum(reference, floor)
        ax.set_yscale("log")
        ylabel = "Per-signature probability mass (log scale)"
        title_prefix = "Log per-signature empirical mass"
    else:
        ylabel = "Per-signature probability mass"
        title_prefix = "Per-signature empirical mass"

        def count_formatter(value: float, _position: float) -> str:
            count = value * samples
            count_text = (
                str(int(round(count)))
                if abs(count - round(count)) < 1e-6
                else f"{count:.1f}"
            )
            return f"{count_text}/{samples // 1000}k"

        ax.yaxis.set_major_formatter(FuncFormatter(count_formatter))

    # Full-model runs can have nearly one million distinct signatures. Small,
    # transparent, rasterized marks preserve the requested gray-dot/blue-cross
    # convention without creating an enormous vector artist.
    ax.scatter(
        x,
        reference,
        marker="o",
        s=2.2,
        linewidths=0,
        color=GRAY,
        alpha=0.18,
        rasterized=True,
        label="ideal reward reference (IPS/SNIS)",
        zorder=2,
    )
    ax.scatter(
        x,
        model,
        marker="x",
        s=3.0,
        linewidths=0.28,
        color=BLUE,
        alpha=0.20,
        rasterized=True,
        label="learned checkpoint samples",
        zorder=3,
    )
    ax.set_xlabel("Terminal-tree log likelihood")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title_prefix} vs log likelihood "
        f"({len(x):,} observed signatures; {samples:,} samples)"
    )
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(frameon=False, markerscale=5, loc="upper left")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_og_scatter(
    output: Path,
    *,
    log_model_probability: np.ndarray,
    log_target_reward: np.ndarray,
    max_points: int,
    seed: int,
    unique_signatures: int,
    title_prefix: str,
    reference_log_partition: float | None = None,
    axis_spec: dict[str, float] | None = None,
) -> tuple[float, float]:
    selected = select_scatter_points(
        len(log_model_probability),
        maximum=max_points,
        seed=seed,
    )
    x = log_target_reward[selected]
    y = log_model_probability[selected]
    estimated_log_partition = float(np.mean(log_target_reward - log_model_probability))
    line_log_partition = (
        reference_log_partition
        if reference_log_partition is not None
        else estimated_log_partition
    )
    pearson = pearson_vs_ideal_sampling(
        log_model_probability,
        log_target_reward,
        line_log_partition,
    )
    marker_size, alpha = scatter_style(len(selected))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=220, constrained_layout=True)
    ax.scatter(
        x,
        y,
        s=marker_size,
        alpha=alpha,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
        label=sample_legend_label(
            n_trajectories=len(log_model_probability),
            unique_signatures=unique_signatures,
            pearson=pearson,
        ),
    )
    if axis_spec is not None:
        line_x = np.linspace(
            axis_spec["log_reward_min"],
            axis_spec["log_reward_max"],
            200,
        )
    elif x.size == 1:
        center = float(x[0])
        line_x = np.linspace(center - 1.0, center + 1.0, 200)
    else:
        line_x = np.linspace(float(x.min()), float(x.max()), 200)
    ax.plot(
        line_x,
        line_x - line_log_partition,
        color=GRAY,
        linestyle="--",
        linewidth=1.3,
        label=ideal_line_label(line_log_partition, show_log_z=False),
    )
    if axis_spec is not None:
        ax.set_xlim(axis_spec["log_reward_min"], axis_spec["log_reward_max"])
    y_min, y_max = log_probability_limits(y, line_x, line_log_partition)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Log terminal reward: $\log R(x)=\log L(x)$")
    ax.set_ylabel("Pathwise implied log terminal probability")
    ax.set_title(f"{title_prefix}: terminal probability vs reward")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return pearson, estimated_log_partition


def select_scatter_points(size: int, *, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or size <= maximum:
        return np.arange(size)
    return np.random.default_rng(seed).choice(size, size=maximum, replace=False)


def scatter_style(displayed_points: int) -> tuple[float, float]:
    if displayed_points >= 500_000:
        return 16.0, 0.14
    if displayed_points >= 100_000:
        return 24.0, 0.20
    return 32.0, 0.35


def sample_legend_label(
    *,
    n_trajectories: int,
    unique_signatures: int,
    pearson: float,
) -> str:
    return (
        f"{n_trajectories:,} trajectories\n"
        f"{unique_signatures:,} unique signatures\n"
        f"Pearson r vs ideal={pearson:.4f}"
    )


def save_linear_probability_reward_plots(
    raw_output: Path,
    calibrated_output: Path,
    *,
    log_model_probability: np.ndarray,
    log_target_reward: np.ndarray,
    max_points: int,
    seed: int,
    unique_signatures: int,
    title_prefix: str,
    reference_log_partition: float | None = None,
    axis_spec: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Linear-axis counterpart of the og-style log-probability plot."""
    selected = select_scatter_points(
        len(log_model_probability),
        maximum=max_points,
        seed=seed,
    )

    selected_log_probability = log_model_probability[selected]
    selected_log_reward = log_target_reward[selected]
    model_probability = np.exp(selected_log_probability)
    target_reward = np.exp(selected_log_reward)
    estimated_log_partition = float(np.mean(log_target_reward - log_model_probability))
    line_log_partition = (
        reference_log_partition
        if reference_log_partition is not None
        else estimated_log_partition
    )
    partition = float(np.exp(line_log_partition))
    pearson = pearson_vs_ideal_sampling_linear(
        log_model_probability,
        log_target_reward,
        line_log_partition,
    )
    marker_size, alpha = scatter_style(len(selected))
    sample_label = sample_legend_label(
        n_trajectories=len(log_model_probability),
        unique_signatures=unique_signatures,
        pearson=pearson,
    )

    fig, ax = plt.subplots(figsize=(10, 10), dpi=220, constrained_layout=True)
    ax.scatter(
        target_reward,
        model_probability,
        s=marker_size,
        alpha=alpha,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
        label=sample_label,
    )
    if axis_spec is not None:
        line_x = np.linspace(axis_spec["reward_min"], axis_spec["reward_max"], 200)
    elif target_reward.size == 1:
        center = float(target_reward[0])
        line_x = np.linspace(center - 1.0, center + 1.0, 200)
    else:
        line_x = np.linspace(
            float(target_reward.min()),
            float(target_reward.max()),
            200,
        )
    ax.plot(
        line_x,
        line_x / partition,
        color=GRAY,
        linestyle="--",
        linewidth=1.3,
        label=ideal_line_label(line_log_partition, show_log_z=False),
    )
    if axis_spec is not None:
        ax.set_xlim(axis_spec["reward_min"], axis_spec["reward_max"])
    y_min, y_max = linear_probability_limits(model_probability, line_x, partition)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Terminal reward: $R(x)=L(x)$")
    ax.set_ylabel("Pathwise implied terminal probability")
    ax.set_title(f"{title_prefix}: terminal probability vs reward")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.savefig(raw_output, bbox_inches="tight")
    plt.close(fig)

    calibrated_probability = np.exp(selected_log_probability + estimated_log_partition)
    lower = min(float(calibrated_probability.min()), float(target_reward.min()))
    upper = max(float(calibrated_probability.max()), float(target_reward.max()))
    padding = 0.03 * (upper - lower)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=220, constrained_layout=True)
    ax.scatter(
        target_reward,
        calibrated_probability,
        s=marker_size,
        alpha=alpha,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
        label=sample_label,
    )
    ax.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        color=GRAY,
        linestyle="--",
        linewidth=1.3,
        label="ideal: reward = calibrated model probability",
    )
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Terminal reward: $R(x)=L(x)$")
    ax.set_ylabel(r"Partition-calibrated terminal probability: $\hat ZP(x)$")
    ax.set_title(f"{title_prefix}: calibrated probability vs reward")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.savefig(calibrated_output, bbox_inches="tight")
    plt.close(fig)
    return pearson, estimated_log_partition


def main() -> None:
    args = parse_args()
    if args.score_bins < 2:
        raise ValueError("--score-bins must be at least 2")
    if args.signature_top_k < 1:
        raise ValueError("--signature-top-k must be positive")

    reference_log_partition = args.reference_log_partition
    axis_spec = None
    reference_run_dir = None
    if args.shared_reference or args.reference_run_dir is not None:
        axis_spec = shared_reward_axis_bounds(args.reference_run_dir)

    metadata_path = args.samples.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    shift = float(metadata.get("log_score_shift", 3600.0))

    with np.load(args.samples) as payload:
        log_score = payload["log_score"].astype(np.float64)
        log_pf = payload["log_pf"].astype(np.float64)
        log_q_reverse = payload["log_q_reverse"].astype(np.float64)
        topology_index = payload["topology_index"].astype(np.int64)
        topology_ids = payload["topology_ids"].astype(str)
        log_likelihood = load_terminal_log_likelihood(payload, shift=shift)

    n = len(log_score)
    if not (len(log_pf) == len(log_q_reverse) == len(topology_index) == n):
        raise ValueError("sample arrays have inconsistent lengths")

    reference_weights, log_weights, ess = normalized_reference_weights(
        log_score,
        log_pf,
        log_q_reverse,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topology_counts, topology_model, topology_reference = aggregate(
        topology_index,
        len(topology_ids),
        reference_weights,
    )
    topology_order = save_rank_plot(
        args.output_dir / "topology_checkpoint_vs_reward_reference.png",
        model_frequency=topology_model,
        reference_frequency=topology_reference,
        title="Full model: topology mass versus shifted-linear reward reference",
        xlabel="Topology rank under reward reference",
    )

    original_log_likelihood = log_score - shift
    edges, centers, score_model, score_reference = save_score_plot(
        args.output_dir / "loglikelihood_checkpoint_vs_reward_reference.png",
        original_log_likelihood=original_log_likelihood,
        reference_weights=reference_weights,
        bins=args.score_bins,
    )

    rounded_score = np.rint(log_score * 1000.0).astype(np.int64)
    signature_pairs = np.empty(
        n,
        dtype=[("topology", np.int16), ("score_milli", np.int64)],
    )
    signature_pairs["topology"] = topology_index.astype(np.int16)
    signature_pairs["score_milli"] = rounded_score
    unique_signatures, signature_index = np.unique(
        signature_pairs,
        return_inverse=True,
    )
    signature_counts, signature_model, signature_reference = aggregate(
        signature_index,
        len(unique_signatures),
        reference_weights,
    )
    signature_log_likelihood = (
        unique_signatures["score_milli"].astype(np.float64) / 1000.0 - shift
    )
    sample_label = f"{n // 1000}k" if n % 1000 == 0 else str(n)
    signature_output = (
        args.output_dir / f"signature_qhat_vs_loglikelihood_{sample_label}.png"
    )
    save_signature_likelihood_plot(
        signature_output,
        signature_log_likelihood=signature_log_likelihood,
        model_frequency=signature_model,
        reference_frequency=signature_reference,
        samples=n,
        log_y=False,
    )
    save_signature_likelihood_plot(
        signature_output.with_name(
            f"{signature_output.stem}_logq{signature_output.suffix}"
        ),
        signature_log_likelihood=signature_log_likelihood,
        model_frequency=signature_model,
        reference_frequency=signature_reference,
        samples=n,
        log_y=True,
    )
    top_k = min(args.signature_top_k, len(unique_signatures))
    selected_signatures = np.argsort(-signature_reference)[:top_k]
    save_rank_plot(
        args.output_dir / "signature_checkpoint_vs_reward_reference_topk.png",
        model_frequency=signature_model[selected_signatures],
        reference_frequency=signature_reference[selected_signatures],
        title=f"Top {top_k} observed signatures by reward-reference mass",
        xlabel="Observed-signature reward rank",
        log_y=True,
    )

    log_model_probability = log_pf - log_q_reverse
    if axis_spec is not None:
        axis_spec = merge_reward_axis_bounds(
            axis_spec,
            reward=np.exp(log_likelihood),
            log_reward=log_likelihood,
        )
        reference_run_dir = axis_spec["reference_run_dir"]
    if args.plot_method == "ppo":
        title_prefix = "Full model"
    else:
        title_prefix = "Learned-reverse IPS-GRPO"

    pearson, estimated_log_partition = save_og_scatter(
        args.output_dir / "log_model_probability_vs_log_reward.png",
        log_model_probability=log_model_probability,
        log_target_reward=log_likelihood,
        max_points=args.scatter_points,
        seed=args.seed,
        unique_signatures=len(unique_signatures),
        title_prefix=title_prefix,
        reference_log_partition=reference_log_partition,
        axis_spec=axis_spec,
    )
    linear_pearson, estimated_log_partition = save_linear_probability_reward_plots(
        args.output_dir / "model_probability_vs_reward.png",
        args.output_dir / "partition_calibrated_model_probability_vs_reward.png",
        log_model_probability=log_model_probability,
        log_target_reward=log_likelihood,
        max_points=args.scatter_points,
        seed=args.seed,
        unique_signatures=len(unique_signatures),
        title_prefix=title_prefix,
        reference_log_partition=reference_log_partition,
        axis_spec=axis_spec,
    )

    np.savez_compressed(
        args.output_dir / "observed_support_reward_reference.npz",
        normalized_trajectory_weight=reference_weights.astype(np.float32),
        topology_model_frequency=topology_model,
        topology_reference_frequency=topology_reference,
        signature_topology_index=unique_signatures["topology"],
        signature_score_milli=unique_signatures["score_milli"],
        signature_model_count=signature_counts,
        signature_model_frequency=signature_model,
        signature_reference_frequency=signature_reference,
        loglikelihood_bin_edges=edges,
        loglikelihood_bin_centers=centers,
        loglikelihood_model_frequency=score_model,
        loglikelihood_reference_frequency=score_reference,
    )

    topology_rows = []
    for rank, index in enumerate(topology_order, start=1):
        topology_rows.append(
            {
                "rank": rank,
                "topology_id": str(topology_ids[index]),
                "checkpoint_count": int(topology_counts[index]),
                "checkpoint_frequency": float(topology_model[index]),
                "reference_frequency": float(topology_reference[index]),
                "reference_expected_count_at_n": float(topology_reference[index] * n),
            }
        )
    reference_summary = {
        "metadata": {
            **metadata,
            "samples_file": str(args.samples),
            "reference_method": "self-normalized R(x) q_phi(tau|x) / P_F(tau)",
            "samples": n,
            "unique_observed_signatures": int(len(unique_signatures)),
            "importance_ess": ess,
            "importance_ess_fraction": ess / n,
        },
        "topologies": topology_rows,
        "loglikelihood_bins": [
            {
                "left": float(edges[i]),
                "right": float(edges[i + 1]),
                "checkpoint_frequency": float(score_model[i]),
                "reference_frequency": float(score_reference[i]),
            }
            for i in range(len(centers))
        ],
    }
    (args.output_dir / "reward_reference_summary.json").write_text(
        json.dumps(reference_summary, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "samples": n,
        "plotted_samples": (
            n if args.scatter_points <= 0 else min(n, args.scatter_points)
        ),
        "observed_topologies": int(len(np.unique(topology_index))),
        "unique_observed_signatures": int(len(unique_signatures)),
        "importance_ess": ess,
        "importance_ess_fraction": ess / n,
        "log_weight_std": float(log_weights.std()),
        "log_weight_range": float(log_weights.max() - log_weights.min()),
        "topology_total_variation": float(
            0.5 * np.abs(topology_model - topology_reference).sum()
        ),
        "signature_total_variation_on_observed_support": float(
            0.5 * np.abs(signature_model - signature_reference).sum()
        ),
        "loglikelihood_histogram_total_variation": float(
            0.5 * np.abs(score_model - score_reference).sum()
        ),
        "log_model_probability_vs_log_reward_pearson_vs_ideal": pearson,
        "model_probability_vs_reward_pearson_vs_ideal": linear_pearson,
        "estimated_log_partition": estimated_log_partition,
    }
    if reference_log_partition is not None:
        metrics["reference_log_partition"] = reference_log_partition
    if reference_run_dir is not None:
        metrics["reference_run_dir"] = reference_run_dir
        metrics["shared_reward_axes"] = True
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    for path in sorted(args.output_dir.iterdir()):
        print(f"wrote {path.resolve()}")


if __name__ == "__main__":
    main()
