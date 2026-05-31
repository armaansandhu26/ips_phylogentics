#!/usr/bin/env python3
"""Sample terminal trees from trained policies and compare final performance."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grpo_experiments.eval_utils import (
    choose_device,
    entropy_from_counts,
    load_manifest,
    manifest_run_entries,
    resolve_run_artifacts,
    save_json,
    set_seed,
)
from src.gfn.rollout_worker_phylo import RolloutWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from trained checkpoints and compare final policy quality.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path, help="Matrix manifest JSON")
    src.add_argument("--run-dirs", nargs="+", type=Path, help="Run directories")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of equal log-score bins for the reward histogram.",
    )
    parser.add_argument(
        "--estimate-mll",
        action="store_true",
        help="Run GFNEvaluator marginal likelihood estimate (slow).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename inside each run dir (default: final_checkpoint.pt).",
    )
    return parser.parse_args()


def collect_run_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    if args.manifest is not None:
        manifest = load_manifest(args.manifest)
        for row in manifest_run_entries(manifest):
            label = row.get("label") or row.get("id") or Path(row["run_dir"]).name
            specs.append((label, Path(row["run_dir"])))
        return specs

    labels = args.labels or [path.name for path in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise ValueError("--labels count must match --run-dirs count")
    return list(zip(labels, args.run_dirs))


def sample_run(
    run_dir: Path,
    label: str,
    *,
    device: str,
    samples: int,
    batch_size: int,
    seed: int,
    checkpoint_name: str | None,
    estimate_mll: bool,
) -> dict[str, Any]:
    from grpo_experiments.eval_utils import load_generator

    set_seed(seed)
    artifacts = resolve_run_artifacts(str(run_dir), label=label)
    if checkpoint_name is not None:
        artifacts.checkpoint_path = artifacts.root / checkpoint_name
        if not artifacts.checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {artifacts.checkpoint_path}")

    cfg, env, generator = load_generator(artifacts, device)
    rollout_worker = RolloutWorker(env)

    trees = []
    log_scores = []
    log_rewards = []

    with np.errstate(all="ignore"):
        generated = 0
        while generated < samples:
            current_batch = min(batch_size, samples - generated)
            batch, trajectories = rollout_worker.rollout(
                generator,
                current_batch,
                generate_full_trajectories=True,
            )
            batch_trees = [traj.current_state.subtrees[0] for traj in trajectories]
            trees.extend(batch_trees)
            log_scores.extend(float(tree.log_score) for tree in batch_trees)
            log_rewards.extend(float(x) for x in batch["log_rewards"].detach().cpu().numpy())
            generated += current_batch

    signatures = [tree.signature for tree in trees]
    topology_ids = [tree.tree_topology_id for tree in trees]
    signature_counts = Counter(signatures)
    topology_counts = Counter(topology_ids)
    log_scores_arr = np.asarray(log_scores, dtype=np.float64)
    log_rewards_arr = np.asarray(log_rewards, dtype=np.float64)

    summary: dict[str, Any] = {
        "label": label,
        "run_dir": str(run_dir),
        "method": artifacts.method,
        "checkpoint_path": str(artifacts.checkpoint_path),
        "samples": int(len(trees)),
        "unique_signatures": int(len(signature_counts)),
        "unique_topologies": int(len(topology_counts)),
        "signature_entropy": entropy_from_counts(dict(signature_counts)),
        "topology_entropy": entropy_from_counts(dict(topology_counts)),
        "signature_duplicate_fraction": float(
            1.0 - len(signature_counts) / len(signatures)
        ) if signatures else 0.0,
        "topology_duplicate_fraction": float(
            1.0 - len(topology_counts) / len(topology_ids)
        ) if topology_ids else 0.0,
        "max_signature_share": float(max(signature_counts.values()) / len(signatures)),
        "max_topology_share": float(max(topology_counts.values()) / len(topology_ids)),
        "log_score_mean": float(log_scores_arr.mean()),
        "log_score_std": float(log_scores_arr.std()),
        "log_score_min": float(log_scores_arr.min()),
        "log_score_max": float(log_scores_arr.max()),
        "log_reward_mean": float(log_rewards_arr.mean()),
        "log_reward_std": float(log_rewards_arr.std()),
        "top_signatures": [
            {"id": key, "count": int(count), "share": float(count / len(signatures))}
            for key, count in signature_counts.most_common(20)
        ],
        "top_topologies": [
            {"id": key, "count": int(count), "share": float(count / len(topology_ids))}
            for key, count in topology_counts.most_common(20)
        ],
        "log_scores": log_scores_arr,
        "topology_ids": topology_ids,
        "signatures": signatures,
        "topology_counts": topology_counts,
        "signature_counts": signature_counts,
    }

    if estimate_mll:
        from src.gfn.gfn_evaluator import GFNEvaluator

        evaluator = GFNEvaluator(cfg.GFN.MODEL.EVALUATION, rollout_worker, generator)
        mll = evaluator.evaluate_marginal_likelihood(traj_size=1024)
        pearson = evaluator.evaluate_gfn_quality_pearsonr()[2]
        summary["marginal_likelihood"] = float(mll)
        summary["log_prob_reward_pearsonr"] = float(pearson)

    return summary


def sample_all_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = choose_device(args.device)
    specs = collect_run_specs(args)
    summaries = []
    for idx, (label, run_dir) in enumerate(specs):
        print(f"sampling {args.samples} trees from {label} ({run_dir}) on {device}")
        summaries.append(
            sample_run(
                run_dir,
                label,
                device=device,
                samples=args.samples,
                batch_size=args.batch_size,
                seed=args.seed + idx,
                checkpoint_name=args.checkpoint,
                estimate_mll=args.estimate_mll,
            )
        )
    return summaries


def rank_frequency(counter: Counter[str], top_k: int) -> np.ndarray:
    counts = np.asarray(
        [count for _, count in counter.most_common(top_k)],
        dtype=np.float64,
    )
    if len(counts) == 0:
        return counts
    return counts / counts.sum()


def compute_bin_edges(summaries: list[dict[str, Any]], n_bins: int) -> np.ndarray:
    combined = np.concatenate([row["log_scores"] for row in summaries])
    return np.linspace(combined.min(), combined.max(), n_bins + 1)


def compute_bin_frequencies(
    summaries: list[dict[str, Any]],
    bin_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    frequencies: dict[str, np.ndarray] = {}
    for row in summaries:
        hist, _ = np.histogram(row["log_scores"], bins=bin_edges)
        frequencies[row["label"]] = hist / len(row["log_scores"])
    return frequencies


def print_bin_table(
    summaries: list[dict[str, Any]],
    bin_edges: np.ndarray,
    frequencies: dict[str, np.ndarray],
) -> None:
    labels = [row["label"] for row in summaries]
    n_bins = len(bin_edges) - 1
    print(f"\n{'Bin':>24s}", end="")
    for label in labels:
        print(f"  {label:>20s}", end="")
    print()
    for idx in range(n_bins):
        print(
            f"  [{bin_edges[idx]:8.1f}, {bin_edges[idx + 1]:8.1f})",
            end="",
        )
        for label in labels:
            print(f"  {frequencies[label][idx]:20.4f}", end="")
        print()


def _draw_reward_bins(
    ax: plt.Axes,
    summaries: list[dict[str, Any]],
    bin_edges: np.ndarray,
    frequencies: dict[str, np.ndarray],
    cmap: plt.Colormap,
) -> None:
    labels = [row["label"] for row in summaries]
    n_bins = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    group_width = bin_width / (len(labels) + 1)

    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * group_width
        ax.bar(
            bin_centers + offset,
            frequencies[label],
            width=group_width * 0.95,
            color=cmap(idx % 10),
            alpha=0.85,
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(bin_centers)
    ax.set_xticklabels(
        [f"{bin_edges[i]:.0f}\n{bin_edges[i + 1]:.0f}" for i in range(n_bins)],
        fontsize=6,
    )
    ax.set_xlabel("Log Score Bin")
    ax.set_ylabel("Sampling Frequency")
    ax.set_title(f"Reward Distribution ({n_bins} Equal Bins)")
    ax.grid(True, alpha=0.2, axis="y")


def _draw_log_score_pdf(
    ax: plt.Axes,
    summaries: list[dict[str, Any]],
    cmap: plt.Colormap,
    *,
    bins: int = 35,
) -> None:
    for idx, row in enumerate(summaries):
        ax.hist(
            row["log_scores"],
            bins=bins,
            alpha=0.45,
            density=True,
            label=row["label"],
            color=cmap(idx % 10),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_title("Log Score Sampling PDF")
    ax.set_xlabel("Log Score")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)


def _draw_log_score_cdf(
    ax: plt.Axes,
    summaries: list[dict[str, Any]],
    cmap: plt.Colormap,
) -> None:
    for idx, row in enumerate(summaries):
        values = np.sort(row["log_scores"])
        y = np.arange(1, len(values) + 1, dtype=np.float64) / len(values)
        ax.plot(values, y, label=row["label"], color=cmap(idx % 10), linewidth=2.0)
    ax.set_title("Log Score Empirical CDF")
    ax.set_xlabel("Log Score")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.25)


def _draw_rank_frequency(
    ax: plt.Axes,
    summaries: list[dict[str, Any]],
    *,
    count_key: str,
    title: str,
    top_k: int,
    cmap: plt.Colormap,
) -> None:
    for idx, row in enumerate(summaries):
        shares = rank_frequency(row[count_key], top_k)
        ranks = np.arange(1, len(shares) + 1)
        ax.plot(
            ranks,
            shares,
            marker="o",
            markersize=3,
            color=cmap(idx % 10),
            linewidth=1.6,
            label=row["label"],
        )
    ax.set_title(title)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Share Within Top-K")
    ax.grid(True, alpha=0.25)


def _draw_metric_bar(
    ax: plt.Axes,
    summaries: list[dict[str, Any]],
    metric_key: str,
    title: str,
    cmap: plt.Colormap,
) -> None:
    labels = [row["label"] for row in summaries]
    x = np.arange(len(labels))
    values = [float(row[metric_key]) for row in summaries]
    bars = ax.bar(x, values, color=[cmap(i % 10) for i in range(len(labels))])
    ax.set_title(title, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=6,
        )


def plot_sampling_comparison(
    summaries: list[dict[str, Any]],
    bin_edges: np.ndarray,
    frequencies: dict[str, np.ndarray],
    output_path: Path,
    top_k: int,
    *,
    samples: int,
    n_bins: int,
) -> None:
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(18, 22), dpi=200)
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[1.15, 1.0, 1.0, 0.95],
        hspace=0.38,
        wspace=0.32,
    )

    ax_bins = fig.add_subplot(gs[0, :])
    ax_pdf = fig.add_subplot(gs[1, 0:2])
    ax_cdf = fig.add_subplot(gs[1, 2:4])
    ax_topo = fig.add_subplot(gs[2, 0:2])
    ax_sig = fig.add_subplot(gs[2, 2:4])
    ax_unique = fig.add_subplot(gs[3, 0])
    ax_entropy = fig.add_subplot(gs[3, 1])
    ax_mean = fig.add_subplot(gs[3, 2])
    ax_dup = fig.add_subplot(gs[3, 3])

    _draw_reward_bins(ax_bins, summaries, bin_edges, frequencies, cmap)
    _draw_log_score_pdf(ax_pdf, summaries, cmap)
    _draw_log_score_cdf(ax_cdf, summaries, cmap)
    _draw_rank_frequency(
        ax_topo,
        summaries,
        count_key="topology_counts",
        title=f"Top-{top_k} Topology Rank-Frequency",
        top_k=top_k,
        cmap=cmap,
    )
    _draw_rank_frequency(
        ax_sig,
        summaries,
        count_key="signature_counts",
        title=f"Top-{top_k} Signature Rank-Frequency",
        top_k=top_k,
        cmap=cmap,
    )
    _draw_metric_bar(ax_unique, summaries, "unique_topologies", "Unique Topologies", cmap)
    _draw_metric_bar(ax_entropy, summaries, "topology_entropy", "Topology Entropy", cmap)
    _draw_metric_bar(ax_mean, summaries, "log_score_mean", "Mean Log Score", cmap)
    _draw_metric_bar(
        ax_dup,
        summaries,
        "topology_duplicate_fraction",
        "Topology Duplicate Fraction",
        cmap,
    )

    handles, legend_labels = ax_cdf.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        f"Sampling comparison ({samples} trees/run, {n_bins} reward bins)",
        fontsize=15,
        y=1.01,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def json_ready_summary(
    row: dict[str, Any],
    *,
    bin_edges: np.ndarray | None = None,
    bin_frequencies: np.ndarray | None = None,
) -> dict[str, Any]:
    out = dict(row)
    for key in ("log_scores", "topology_ids", "signatures", "topology_counts", "signature_counts"):
        out.pop(key, None)
    if bin_edges is not None and bin_frequencies is not None:
        out["reward_bin_frequencies"] = {
            "edges": [float(x) for x in bin_edges],
            "frequencies": [float(x) for x in bin_frequencies],
        }
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = sample_all_runs(args)
    bin_edges = compute_bin_edges(summaries, args.n_bins)
    frequencies = compute_bin_frequencies(summaries, bin_edges)

    payload = {
        "metadata": {
            "samples_per_run": args.samples,
            "batch_size": args.batch_size,
            "seed_base": args.seed,
            "n_bins": args.n_bins,
            "estimate_mll": args.estimate_mll,
            "reward_bin_edges": [float(x) for x in bin_edges],
        },
        "runs": [
            json_ready_summary(
                row,
                bin_edges=bin_edges,
                bin_frequencies=frequencies[row["label"]],
            )
            for row in summaries
        ],
    }
    save_json(args.output_dir / "sampling_summary.json", payload)

    plot_path = args.output_dir / "sampling_comparison.png"
    plot_sampling_comparison(
        summaries,
        bin_edges,
        frequencies,
        plot_path,
        args.top_k,
        samples=args.samples,
        n_bins=args.n_bins,
    )

    print_bin_table(summaries, bin_edges, frequencies)

    print(f"\nsaved outputs under: {args.output_dir}")
    print("  sampling_summary.json")
    print("  sampling_comparison.png")
    for row in summaries:
        print(
            f"  {row['label']}: unique_topo={row['unique_topologies']} "
            f"topo_dup={row['topology_duplicate_fraction']:.3f} "
            f"log_score_mean={row['log_score_mean']:.2f} "
            f"log_score_best={row['log_score_max']:.2f}"
        )


if __name__ == "__main__":
    main()
