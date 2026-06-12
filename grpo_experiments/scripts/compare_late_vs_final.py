#!/usr/bin/env python3
"""Side-by-side comparison of training trees vs final checkpoint sampling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grpo_experiments.eval_utils import choose_device, load_json, save_json
from grpo_experiments.scripts.compare_sampling import (
    _draw_single_distribution_panel,
    _is_collapsed_distribution,
    load_scores_cache,
    sample_run,
    save_scores_cache,
)
from grpo_experiments.scripts.compare_training_samples import (
    build_summary,
    subsample_trajectory_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--global-step-min",
        type=int,
        default=None,
        help="Optional lower global-step bound (inclusive). Default: entire run.",
    )
    parser.add_argument(
        "--global-step-max",
        type=int,
        default=None,
        help="Optional upper global-step bound (inclusive). Default: entire run.",
    )
    parser.add_argument(
        "--sampling-dir",
        type=Path,
        default=None,
        help="Directory with sampling_summary.json (+ optional sampling_scores.npz).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def load_final_summaries(
    args: argparse.Namespace,
    labels: list[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    scores_by_label: dict[str, np.ndarray] = {}

    if args.sampling_dir is not None:
        summary_path = args.sampling_dir / "sampling_summary.json"
        scores_path = args.sampling_dir / "sampling_scores.npz"
        if summary_path.exists():
            payload = load_json(summary_path)
            if scores_path.exists():
                scores_by_label = load_scores_cache(scores_path)
            for row in payload.get("runs", []):
                label = row["label"]
                if label in scores_by_label:
                    summaries.append({**row, "log_scores": scores_by_label[label]})

    if len(summaries) == len(labels):
        return summaries

    device = choose_device(args.device)
    summaries = []
    for idx, (label, run_dir) in enumerate(zip(labels, args.run_dirs)):
        print(f"sampling {args.samples} final trees from {label} ({run_dir}) on {device}")
        summaries.append(
            sample_run(
                run_dir,
                label,
                device=device,
                samples=args.samples,
                batch_size=args.batch_size,
                seed=args.seed + idx,
                checkpoint_name=args.checkpoint,
                estimate_mll=False,
            )
        )
    return summaries


def training_window_label(step_min: int | None, step_max: int | None) -> str:
    if step_min is None and step_max is None:
        return "Full training (all steps)"
    lo = step_min if step_min is not None else 0
    hi = step_max if step_max is not None else "end"
    return f"Training (steps {lo:,}–{hi})"


def load_training_summaries(
    args: argparse.Namespace,
    labels: list[str],
) -> list[dict[str, Any]]:
    window = training_window_label(args.global_step_min, args.global_step_max)
    summaries: list[dict[str, Any]] = []
    for idx, (label, run_dir) in enumerate(zip(labels, args.run_dirs)):
        print(
            f"subsample {args.samples} training trees ({window}) from {label}",
            end="",
        )
        rows = subsample_trajectory_rows(
            run_dir / "trajectory_samples.jsonl",
            args.samples,
            seed=args.seed + idx,
            step_min=args.global_step_min,
            step_max=args.global_step_max,
        )
        print(f" -> got {len(rows)}")
        summaries.append(build_summary(rows, label=label, run_dir=run_dir))
    return summaries


def _shared_log_score_range(
    late: dict[str, Any],
    final: dict[str, Any],
) -> tuple[float, float]:
    combined = np.concatenate([late["log_scores"], final["log_scores"]])
    return float(combined.min()), float(combined.max())


def plot_side_by_side_grid(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    output_path: Path,
    *,
    training_title: str,
) -> None:
    cmap = plt.get_cmap("tab10")
    n_rows = len(pairs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.6 * n_rows), dpi=200)
    if n_rows == 1:
        axes = np.asarray([axes])

    for row_idx, (label, late, final) in enumerate(pairs):
        color = cmap(row_idx % 10)
        shared_range = _shared_log_score_range(late, final)
        late_panel = {**late, "label": f"{label}\ntraining"}
        final_panel = {**final, "label": f"{label}\nfinal sampling"}

        _draw_single_distribution_panel(
            axes[row_idx, 0],
            late_panel,
            color=color,
            global_range=shared_range
            if not _is_collapsed_distribution(late["log_scores"])
            else None,
        )
        _draw_single_distribution_panel(
            axes[row_idx, 1],
            final_panel,
            color=color,
            global_range=shared_range
            if not _is_collapsed_distribution(final["log_scores"])
            else None,
        )
        axes[row_idx, 0].set_ylabel(label, fontsize=9, fontweight="bold")

    axes[0, 0].set_title(training_title, fontsize=11)
    axes[0, 1].set_title("Final checkpoint sampling", fontsize=11)
    fig.suptitle(
        "Training vs final sampling — per-method score distributions",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_paired_overlay(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    output_path: Path,
    *,
    training_title: str,
    bins: int = 50,
) -> None:
    cmap = plt.get_cmap("tab10")
    n_cols = min(3, len(pairs))
    n_rows = int(np.ceil(len(pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), dpi=200)
    axes = np.atleast_2d(axes)

    for idx, (label, late, final) in enumerate(pairs):
        ax = axes[idx // n_cols, idx % n_cols]
        combined = np.concatenate([late["log_scores"], final["log_scores"]])
        edges = np.linspace(float(combined.min()), float(combined.max()), bins + 1)
        color = cmap(idx % 10)

        ax.hist(
            late["log_scores"],
            bins=edges,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=color,
            label=(
                f"train (mean={late['log_score_mean']:.1f}, "
                f"topo={late['unique_topologies']})"
            ),
        )
        ax.hist(
            late["log_scores"],
            bins=edges,
            density=True,
            alpha=0.12,
            color=color,
            edgecolor="none",
        )
        ax.hist(
            final["log_scores"],
            bins=edges,
            density=True,
            histtype="step",
            linewidth=2.2,
            color="0.25",
            linestyle="--",
            label=(
                f"final (mean={final['log_score_mean']:.1f}, "
                f"topo={final['unique_topologies']})"
            ),
        )
        ax.hist(
            final["log_scores"],
            bins=edges,
            density=True,
            alpha=0.10,
            color="0.25",
            edgecolor="none",
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Log Score")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)

    for idx in range(len(pairs), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.suptitle(
        f"Training vs final sampling overlay ({training_title.lower()})",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_side_by_side(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    output_path: Path,
    *,
    training_title: str,
) -> None:
    metrics = [
        ("unique_topologies", "Unique Topologies"),
        ("log_score_mean", "Mean Log Score"),
        ("topology_entropy", "Topology Entropy"),
        ("topology_duplicate_fraction", "Topology Dup. Fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=200)
    cmap = plt.get_cmap("tab10")
    x = np.arange(len(pairs))
    width = 0.34

    for ax, (metric_key, title) in zip(axes.flatten(), metrics):
        for offset, phase, alpha in [(-width / 2, "train", 0.9), (width / 2, "final", 0.55)]:
            values = [
                float(late[metric_key] if phase == "train" else final[metric_key])
                for _, late, final in pairs
            ]
            colors = [cmap(i % 10) for i in range(len(pairs))]
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=phase,
                color=colors,
                alpha=alpha,
                edgecolor="white",
                linewidth=0.6,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _, _ in pairs], rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        if metric_key == metrics[0][0]:
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"{training_title} vs final sampling metrics",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_three_way_overlay(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    output_path: Path,
    *,
    training_title: str,
) -> None:
    """All methods: training overlay (left) vs final sampling overlay (right)."""
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=200, sharey=True)

    for side, ax, summaries, title in [
        (0, axes[0], [late for _, late, _ in pairs], training_title),
        (1, axes[1], [final for _, _, final in pairs], "Final checkpoint sampling"),
    ]:
        combined = np.concatenate([row["log_scores"] for row in summaries])
        edges = np.linspace(float(combined.min()), float(combined.max()), 60 + 1)
        for idx, row in enumerate(summaries):
            scores = row["log_scores"]
            color = cmap(idx % 10)
            label = (
                f"{row['label']}  "
                f"(mean={scores.mean():.1f}, std={scores.std():.2f}, "
                f"topo={row['unique_topologies']})"
            )
            ax.hist(scores, bins=edges, density=True, histtype="step", linewidth=2.0, color=color, label=label)
            ax.hist(scores, bins=edges, density=True, alpha=0.12, color=color, edgecolor="none")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Log Score")
        if side == 0:
            ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Training vs final sampling — three-way overlay", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_report(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    step_min: int,
    step_max: int,
    samples: int,
) -> dict[str, Any]:
    runs = []
    for label, late, final in pairs:
        runs.append(
            {
                "label": label,
                "training_trees": {
                    "global_step_min": step_min,
                    "global_step_max": step_max,
                    "samples": late["samples"],
                    "unique_topologies": late["unique_topologies"],
                    "unique_signatures": late["unique_signatures"],
                    "topology_entropy": late["topology_entropy"],
                    "topology_duplicate_fraction": late["topology_duplicate_fraction"],
                    "log_score_mean": late["log_score_mean"],
                    "log_score_std": late["log_score_std"],
                    "log_score_min": late["log_score_min"],
                    "log_score_max": late["log_score_max"],
                },
                "final_sampling": {
                    "samples": final["samples"],
                    "unique_topologies": final["unique_topologies"],
                    "unique_signatures": final["unique_signatures"],
                    "topology_entropy": final["topology_entropy"],
                    "topology_duplicate_fraction": final["topology_duplicate_fraction"],
                    "log_score_mean": final["log_score_mean"],
                    "log_score_std": final["log_score_std"],
                    "log_score_min": final["log_score_min"],
                    "log_score_max": final["log_score_max"],
                },
                "delta_final_minus_training": {
                    "unique_topologies": final["unique_topologies"] - late["unique_topologies"],
                    "log_score_mean": final["log_score_mean"] - late["log_score_mean"],
                    "topology_entropy": final["topology_entropy"] - late["topology_entropy"],
                },
            }
        )
    return {
        "metadata": {
            "samples_per_side": samples,
            "global_step_min": step_min,
            "global_step_max": step_max,
        },
        "runs": runs,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels or [path.name for path in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise ValueError("--labels count must match --run-dirs count")

    training_title = training_window_label(args.global_step_min, args.global_step_max)

    training_summaries = load_training_summaries(args, labels)
    final_summaries = load_final_summaries(args, labels)
    if len(final_summaries) != len(training_summaries):
        raise RuntimeError("training/final summary count mismatch")

    label_by_name = {row["label"]: row for row in final_summaries}
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for training in training_summaries:
        final = label_by_name[training["label"]]
        pairs.append((training["label"], training, final))

    plot_side_by_side_grid(
        pairs,
        args.output_dir / "training_vs_final_grid.png",
        training_title=training_title,
    )
    plot_paired_overlay(
        pairs,
        args.output_dir / "training_vs_final_paired_overlay.png",
        training_title=training_title,
    )
    plot_metrics_side_by_side(
        pairs,
        args.output_dir / "training_vs_final_metrics.png",
        training_title=training_title,
    )
    plot_three_way_overlay(
        pairs,
        args.output_dir / "training_vs_final_three_way_overlay.png",
        training_title=training_title,
    )

    report = build_report(
        pairs,
        step_min=args.global_step_min,
        step_max=args.global_step_max,
        samples=args.samples,
    )
    save_json(args.output_dir / "training_vs_final_summary.json", report)

    print(f"\nsaved outputs under: {args.output_dir}")
    for path in sorted(args.output_dir.glob("*")):
        print(f"  {path.name}")
    print()
    for label, training, final in pairs:
        print(
            f"{label}: "
            f"train topo={training['unique_topologies']} mean={training['log_score_mean']:.2f} | "
            f"final topo={final['unique_topologies']} mean={final['log_score_mean']:.2f} | "
            f"Δmean={final['log_score_mean'] - training['log_score_mean']:+.2f} "
            f"Δtopo={final['unique_topologies'] - training['unique_topologies']:+d}"
        )


if __name__ == "__main__":
    main()
