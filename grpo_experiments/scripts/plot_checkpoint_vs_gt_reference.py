#!/usr/bin/env python3
"""Plot checkpoint versus finite-sample GT in the og_code sampling style."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-samples", type=Path, required=True)
    parser.add_argument("--gt-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def count_formatter(samples: int):
    def formatter(value: float, _position: float) -> str:
        return f"{int(round(value * samples))}/{samples // 1000}k"

    return FuncFormatter(formatter)


def save_log_likelihood_plot(
    output: Path,
    *,
    log_likelihoods: np.ndarray,
    exact_frequency: np.ndarray,
    gt_frequency: np.ndarray,
    checkpoint_frequency: np.ndarray,
    gt_total: int,
    checkpoint_total: int,
    log_y: bool,
) -> None:
    order = np.argsort(log_likelihoods)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=220, constrained_layout=True)
    ax.plot(
        log_likelihoods[order],
        exact_frequency[order],
        linestyle="--",
        color="0.35",
        linewidth=1.8,
        label="exact q* proportional to R(x)",
        zorder=1,
    )

    if log_y:
        floor = 0.5 / max(gt_total, checkpoint_total)
        gt_values = np.maximum(gt_frequency, floor)
        checkpoint_values = np.maximum(checkpoint_frequency, floor)
        ax.set_yscale("log")
        ylabel = "log empirical mass"
        title_prefix = "Log per-topology empirical mass"
    else:
        gt_values = gt_frequency
        checkpoint_values = checkpoint_frequency
        ylabel = "q_hat(x)"
        title_prefix = "Per-topology empirical mass"
        if gt_total == checkpoint_total:
            ax.yaxis.set_major_formatter(count_formatter(gt_total))

    ax.scatter(
        log_likelihoods,
        gt_values,
        marker="o",
        s=42,
        color="#777777",
        alpha=0.85,
        label=f"GT empirical ({np.count_nonzero(gt_frequency)} topologies)",
        zorder=2,
    )
    ax.scatter(
        log_likelihoods,
        checkpoint_values,
        marker="x",
        s=52,
        linewidths=1.6,
        color="#1976d2",
        label=(
            "checkpoint "
            f"({np.count_nonzero(checkpoint_frequency)} topologies)"
        ),
        zorder=3,
    )
    ax.set_xlabel("Terminal-tree log likelihood")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title_prefix} vs log likelihood "
        f"({checkpoint_total:,} checkpoint / {gt_total:,} GT samples)"
    )
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_rank_plot(
    output: Path,
    *,
    ranks: np.ndarray,
    gt_frequency: np.ndarray,
    checkpoint_frequency: np.ndarray,
    gt_total: int,
    checkpoint_total: int,
) -> None:
    positive = np.concatenate(
        [
            checkpoint_frequency[checkpoint_frequency > 0],
            gt_frequency[gt_frequency > 0],
        ]
    )
    floor = max(float(positive.min()) / 2.0, 1e-9)
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=220, constrained_layout=True)
    ax.scatter(
        ranks,
        np.maximum(gt_frequency, floor),
        marker="o",
        s=34,
        color="#777777",
        alpha=0.85,
        label=(
            f"GT empirical: {np.count_nonzero(gt_frequency)}/105 "
            f"({gt_total:,} samples)"
        ),
    )
    ax.scatter(
        ranks,
        np.maximum(checkpoint_frequency, floor),
        marker="x",
        s=46,
        linewidths=1.5,
        color="#1976d2",
        label=(
            f"checkpoint: {np.count_nonzero(checkpoint_frequency)}/105 "
            f"({checkpoint_total:,} samples)"
        ),
    )
    ax.axhline(floor, color="0.75", linestyle=":", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlim(0.5, len(ranks) + 0.5)
    ax.set_xlabel("Topology reward rank")
    ax.set_ylabel("Empirical frequency (log scale)")
    ax.set_title("Checkpoint versus GT empirical mass across all 105 topologies")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False, loc="best")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sample_payload = json.loads(args.checkpoint_samples.read_text())
    gt_payload = json.loads(args.gt_reference.read_text())
    trees = sample_payload["trees"]
    gt_rows = gt_payload["topologies"]
    checkpoint_total = len(trees)
    gt_total = int(gt_payload["metadata"]["samples"])

    checkpoint_counts = Counter(str(tree["tree_topology_id"]) for tree in trees)
    ranks = np.asarray([int(row["rank"]) for row in gt_rows])
    exact_frequency = np.asarray(
        [float(row["expected_frequency"]) for row in gt_rows],
        dtype=np.float64,
    )
    gt_frequency = np.asarray(
        [int(row["sampled_count"]) / gt_total for row in gt_rows],
        dtype=np.float64,
    )
    checkpoint_frequency = np.asarray(
        [
            checkpoint_counts.get(str(row["topology_id"]), 0) / checkpoint_total
            for row in gt_rows
        ],
        dtype=np.float64,
    )
    log_likelihoods = np.asarray(
        [
            float(
                row.get(
                    "terminal_log_likelihood",
                    float(row["log_score"])
                    - float(gt_payload["metadata"].get("model_log_score_shift", 0.0)),
                )
            )
            for row in gt_rows
        ],
        dtype=np.float64,
    )

    save_log_likelihood_plot(
        args.output,
        log_likelihoods=log_likelihoods,
        exact_frequency=exact_frequency,
        gt_frequency=gt_frequency,
        checkpoint_frequency=checkpoint_frequency,
        gt_total=gt_total,
        checkpoint_total=checkpoint_total,
        log_y=False,
    )
    log_output = args.output.with_name(f"{args.output.stem}_logq{args.output.suffix}")
    save_log_likelihood_plot(
        log_output,
        log_likelihoods=log_likelihoods,
        exact_frequency=exact_frequency,
        gt_frequency=gt_frequency,
        checkpoint_frequency=checkpoint_frequency,
        gt_total=gt_total,
        checkpoint_total=checkpoint_total,
        log_y=True,
    )
    rank_output = args.output.with_name(f"{args.output.stem}_rank{args.output.suffix}")
    save_rank_plot(
        rank_output,
        ranks=ranks,
        gt_frequency=gt_frequency,
        checkpoint_frequency=checkpoint_frequency,
        gt_total=gt_total,
        checkpoint_total=checkpoint_total,
    )

    metrics = {
        "reward_target": gt_payload["metadata"].get("reward_target", "unknown"),
        "checkpoint_samples": checkpoint_total,
        "gt_samples": gt_total,
        "checkpoint_observed_topologies": int(
            np.count_nonzero(checkpoint_frequency)
        ),
        "gt_observed_topologies": int(np.count_nonzero(gt_frequency)),
        "checkpoint_vs_exact": {
            "total_variation": float(
                0.5 * np.abs(checkpoint_frequency - exact_frequency).sum()
            ),
            "rmse": float(
                np.sqrt(np.square(checkpoint_frequency - exact_frequency).mean())
            ),
            "mae": float(np.abs(checkpoint_frequency - exact_frequency).mean()),
        },
        "checkpoint_vs_gt_empirical": {
            "total_variation": float(
                0.5 * np.abs(checkpoint_frequency - gt_frequency).sum()
            ),
            "rmse": float(
                np.sqrt(np.square(checkpoint_frequency - gt_frequency).mean())
            ),
            "mae": float(np.abs(checkpoint_frequency - gt_frequency).mean()),
        },
    }
    metrics_output = args.output.with_name(
        f"{args.output.stem}_metrics.json"
    )
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {log_output.resolve()}")
    print(f"wrote {rank_output.resolve()}")
    print(f"wrote {metrics_output.resolve()}")


if __name__ == "__main__":
    main()
