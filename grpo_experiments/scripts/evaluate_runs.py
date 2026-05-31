#!/usr/bin/env python3
"""Plot training metrics and diversity curves for one or more experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grpo_experiments.eval_utils import (
    load_manifest,
    load_metrics,
    manifest_run_entries,
    resolve_run_artifacts,
    sample_series,
    save_json,
)


METRIC_SPECS = [
    ("loss", "Loss", True),
    ("mean_log_reward", "Mean Log Reward", False),
    ("global_duplicate_fraction", "Global Duplicate Fraction (outcomes)", False),
    ("global_duplicate_topology_fraction", "Global Duplicate Topology Fraction", False),
    ("cumulative_unique_outcomes", "Cumulative Unique Outcomes", False),
    ("batch_duplicate_topology_fraction", "Batch Duplicate Topology Fraction", False),
]

OPTIONAL_METRICS = [
    ("mean_importance_ratio", "Mean Importance Ratio (pi_new/pi_old)", False),
    ("ips_prob_mean", "IPS Mean p_hat", False),
    ("ips_scaled_reward_mean", "IPS Scaled Reward Mean", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate training runs from metrics.jsonl and plot diversity curves.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--manifest",
        type=Path,
        help="Matrix manifest JSON written by run_sanity_matrix.sh",
    )
    src.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        help="One or more run directories containing metrics.jsonl",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels aligned with --run-dirs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for plots and evaluation_summary.json",
    )
    parser.add_argument("--smoothing-window", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def collect_runs(args: argparse.Namespace) -> list[tuple[str, Path, list[dict]]]:
    runs: list[tuple[str, Path, list[dict]]] = []
    if args.manifest is not None:
        manifest = load_manifest(args.manifest)
        for row in manifest_run_entries(manifest):
            label = row.get("label") or row.get("id") or Path(row["run_dir"]).name
            run_dir = Path(row["run_dir"])
            runs.append((label, run_dir, load_metrics(run_dir)))
        return runs

    labels = args.labels or [path.name for path in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise ValueError("--labels count must match --run-dirs count")
    for label, run_dir in zip(labels, args.run_dirs):
        runs.append((label, run_dir, load_metrics(run_dir)))
    return runs


def available_metric_specs(rows_list: list[list[dict]]) -> list[tuple[str, str, bool]]:
    specs = list(METRIC_SPECS)
    keys_present = {key for rows in rows_list for row in rows for key in row}
    for key, title, log_y in OPTIONAL_METRICS:
        if key in keys_present:
            specs.append((key, title, log_y))
    return specs


def plot_training_curves(
    runs: list[tuple[str, Path, list[dict]]],
    output_path: Path,
    metric_specs: list[tuple[str, str, bool]],
    smoothing_window: int,
    stride: int,
) -> None:
    n_cols = 2
    n_rows = (len(metric_specs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), dpi=220, constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    cmap = plt.get_cmap("tab10")

    for ax_idx, (key, title, use_log_y) in enumerate(metric_specs):
        ax = axes_flat[ax_idx]
        plotted = False
        for run_idx, (label, _, rows) in enumerate(runs):
            if key not in rows[0]:
                continue
            steps, values = sample_series(rows, key, smoothing_window, stride)
            ax.plot(
                steps,
                values,
                label=label,
                color=cmap(run_idx % 10),
                linewidth=1.8,
            )
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("Global Step")
        ax.grid(True, alpha=0.25)
        if use_log_y and plotted:
            positive = [v for v in ax.lines[0].get_ydata() if v > 0] if ax.lines else []
            if positive:
                ax.set_yscale("log")
        if plotted and ax_idx == 0:
            ax.legend(frameon=False, fontsize=8, loc="best")

    for ax in axes_flat[len(metric_specs):]:
        ax.axis("off")

    fig.suptitle("Training and diversity metrics", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def build_summary(runs: list[tuple[str, Path, list[dict]]]) -> dict:
    summary_runs = []
    for label, run_dir, rows in runs:
        last = rows[-1]
        first = rows[0]
        try:
            artifacts = resolve_run_artifacts(str(run_dir), label=label)
            method = artifacts.method
            checkpoint = str(artifacts.checkpoint_path)
        except Exception:
            method = last.get("method", "unknown")
            checkpoint = str(run_dir / "final_checkpoint.pt")

        summary_runs.append({
            "label": label,
            "run_dir": str(run_dir),
            "method": method,
            "checkpoint": checkpoint,
            "global_steps": int(last.get("global_step", len(rows) - 1)),
            "final": {
                "loss": last.get("loss"),
                "mean_log_reward": last.get("mean_log_reward"),
                "global_duplicate_fraction": last.get("global_duplicate_fraction"),
                "global_duplicate_topology_fraction": last.get("global_duplicate_topology_fraction"),
                "cumulative_unique_outcomes": last.get("cumulative_unique_outcomes"),
                "global_unique_outcomes": last.get("global_unique_outcomes"),
                "global_unique_topologies": last.get("global_unique_topologies"),
            },
            "initial": {
                "global_duplicate_fraction": first.get("global_duplicate_fraction"),
                "cumulative_unique_outcomes": first.get("cumulative_unique_outcomes"),
            },
        })
    return {"runs": summary_runs}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(args)
    metric_specs = available_metric_specs([rows for _, _, rows in runs])

    plot_path = args.output_dir / "training_curves.png"
    plot_training_curves(
        runs,
        plot_path,
        metric_specs,
        args.smoothing_window,
        args.stride,
    )

    summary = build_summary(runs)
    summary_path = args.output_dir / "evaluation_summary.json"
    save_json(summary_path, summary)

    print(f"saved training plot to: {plot_path}")
    print(f"saved summary to: {summary_path}")
    for row in summary["runs"]:
        final = row["final"]
        print(
            f"  {row['label']}: log_R={final.get('mean_log_reward')} "
            f"global_dup={final.get('global_duplicate_fraction')} "
            f"unique_outcomes={final.get('cumulative_unique_outcomes')}"
        )


if __name__ == "__main__":
    main()
