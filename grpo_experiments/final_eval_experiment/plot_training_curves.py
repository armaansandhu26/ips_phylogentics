#!/usr/bin/env python3
"""Plot convergence and training diagnostics for final-eval follow-up runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "training_curves"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=500,
        help="Centered moving-average window in training steps.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2500,
        help="Maximum plotted points per series after subsampling.",
    )
    return parser.parse_args()


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values[:]
    half = window // 2
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        smoothed.append((prefix[end] - prefix[start]) / (end - start))
    return smoothed


def subsample_xy(
    steps: list[int],
    values: list[float],
    *,
    max_points: int,
) -> tuple[list[int], list[float]]:
    if len(steps) <= max_points:
        return steps, values
    stride = max(1, (len(steps) + max_points - 1) // max_points)
    out_steps = steps[::stride]
    out_values = values[::stride]
    if out_steps[-1] != steps[-1]:
        out_steps.append(steps[-1])
        out_values.append(values[-1])
    return out_steps, out_values


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def load_metrics(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.jsonl"
    rows = []
    with metrics_path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metrics rows found in {metrics_path}")
    return rows


def label_for_run(row: dict) -> str:
    method = row.get("method")
    if method == "phylgfn":
        return "PhyloGFN"
    pfloor = row.get("ips_prob_floor")
    if method == "hybrid_ips_grpo" and pfloor is not None:
        return f"hyb_ips p={pfloor:g}"
    return row.get("id", Path(row["run_dir"]).name)


def color_map(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {label: cmap(idx % 10) for idx, label in enumerate(labels)}


def running_best(values: list[float]) -> list[float]:
    best = []
    current = float("-inf")
    for value in values:
        current = max(current, value)
        best.append(current)
    return best


def plot_series(
    ax: plt.Axes,
    steps: list[int],
    values: list[float],
    *,
    label: str,
    color,
    smooth_window: int,
    max_points: int,
    raw_alpha: float = 0.12,
    linewidth: float = 1.8,
) -> None:
    smooth = moving_average(values, smooth_window)
    raw_steps, raw_values = subsample_xy(steps, values, max_points=max_points)
    smooth_steps, smooth_values = subsample_xy(steps, smooth, max_points=max_points)
    ax.plot(raw_steps, raw_values, color=color, alpha=raw_alpha, linewidth=0.7)
    ax.plot(smooth_steps, smooth_values, color=color, linewidth=linewidth, label=label)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_summary(runs: list[dict], smooth_window: int) -> dict:
    summary_runs = []
    for run in runs:
        rows = run["rows"]
        rewards = [float(r["mean_log_reward"]) for r in rows]
        unique_topo = [float(r.get("batch_unique_topologies", np.nan)) for r in rows]
        grad_norm = [float(r.get("grad_norm", np.nan)) for r in rows]
        tail = max(1, min(len(rows) // 20, 5000))
        summary_runs.append(
            {
                "label": run["label"],
                "method": run["row"].get("method"),
                "run_dir": str(run["run_dir"]),
                "steps": len(rows),
                "smooth_window": smooth_window,
                "reward": {
                    "start": rewards[0],
                    "final": rewards[-1],
                    "best": max(rewards),
                    "late_mean": float(np.mean(rewards[-tail:])),
                    "late_std": float(np.std(rewards[-tail:])),
                },
                "batch_unique_topologies": {
                    "start": unique_topo[0],
                    "final": unique_topo[-1],
                    "late_mean": float(np.nanmean(unique_topo[-tail:])),
                },
                "grad_norm": {
                    "start": grad_norm[0],
                    "final": grad_norm[-1],
                    "late_mean": float(np.nanmean(grad_norm[-tail:])),
                },
            }
        )
    return {"runs": summary_runs}


def plot_comparison_overview(
    runs: list[dict],
    *,
    output_dir: Path,
    smooth_window: int,
    max_points: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=200, sharex=True)
    labels = [run["label"] for run in runs]
    colors = color_map(labels)

    ax_reward = axes[0, 0]
    ax_best = axes[0, 1]
    ax_div = axes[1, 0]
    ax_grad = axes[1, 1]

    for run in runs:
        label = run["label"]
        color = colors[label]
        rows = run["rows"]
        steps = [int(r["global_step"]) for r in rows]
        rewards = [float(r["mean_log_reward"]) for r in rows]
        batch_unique_topo = [float(r.get("batch_unique_topologies", np.nan)) for r in rows]
        grad_norm = [float(r.get("grad_norm", np.nan)) for r in rows]

        plot_series(
            ax_reward,
            steps,
            rewards,
            label=label,
            color=color,
            smooth_window=smooth_window,
            max_points=max_points,
        )
        plot_series(
            ax_best,
            steps,
            running_best(rewards),
            label=label,
            color=color,
            smooth_window=max(1, smooth_window // 4),
            max_points=max_points,
            raw_alpha=0.0,
        )
        plot_series(
            ax_div,
            steps,
            batch_unique_topo,
            label=label,
            color=color,
            smooth_window=smooth_window,
            max_points=max_points,
        )
        plot_series(
            ax_grad,
            steps,
            [max(v, 1e-12) for v in grad_norm],
            label=label,
            color=color,
            smooth_window=smooth_window,
            max_points=max_points,
        )

    ax_reward.set_title("Mean log reward")
    ax_reward.set_ylabel("Mean log reward")
    ax_reward.grid(True, alpha=0.25)
    ax_reward.legend(frameon=False, fontsize=9)

    ax_best.set_title("Running best mean log reward")
    ax_best.set_ylabel("Best-so-far reward")
    ax_best.grid(True, alpha=0.25)

    ax_div.set_title("Batch unique topologies")
    ax_div.set_xlabel("Global step / resample round")
    ax_div.set_ylabel("Unique topologies")
    ax_div.grid(True, alpha=0.25)

    ax_grad.set_title("Gradient norm")
    ax_grad.set_xlabel("Global step / resample round")
    ax_grad.set_ylabel("Grad norm")
    ax_grad.set_yscale("log")
    ax_grad.grid(True, alpha=0.25)

    savefig(fig, output_dir / "training_overview.png")


def plot_loss_and_duplication(
    runs: list[dict],
    *,
    output_dir: Path,
    smooth_window: int,
    max_points: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=200, sharex=True)
    labels = [run["label"] for run in runs]
    colors = color_map(labels)

    ax_loss, ax_dup = axes
    for run in runs:
        label = run["label"]
        color = colors[label]
        rows = run["rows"]
        steps = [int(r["global_step"]) for r in rows]
        abs_loss = [abs(float(r["loss"])) for r in rows if "loss" in r]
        dup_frac = [float(r.get("batch_duplicate_topology_fraction", np.nan)) for r in rows]

        plot_series(
            ax_loss,
            steps,
            [max(v, 1e-12) for v in abs_loss],
            label=label,
            color=color,
            smooth_window=smooth_window,
            max_points=max_points,
        )
        plot_series(
            ax_dup,
            steps,
            dup_frac,
            label=label,
            color=color,
            smooth_window=smooth_window,
            max_points=max_points,
        )

    ax_loss.set_title("|Loss| (method-specific scale)")
    ax_loss.set_xlabel("Global step / resample round")
    ax_loss.set_ylabel("|loss|")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend(frameon=False, fontsize=9)

    ax_dup.set_title("Batch duplicate topology fraction")
    ax_dup.set_xlabel("Global step / resample round")
    ax_dup.set_ylabel("Duplicate fraction")
    ax_dup.grid(True, alpha=0.25)

    savefig(fig, output_dir / "training_loss_duplication.png")


def plot_ips_diagnostics(
    runs: list[dict],
    *,
    output_dir: Path,
    smooth_window: int,
    max_points: int,
) -> None:
    ips_runs = [run for run in runs if run["row"].get("method") == "hybrid_ips_grpo"]
    if not ips_runs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=200, sharex=True)
    labels = [run["label"] for run in ips_runs]
    colors = color_map(labels)

    ax_entropy = axes[0, 0]
    ax_ratio = axes[0, 1]
    ax_prob = axes[1, 0]
    ax_buffer = axes[1, 1]

    for run in ips_runs:
        label = run["label"]
        color = colors[label]
        rows = run["rows"]
        steps = [int(r["global_step"]) for r in rows]
        entropy = [float(r.get("mean_policy_entropy", np.nan)) for r in rows]
        max_ratio = [float(r.get("max_importance_ratio", np.nan)) for r in rows]
        ips_prob_mean = [float(r.get("ips_prob_mean", np.nan)) for r in rows]
        buffer_size = [float(r.get("best_tree_buffer_size", np.nan)) for r in rows]

        plot_series(
            ax_entropy,
            steps,
            entropy,
            label=label,
            color=color,
            smooth_window=smooth_window,
            max_points=max_points,
        )
        plot_series(
            ax_ratio,
            steps,
            [max(v, 1e-12) for v in max_ratio],
            label=label,
            color=color,
            smooth_window=smooth_window,
            max_points=max_points,
        )
        plot_series(
            ax_prob,
            steps,
            ips_prob_mean,
            label=label,
            color=color,
            smooth_window=smooth_window,
            max_points=max_points,
        )
        plot_series(
            ax_buffer,
            steps,
            buffer_size,
            label=label,
            color=color,
            smooth_window=max(1, smooth_window // 8),
            max_points=max_points,
        )

    ax_entropy.set_title("IPS policy entropy")
    ax_entropy.set_ylabel("Mean policy entropy")
    ax_entropy.grid(True, alpha=0.25)
    ax_entropy.legend(frameon=False, fontsize=9)

    ax_ratio.set_title("IPS max importance ratio")
    ax_ratio.set_ylabel("Max importance ratio")
    ax_ratio.set_yscale("log")
    ax_ratio.grid(True, alpha=0.25)

    ax_prob.set_title("IPS mean outcome probability")
    ax_prob.set_xlabel("Global step / resample round")
    ax_prob.set_ylabel("Mean p_hat")
    ax_prob.grid(True, alpha=0.25)

    ax_buffer.set_title("Replay buffer occupancy")
    ax_buffer.set_xlabel("Global step / resample round")
    ax_buffer.set_ylabel("Best-tree buffer size")
    ax_buffer.grid(True, alpha=0.25)

    savefig(fig, output_dir / "training_ips_diagnostics.png")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    run_rows = manifest.get("runs", [])
    if not run_rows:
        raise ValueError(f"No runs found in {args.manifest}")

    runs = []
    for row in run_rows:
        run_dir = Path(row["run_dir"])
        runs.append(
            {
                "row": row,
                "label": label_for_run(row),
                "run_dir": run_dir,
                "rows": load_metrics(run_dir),
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_comparison_overview(
        runs,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
        max_points=args.max_points,
    )
    plot_loss_and_duplication(
        runs,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
        max_points=args.max_points,
    )
    plot_ips_diagnostics(
        runs,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
        max_points=args.max_points,
    )

    summary = build_summary(runs, smooth_window=args.smooth_window)
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
