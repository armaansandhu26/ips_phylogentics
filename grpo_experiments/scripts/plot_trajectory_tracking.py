#!/usr/bin/env python3
"""Plot training trajectory diversity + final sampler comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grpo_experiments.eval_utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--sampling-summary",
        type=Path,
        default=None,
        help="Optional sampling_summary.json from compare_sampling.",
    )
    p.add_argument(
        "--max-scatter-points",
        type=int,
        default=8000,
        help="Max training sample points per run on scatter plot.",
    )
    return p.parse_args()


def load_summary_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def subsample_rows(rows: list[dict], max_points: int) -> list[dict]:
    if len(rows) <= max_points:
        return rows
    idx = np.linspace(0, len(rows) - 1, max_points, dtype=int)
    return [rows[i] for i in idx]


def load_sample_rows(path: Path, max_points: int) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    stride = max(1, len(rows) // max(max_points, 1))
    return rows[::stride][:max_points]


def plot_training_diversity(
    runs: list[tuple[str, Path]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=200)
    cmap = plt.get_cmap("tab10")

    for idx, (label, run_dir) in enumerate(runs):
        color = cmap(idx % 10)
        rows = load_summary_rows(run_dir / "trajectory_step_summary.jsonl")
        if not rows:
            continue
        steps = [int(r["global_step"]) for r in rows]
        axes[0, 0].plot(
            steps,
            [r["unique_signatures"] for r in rows],
            label=label,
            color=color,
            linewidth=1.2,
        )
        axes[0, 1].plot(
            steps,
            [r["unique_topologies"] for r in rows],
            label=label,
            color=color,
            linewidth=1.2,
        )
        axes[1, 0].plot(
            steps,
            [r["mean_log_score"] for r in rows],
            label=label,
            color=color,
            linewidth=1.2,
        )
        axes[1, 1].plot(
            steps,
            [r["topology_duplicate_fraction"] for r in rows],
            label=label,
            color=color,
            linewidth=1.2,
        )

    axes[0, 0].set_title("Unique signatures per training batch")
    axes[0, 1].set_title("Unique topologies per training batch")
    axes[1, 0].set_title("Mean log score per training batch")
    axes[1, 1].set_title("Topology duplicate fraction per batch")
    for ax in axes.flatten():
        ax.set_xlabel("Global step")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Training-time trajectory diversity", fontsize=13)
    fig.tight_layout()
    out = output_dir / "training_trajectory_diversity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_training_score_scatter(
    runs: list[tuple[str, Path]],
    output_dir: Path,
    *,
    max_points: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
    cmap = plt.get_cmap("tab10")

    for idx, (label, run_dir) in enumerate(runs):
        rows = load_sample_rows(run_dir / "trajectory_samples.jsonl", max_points)
        if not rows:
            continue
        steps = [int(r["gs"]) for r in rows]
        scores = [float(r["ls"]) for r in rows]
        ax.scatter(
            steps,
            scores,
            s=4,
            alpha=0.25,
            label=label,
            color=cmap(idx % 10),
            linewidths=0,
        )

    ax.set_xlabel("Global step")
    ax.set_ylabel("Log score")
    ax.set_title("Training samples (subsampled scatter)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = output_dir / "training_log_score_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_final_sampling_bars(summary_path: Path | None, output_dir: Path) -> None:
    if summary_path is None or not summary_path.exists():
        return
    payload = load_json(summary_path)
    runs = payload.get("runs", [])
    if not runs:
        return

    labels = [r["label"] for r in runs]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)
    ax.bar(
        x - width / 2,
        [r["unique_topologies"] for r in runs],
        width,
        label="Unique topologies",
        color="#2ca02c",
    )
    ax.bar(
        x + width / 2,
        [r["unique_signatures"] for r in runs],
        width,
        label="Unique signatures",
        color="#ff7f0e",
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count (final sampler)")
    ax.set_title("Final checkpoint sampling diversity")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "final_sampling_diversity.png", bbox_inches="tight")
    plt.close(fig)


def build_report(
    runs: list[tuple[str, Path]],
    sampling_summary: Path | None,
) -> dict:
    report = {"runs": []}
    for label, run_dir in runs:
        summary_rows = load_summary_rows(run_dir / "trajectory_step_summary.jsonl")
        entry = {
            "label": label,
            "run_dir": str(run_dir),
            "training_steps_logged": len(summary_rows),
        }
        if summary_rows:
            last = summary_rows[-1]
            entry["training_final"] = {
                "unique_signatures": last.get("unique_signatures"),
                "unique_topologies": last.get("unique_topologies"),
                "mean_log_score": last.get("mean_log_score"),
            }
        if sampling_summary and sampling_summary.exists():
            for row in load_json(sampling_summary).get("runs", []):
                if row.get("label") == label:
                    entry["final_sampling"] = {
                        "unique_signatures": row.get("unique_signatures"),
                        "unique_topologies": row.get("unique_topologies"),
                        "log_score_mean": row.get("log_score_mean"),
                        "log_score_std": row.get("log_score_std"),
                    }
                    break
        report["runs"].append(entry)
    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = args.labels or [p.name for p in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise ValueError("--labels count must match --run-dirs")
    runs = list(zip(labels, args.run_dirs))

    plot_training_diversity(runs, args.output_dir)
    plot_training_score_scatter(runs, args.output_dir, max_points=args.max_scatter_points)
    plot_final_sampling_bars(args.sampling_summary, args.output_dir)

    report = build_report(runs, args.sampling_summary)
    save_json(args.output_dir / "trajectory_tracking_report.json", report)

    print(f"saved plots under: {args.output_dir}")
    for path in sorted(args.output_dir.glob("*.png")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
