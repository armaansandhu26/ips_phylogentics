#!/usr/bin/env python3
"""Evaluate Pearson correlation between estimated log q(tree) and true log score.

Consumes a frozen benchmark built by `build_tree_eval_benchmark.py`, reconstructs
each stored tree from its saved action sequence, estimates log q(tree) under one
or more checkpoints, and writes raw pairs plus summary metrics/plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from grpo_experiments.eval_utils import (
    choose_device,
    estimate_tree_logq,
    load_generator,
    load_json,
    resolve_run_artifacts,
    save_json,
    set_seed,
)
from src.gfn.rollout_worker_phylo import RolloutWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate log q(tree) Pearson correlation on a frozen tree benchmark.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Path to benchmark.json produced by build_tree_eval_benchmark.py",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory to evaluate. Can be provided multiple times.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional display label for a run-dir. Must match count if provided.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint filename inside each run dir (default: auto-resolve).",
    )
    parser.add_argument(
        "--backward-trajectories",
        type=int,
        default=200,
        help="Backward trajectories per tree for log q(tree) estimation.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: sibling of benchmark.json named pearson_eval).",
    )
    return parser.parse_args()


def resolve_run_specs(run_dirs: list[str], labels: list[str]) -> list[tuple[str, str]]:
    if labels and len(labels) != len(run_dirs):
        raise ValueError("--label count must match --run-dir count when provided.")
    specs = []
    for idx, run_dir in enumerate(run_dirs):
        label = labels[idx] if labels else None
        artifacts = resolve_run_artifacts(run_dir, label=label)
        specs.append((artifacts.label, run_dir))
    return specs


def pearson_or_none(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    corr = np.corrcoef(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    return float(corr[0, 1])


def reconstruct_tree(env, row: dict[str, Any]):
    tree = env.build_tree_from_actions(row["actions"], row["log_score"])
    stored_topology = row.get("topology_id")
    if stored_topology is not None and tree.tree_topology_id != stored_topology:
        raise ValueError(
            f"Reconstructed topology mismatch for benchmark_index={row.get('benchmark_index')}: "
            f"{tree.tree_topology_id} != {stored_topology}"
        )
    return tree


def scatter_colors(bands: list[str]) -> list[str]:
    color_map = {"low": "#d55e00", "medium": "#0072b2", "high": "#009e73"}
    return [color_map.get(band, "#444444") for band in bands]


def plot_run_scatter(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> None:
    bands = [row["score_band"] for row in rows]
    x = np.asarray([row["estimated_log_q"] for row in rows], dtype=np.float64)
    y = np.asarray([row["true_log_score"] for row in rows], dtype=np.float64)
    colors = scatter_colors(bands)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180, constrained_layout=True)
    ax.scatter(x, y, s=16, c=colors, alpha=0.75, edgecolors="none")
    ax.set_xlabel("Estimated log q(tree)")
    ax.set_ylabel("True unnormalized log posterior score")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    # Small legend without importing patch handles.
    for band, color in [("low", "#d55e00"), ("medium", "#0072b2"), ("high", "#009e73")]:
        ax.scatter([], [], s=24, c=color, label=band)
    ax.legend(frameon=False, title="Score band")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_band_panels(rows: list[dict[str, Any]], output_path: Path, *, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=180, constrained_layout=True)
    for ax, band in zip(axes, ("low", "medium", "high")):
        band_rows = [row for row in rows if row["score_band"] == band]
        x = np.asarray([row["estimated_log_q"] for row in band_rows], dtype=np.float64)
        y = np.asarray([row["true_log_score"] for row in band_rows], dtype=np.float64)
        ax.scatter(x, y, s=16, c=scatter_colors([band] * len(band_rows)), alpha=0.75, edgecolors="none")
        ax.set_title(f"{band} (n={len(band_rows)})")
        ax.set_xlabel("Estimated log q(tree)")
        ax.grid(True, alpha=0.25)
        if band == "low":
            ax.set_ylabel("True log score")
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_one_run(
    benchmark_rows: list[dict[str, Any]],
    run_label: str,
    run_dir: str,
    *,
    checkpoint_name: str | None,
    device: str,
    n_backward_trajectories: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = resolve_run_artifacts(run_dir, label=run_label)
    if checkpoint_name is not None:
        checkpoint_path = artifacts.root / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        artifacts.checkpoint_path = checkpoint_path

    _, env, generator = load_generator(artifacts, device)
    rollout_worker = RolloutWorker(env)

    pair_rows: list[dict[str, Any]] = []
    for row in benchmark_rows:
        tree = reconstruct_tree(env, row)
        estimate = estimate_tree_logq(
            env,
            rollout_worker,
            generator,
            tree,
            n_backward_trajectories=n_backward_trajectories,
        )
        pair_rows.append(
            {
                "benchmark_index": row["benchmark_index"],
                "score_band": row["score_band"],
                "topology_id": row["topology_id"],
                "signature": row["signature"],
                "true_log_score": float(row["log_score"]),
                "estimated_log_q": float(estimate["log_q_tree"]),
                "log_q_importance_term_std": float(estimate["importance_term_std"]),
                "source_label": row["source_label"],
            }
        )

    overall = pearson_or_none(
        [row["estimated_log_q"] for row in pair_rows],
        [row["true_log_score"] for row in pair_rows],
    )
    by_band = {}
    for band in ("low", "medium", "high"):
        band_rows = [row for row in pair_rows if row["score_band"] == band]
        by_band[band] = pearson_or_none(
            [row["estimated_log_q"] for row in band_rows],
            [row["true_log_score"] for row in band_rows],
        )

    summary = {
        "label": run_label,
        "run_dir": str(artifacts.root),
        "method": artifacts.method,
        "checkpoint_path": str(artifacts.checkpoint_path),
        "backward_trajectories": n_backward_trajectories,
        "num_benchmark_trees": len(pair_rows),
        "pearson_overall": overall,
        "pearson_by_band": by_band,
    }
    return pair_rows, summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    benchmark_payload = load_json(args.benchmark)
    benchmark_rows = benchmark_payload["benchmark"]
    run_specs = resolve_run_specs(args.run_dir, args.label)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.benchmark.parent / "pearson_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_summary = {
        "benchmark_path": str(args.benchmark),
        "device": device,
        "seed": args.seed,
        "backward_trajectories": args.backward_trajectories,
        "num_benchmark_trees": len(benchmark_rows),
        "runs": [],
    }

    for run_label, run_dir in run_specs:
        print(f"evaluating {run_label}: {run_dir}")
        pair_rows, summary = evaluate_one_run(
            benchmark_rows,
            run_label,
            run_dir,
            checkpoint_name=args.checkpoint,
            device=device,
            n_backward_trajectories=args.backward_trajectories,
        )

        run_slug = summary["label"].replace("/", "_").replace(" ", "_")
        run_output_dir = output_dir / run_slug
        run_output_dir.mkdir(parents=True, exist_ok=True)

        pairs_payload = {
            "metadata": {
                "label": summary["label"],
                "run_dir": summary["run_dir"],
                "method": summary["method"],
                "checkpoint_path": summary["checkpoint_path"],
                "benchmark_path": str(args.benchmark),
                "backward_trajectories": args.backward_trajectories,
            },
            "pairs": pair_rows,
            "summary": summary,
        }
        save_json(run_output_dir / "pairs.json", pairs_payload)
        plot_run_scatter(
            pair_rows,
            run_output_dir / "scatter_overall.png",
            title=f"{summary['label']} — overall Pearson={summary['pearson_overall']:.4f}"
            if summary["pearson_overall"] is not None
            else f"{summary['label']} — overall Pearson=n/a",
        )
        plot_band_panels(
            pair_rows,
            run_output_dir / "scatter_by_band.png",
            title=f"{summary['label']} — band-wise log q(tree) vs log score",
        )
        combined_summary["runs"].append(summary)

        overall = summary["pearson_overall"]
        overall_str = f"{overall:.4f}" if overall is not None else "n/a"
        print(f"  overall pearson={overall_str}")

    save_json(output_dir / "summary.json", combined_summary)
    print(f"\nsaved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
