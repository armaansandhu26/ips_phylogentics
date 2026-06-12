#!/usr/bin/env python3
"""Always produce three-way comparison plots (PhyloGFN, GRPO, IPS)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from grpo_experiments.eval_utils import choose_device, load_json, resolve_checkpoint, save_json
from grpo_experiments.scripts.compare_late_vs_final import (
    build_report,
    plot_metrics_side_by_side,
    plot_paired_overlay,
    plot_side_by_side_grid,
    plot_three_way_overlay,
    training_window_label,
)
from grpo_experiments.scripts.compare_sampling import (
    compute_bin_edges,
    compute_bin_frequencies,
    json_ready_summary,
    plot_sampling_comparison,
    plot_sampling_distributions,
    plot_sampling_overlay,
    plot_score_density,
    print_bin_table,
    sample_run,
    save_scores_cache,
)
from grpo_experiments.scripts.compare_training_samples import (
    build_summary,
    subsample_trajectory_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--phylgfn-dir", type=Path, required=True)
    p.add_argument("--grpo-dir", type=Path, required=True)
    p.add_argument("--ips-dir", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=25000)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument(
        "--global-step-min",
        type=int,
        default=None,
        help="Optional training window lower bound (GRPO/IPS trajectories).",
    )
    p.add_argument(
        "--global-step-max",
        type=int,
        default=None,
        help="Optional training window upper bound (GRPO/IPS trajectories).",
    )
    return p.parse_args()


def pick_run_dir(root: Path, suffix: str, *, exclude: tuple[str, ...] = ()) -> Path | None:
    best_dir: Path | None = None
    best_n = -1
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not path.name.endswith(suffix):
            continue
        if any(token in path.name for token in exclude):
            continue
        metrics = path / "metrics.jsonl"
        if not metrics.exists():
            continue
        n_lines = sum(1 for _ in metrics.open())
        if n_lines > best_n:
            best_n = n_lines
            best_dir = path
    return best_dir


def resolve_default_dirs(root: Path) -> tuple[Path, Path, Path]:
    phyl = pick_run_dir(root, "_phylgfn_phylgfn")
    grpo = pick_run_dir(root, "_grpo_hybrid_grpo", exclude=("softmax",))
    ips = pick_run_dir(root, "_ips_hybrid_ips_grpo", exclude=("softmax",))
    missing = [
        name
        for name, path in (("phylgfn", phyl), ("grpo", grpo), ("ips", ips))
        if path is None
    ]
    if missing:
        raise FileNotFoundError(f"missing run dirs under {root}: {', '.join(missing)}")
    return phyl, grpo, ips


def load_training_summary(
    run_dir: Path,
    label: str,
    *,
    samples: int,
    seed: int,
    step_min: int | None,
    step_max: int | None,
) -> dict[str, Any]:
    traj = run_dir / "trajectory_samples.jsonl"
    if not traj.exists():
        raise FileNotFoundError(f"missing trajectory log for training side: {traj}")
    rows = subsample_trajectory_rows(
        traj,
        samples,
        seed=seed,
        step_min=step_min,
        step_max=step_max,
    )
    if not rows:
        raise RuntimeError(f"no trajectory rows for {label} ({run_dir})")
    return build_summary(rows, label=label, run_dir=run_dir)


def load_final_summary(
    run_dir: Path,
    label: str,
    *,
    samples: int,
    seed: int,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    try:
        ckpt = resolve_checkpoint(run_dir)
    except FileNotFoundError:
        ckpt = None
    if ckpt is not None:
        return sample_run(
            run_dir,
            label,
            device=device,
            samples=samples,
            batch_size=batch_size,
            seed=seed,
            checkpoint_name=ckpt.name,
            estimate_mll=False,
        )
    return load_training_summary(
        run_dir,
        label,
        samples=samples,
        seed=seed,
        step_min=None,
        step_max=None,
    )


def write_distribution_suite(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    *,
    prefix: str,
    title_context: str,
    samples: int,
    n_bins: int,
    top_k: int,
    metadata_extra: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_edges = compute_bin_edges(summaries, n_bins)
    frequencies = compute_bin_frequencies(summaries, bin_edges)

    payload = {
        "metadata": {
            "samples_per_run": samples,
            "n_bins": n_bins,
            "reward_bin_edges": [float(x) for x in bin_edges],
            **metadata_extra,
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
    save_json(output_dir / f"{prefix}_summary.json", payload)
    save_scores_cache(summaries, output_dir / f"{prefix}_scores.npz")

    plot_sampling_comparison(
        summaries,
        bin_edges,
        frequencies,
        output_dir / f"{prefix}_comparison.png",
        top_k,
        samples=samples,
        n_bins=n_bins,
        title_context=title_context,
    )
    plot_sampling_distributions(
        summaries,
        output_dir / f"{prefix}_distributions.png",
        samples=samples,
        title_context=title_context,
    )
    plot_sampling_overlay(
        summaries,
        output_dir / f"{prefix}_distributions_overlay.png",
        title_context=title_context,
    )
    plot_score_density(
        summaries,
        output_dir / f"{prefix}_score_density.png",
        title_context=title_context,
    )
    print_bin_table(summaries, bin_edges, frequencies)


def run_training_vs_final(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    output_dir: Path,
    *,
    training_title: str,
    step_min: int | None,
    step_max: int | None,
    samples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_side_by_side_grid(
        pairs,
        output_dir / "training_vs_final_grid.png",
        training_title=training_title,
    )
    plot_paired_overlay(
        pairs,
        output_dir / "training_vs_final_paired_overlay.png",
        training_title=training_title,
    )
    plot_metrics_side_by_side(
        pairs,
        output_dir / "training_vs_final_metrics.png",
        training_title=training_title,
    )
    plot_three_way_overlay(
        pairs,
        output_dir / "training_vs_final_three_way_overlay.png",
        training_title=training_title,
    )
    save_json(
        output_dir / "training_vs_final_summary.json",
        build_report(
            pairs,
            step_min=step_min,
            step_max=step_max,
            samples=samples,
        ),
    )


def main() -> None:
    args = parse_args()
    root = args.output_root
    phyl_dir = args.phylgfn_dir
    grpo_dir = args.grpo_dir
    ips_dir = args.ips_dir
    device = choose_device(args.device)
    rounds = args.rounds

    labels = [
        f"phylgfn_{rounds}",
        f"grpo_{rounds}",
        f"ips_{rounds}",
    ]
    dirs = [phyl_dir, grpo_dir, ips_dir]

    training_title = training_window_label(args.global_step_min, args.global_step_max)

    training_summaries: list[dict[str, Any]] = []
    final_summaries: list[dict[str, Any]] = []
    sources: dict[str, dict[str, str]] = {}

    for idx, (label, run_dir) in enumerate(zip(labels, dirs)):
        seed = args.seed + idx
        has_traj = (run_dir / "trajectory_samples.jsonl").exists()
        if not has_traj:
            final_row = load_final_summary(
                run_dir,
                label,
                samples=args.samples,
                seed=seed,
                device=device,
                batch_size=args.batch_size,
            )
            training_summaries.append(final_row)
            final_summaries.append(final_row)
            try:
                resolve_checkpoint(run_dir)
                ckpt_src = "checkpoint"
            except FileNotFoundError:
                ckpt_src = "missing_checkpoint"
            sources[label] = {
                "training": f"{ckpt_src} (no trajectory log)",
                "final": ckpt_src,
            }
            continue

        training_summaries.append(
            load_training_summary(
                run_dir,
                label,
                samples=args.samples,
                seed=seed,
                step_min=args.global_step_min,
                step_max=args.global_step_max,
            )
        )
        final_summaries.append(
            load_final_summary(
                run_dir,
                label,
                samples=args.samples,
                seed=seed + 100,
                device=device,
                batch_size=args.batch_size,
            )
        )
        train_src: Literal["training_trajectory", "final_checkpoint"] = "training_trajectory"
        try:
            resolve_checkpoint(run_dir)
            final_src: Literal["training_trajectory", "final_checkpoint"] = "final_checkpoint"
        except FileNotFoundError:
            final_src = "training_trajectory"
        sources[label] = {"training": train_src, "final": final_src}

    any_without_traj = any(
        not (run_dir / "trajectory_samples.jsonl").exists() for run_dir in dirs
    )
    note = (
        "Runs without trajectory logs use checkpoint sampling for training panels. "
        if any_without_traj
        else ""
    )

    training_dir = root / "training_eval"
    write_distribution_suite(
        training_summaries,
        training_dir,
        prefix="training",
        title_context=f"Three-way training trees ({note}GRPO/IPS from trajectories)",
        samples=args.samples,
        n_bins=args.n_bins,
        top_k=args.top_k,
        metadata_extra={
            "source": "trajectory_samples.jsonl or final_checkpoint (PhyloGFN)",
            "seed_base": args.seed,
            "global_step_min": args.global_step_min,
            "global_step_max": args.global_step_max,
            "data_sources": sources,
        },
    )

    sampling_dir = root / "sampling_eval"
    write_distribution_suite(
        final_summaries,
        sampling_dir,
        prefix="sampling",
        title_context="Three-way final sampling (checkpoint or training fallback)",
        samples=args.samples,
        n_bins=args.n_bins,
        top_k=args.top_k,
        metadata_extra={
            "seed_base": args.seed,
            "data_sources": sources,
        },
    )

    pairs = list(zip(labels, training_summaries, final_summaries))
    run_training_vs_final(
        pairs,
        root / "training_vs_final_eval",
        training_title=training_title,
        step_min=args.global_step_min,
        step_max=args.global_step_max,
        samples=args.samples,
    )

    traj_runs = [
        (label, run_dir)
        for label, run_dir in zip(labels, dirs)
        if (run_dir / "trajectory_step_summary.jsonl").exists()
    ]
    traj_plot_dir = root / "trajectory_plots"
    if len(traj_runs) == 3:
        from grpo_experiments.scripts import plot_trajectory_tracking as traj

        traj_plot_dir.mkdir(parents=True, exist_ok=True)
        sampling_summary = sampling_dir / "sampling_summary.json"
        traj.plot_training_diversity(traj_runs, traj_plot_dir)
        traj.plot_training_score_scatter(
            traj_runs,
            traj_plot_dir,
            max_points=8000,
        )
        traj.plot_final_sampling_bars(sampling_summary, traj_plot_dir)
        save_json(
            traj_plot_dir / "trajectory_tracking_report.json",
            traj.build_report(traj_runs, sampling_summary),
        )
    else:
        for path in traj_plot_dir.glob("*"):
            if path.is_file():
                path.unlink()
        print(
            f"skip trajectory_plots (need 3 trajectory logs, got {len(traj_runs)}); "
            f"removed any two-way plots in {traj_plot_dir}"
        )

    print(f"\nthree-way eval complete under {root}")
    for label, train, final in pairs:
        print(
            f"  {label}: train mean={train['log_score_mean']:.2f} topo={train['unique_topologies']} | "
            f"final mean={final['log_score_mean']:.2f} topo={final['unique_topologies']}"
        )


if __name__ == "__main__":
    main()
