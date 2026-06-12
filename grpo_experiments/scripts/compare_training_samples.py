#!/usr/bin/env python3
"""Compare training-time tree distributions from trajectory_samples.jsonl."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from grpo_experiments.eval_utils import entropy_from_counts, load_json, save_json
from grpo_experiments.scripts.compare_sampling import (
    compute_bin_edges,
    compute_bin_frequencies,
    json_ready_summary,
    plot_sampling_comparison,
    plot_sampling_distributions,
    plot_sampling_overlay,
    plot_score_density,
    print_bin_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare training-time tree distributions across runs.",
    )
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of training trees to subsample per run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Per-run stride offset seed for reproducible subsampling.",
    )
    parser.add_argument(
        "--global-step-min",
        type=int,
        default=None,
        help="Only include trees logged at or after this global step.",
    )
    parser.add_argument(
        "--global-step-max",
        type=int,
        default=None,
        help="Only include trees logged at or before this global step.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of equal log-score bins for the reward histogram.",
    )
    return parser.parse_args()


def _total_trees_logged(run_dir: Path) -> int | None:
    meta_path = run_dir / "trajectory_log_meta.json"
    if not meta_path.exists():
        return None
    return int(load_json(meta_path).get("total_trees_logged", 0) or 0)


def _passes_step_filter(
    row: dict[str, Any],
    *,
    step_min: int | None,
    step_max: int | None,
) -> bool:
    step = int(row.get("gs", row.get("global_step", -1)))
    if step_min is not None and step < step_min:
        return False
    if step_max is not None and step > step_max:
        return False
    return True


def subsample_trajectory_rows(
    path: Path,
    max_samples: int,
    *,
    seed: int,
    step_min: int | None,
    step_max: int | None,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing trajectory log: {path}")

    if step_min is not None or step_max is not None:
        return _reservoir_sample_filtered(
            path,
            max_samples,
            seed=seed,
            step_min=step_min,
            step_max=step_max,
        )

    total = _total_trees_logged(path.parent)
    if total is None or total <= max_samples:
        rows: list[dict[str, Any]] = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(rows) >= max_samples:
                    break
        return rows

    stride = max(1, total // max_samples)
    offset = seed % stride
    rows = []
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if idx % stride != offset:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= max_samples:
                break
    return rows


def _reservoir_sample_filtered(
    path: Path,
    max_samples: int,
    *,
    seed: int,
    step_min: int | None,
    step_max: int | None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _passes_step_filter(row, step_min=step_min, step_max=step_max):
                continue
            seen += 1
            if len(reservoir) < max_samples:
                reservoir.append(row)
            else:
                j = int(rng.integers(0, seen))
                if j < max_samples:
                    reservoir[j] = row
    return reservoir


def build_summary(
    rows: list[dict[str, Any]],
    *,
    label: str,
    run_dir: Path,
) -> dict[str, Any]:
    signatures = [str(row["sig"]) for row in rows]
    topology_ids = [str(row["topo"]) for row in rows]
    log_scores = np.asarray([float(row["ls"]) for row in rows], dtype=np.float64)
    signature_counts = Counter(signatures)
    topology_counts = Counter(topology_ids)

    return {
        "label": label,
        "run_dir": str(run_dir),
        "method": "training_trajectory",
        "checkpoint_path": "",
        "samples": int(len(rows)),
        "unique_signatures": int(len(signature_counts)),
        "unique_topologies": int(len(topology_counts)),
        "signature_entropy": entropy_from_counts(dict(signature_counts)),
        "topology_entropy": entropy_from_counts(dict(topology_counts)),
        "signature_duplicate_fraction": float(
            1.0 - len(signature_counts) / len(signatures)
        )
        if signatures
        else 0.0,
        "topology_duplicate_fraction": float(
            1.0 - len(topology_counts) / len(topology_ids)
        )
        if topology_ids
        else 0.0,
        "max_signature_share": float(max(signature_counts.values()) / len(signatures)),
        "max_topology_share": float(max(topology_counts.values()) / len(topology_ids)),
        "log_score_mean": float(log_scores.mean()) if log_scores.size else 0.0,
        "log_score_std": float(log_scores.std()) if log_scores.size else 0.0,
        "log_score_min": float(log_scores.min()) if log_scores.size else 0.0,
        "log_score_max": float(log_scores.max()) if log_scores.size else 0.0,
        "log_reward_mean": float(log_scores.mean()) if log_scores.size else 0.0,
        "log_reward_std": float(log_scores.std()) if log_scores.size else 0.0,
        "top_signatures": [
            {"id": key, "count": int(count), "share": float(count / len(signatures))}
            for key, count in signature_counts.most_common(20)
        ],
        "top_topologies": [
            {"id": key, "count": int(count), "share": float(count / len(topology_ids))}
            for key, count in topology_counts.most_common(20)
        ],
        "log_scores": log_scores,
        "topology_ids": topology_ids,
        "signatures": signatures,
        "topology_counts": topology_counts,
        "signature_counts": signature_counts,
    }


def load_all_summaries(args: argparse.Namespace) -> list[dict[str, Any]]:
    labels = args.labels or [path.name for path in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise ValueError("--labels count must match --run-dirs count")

    summaries: list[dict[str, Any]] = []
    for idx, (label, run_dir) in enumerate(zip(labels, args.run_dirs)):
        samples_path = run_dir / "trajectory_samples.jsonl"
        print(
            f"subsample {args.samples} training trees from {label} ({run_dir})",
            end="",
        )
        rows = subsample_trajectory_rows(
            samples_path,
            args.samples,
            seed=args.seed + idx,
            step_min=args.global_step_min,
            step_max=args.global_step_max,
        )
        print(f" -> got {len(rows)}")
        summaries.append(build_summary(rows, label=label, run_dir=run_dir))
    return summaries


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_all_summaries(args)
    bin_edges = compute_bin_edges(summaries, args.n_bins)
    frequencies = compute_bin_frequencies(summaries, bin_edges)

    step_note = ""
    if args.global_step_min is not None or args.global_step_max is not None:
        step_note = (
            f" steps=[{args.global_step_min or 0}, {args.global_step_max or 'end'}]"
        )

    payload = {
        "metadata": {
            "source": "trajectory_samples.jsonl",
            "samples_per_run": args.samples,
            "seed_base": args.seed,
            "n_bins": args.n_bins,
            "global_step_min": args.global_step_min,
            "global_step_max": args.global_step_max,
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
    save_json(args.output_dir / "training_summary.json", payload)

    plot_sampling_comparison(
        summaries,
        bin_edges,
        frequencies,
        args.output_dir / "training_comparison.png",
        args.top_k,
        samples=args.samples,
        n_bins=args.n_bins,
        title_context="Training",
    )
    plot_sampling_distributions(
        summaries,
        args.output_dir / "training_distributions.png",
        samples=args.samples,
        title_context="Training",
    )
    plot_sampling_overlay(
        summaries,
        args.output_dir / "training_distributions_overlay.png",
        title_context="Training",
    )
    plot_score_density(
        summaries,
        args.output_dir / "training_score_density.png",
        title_context="Training",
    )

    print_bin_table(summaries, bin_edges, frequencies)

    print(f"\nsaved outputs under: {args.output_dir}{step_note}")
    print("  training_summary.json")
    print("  training_comparison.png")
    print("  training_distributions.png")
    print("  training_distributions_overlay.png")
    print("  training_score_density.png")
    for row in summaries:
        print(
            f"  {row['label']}: unique_topo={row['unique_topologies']} "
            f"topo_dup={row['topology_duplicate_fraction']:.3f} "
            f"log_score_mean={row['log_score_mean']:.2f} "
            f"log_score_best={row['log_score_max']:.2f}"
        )


if __name__ == "__main__":
    main()
