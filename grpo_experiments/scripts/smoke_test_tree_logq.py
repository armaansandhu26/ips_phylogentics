#!/usr/bin/env python3
"""End-to-end smoke test for tree-level log q(tree) estimation.

This script is meant to validate that a trained checkpoint, including
IPS-GRPO / hybrid IPS-GRPO runs, can be evaluated with the PhyloGFN-style
tree-probability estimator:

    log q(tree) ~= logmeanexp(log_pf - log_pb)

over backward-sampled trajectories for a fixed final tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from grpo_experiments.eval_utils import (
    choose_device,
    estimate_tree_logq,
    load_generator,
    resolve_run_artifacts,
    sample_trees_from_generator,
    save_json,
    set_seed,
)
from src.gfn.rollout_worker_phylo import RolloutWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test end-to-end tree-level log q(tree) estimation for a trained run.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing experiment_config.json/run_args.json and a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint filename inside the run dir (default: auto-resolve).",
    )
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--sample-trees",
        type=int,
        default=5,
        help="Number of trees to sample from the checkpoint for smoke testing.",
    )
    parser.add_argument(
        "--sampling-batch-size",
        type=int,
        default=8,
        help="Batch size used when sampling test trees.",
    )
    parser.add_argument(
        "--backward-trajectories",
        type=int,
        default=16,
        help="Backward trajectories per tree for the main estimate.",
    )
    parser.add_argument(
        "--repeat-estimates",
        type=int,
        default=2,
        help="How many independent log q(tree) estimates to run per tree for a stability check.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling trees and backward trajectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <run-dir>/tree_logq_smoke_test.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    artifacts = resolve_run_artifacts(str(args.run_dir))
    if args.checkpoint is not None:
        checkpoint_path = artifacts.root / args.checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        artifacts.checkpoint_path = checkpoint_path

    cfg, env, generator = load_generator(artifacts, device)
    rollout_worker = RolloutWorker(env)

    output_path = args.output or (artifacts.root / "tree_logq_smoke_test.json")

    print(f"run: {artifacts.root}")
    print(f"method: {artifacts.method}")
    print(f"checkpoint: {artifacts.checkpoint_path}")
    print(f"device: {device}")
    print(f"sampling {args.sample_trees} trees")

    trees = sample_trees_from_generator(
        rollout_worker,
        generator,
        sample_trees=args.sample_trees,
        batch_size=args.sampling_batch_size,
    )

    results = []
    all_logq = []
    all_scores = []
    all_passed = True

    for tree_idx, tree in enumerate(trees):
        tree_id = getattr(tree, "tree_topology_id", None)
        tree_signature = getattr(tree, "signature", None)

        repeats = []
        for repeat_idx in range(args.repeat_estimates):
            estimate = estimate_tree_logq(
                env,
                rollout_worker,
                generator,
                tree,
                n_backward_trajectories=args.backward_trajectories,
            )
            estimate["repeat_idx"] = repeat_idx
            repeats.append(estimate)

        repeat_logq = np.asarray([row["log_q_tree"] for row in repeats], dtype=np.float64)
        repeat_mean = float(repeat_logq.mean())
        repeat_std = float(repeat_logq.std())
        passed = bool(np.isfinite(repeat_logq).all())
        all_passed = all_passed and passed

        all_logq.append(repeat_mean)
        all_scores.append(float(tree.log_score))

        result = {
            "tree_idx": tree_idx,
            "tree_topology_id": tree_id,
            "tree_signature": tree_signature,
            "tree_log_score": float(tree.log_score),
            "estimate_repeat_mean": repeat_mean,
            "estimate_repeat_std": repeat_std,
            "passed": passed,
            "repeats": repeats,
        }
        results.append(result)

        print(
            f"[tree {tree_idx}] "
            f"log_score={float(tree.log_score):.3f} "
            f"log_q_mean={repeat_mean:.3f} "
            f"log_q_std={repeat_std:.3f} "
            f"topology={tree_id}"
        )

    pearson_r = None
    if len(all_logq) >= 2:
        corr = np.corrcoef(np.asarray(all_logq), np.asarray(all_scores))
        pearson_r = float(corr[0, 1])

    summary = {
        "smoke_test_passed": bool(all_passed),
        "run_dir": str(artifacts.root),
        "method": artifacts.method,
        "checkpoint_path": str(artifacts.checkpoint_path),
        "device": device,
        "sample_trees": int(args.sample_trees),
        "sampling_batch_size": int(args.sampling_batch_size),
        "backward_trajectories": int(args.backward_trajectories),
        "repeat_estimates": int(args.repeat_estimates),
        "seed": int(args.seed),
        "tree_level_pearson_r_smoke": pearson_r,
        "notes": [
            "This is a smoke test, not a final benchmark.",
            "log q(tree) is estimated as logmeanexp(log_pf - log_pb) over backward-sampled trajectories.",
            "Use a larger evaluation set and more backward trajectories for the final Pearson figure.",
        ],
        "trees": results,
    }

    save_json(output_path, summary)

    print()
    print(f"saved: {output_path}")
    print(f"smoke_test_passed={summary['smoke_test_passed']}")
    if pearson_r is not None:
        print(f"tree_level_pearson_r_smoke={pearson_r:.4f}")


if __name__ == "__main__":
    main()
