#!/usr/bin/env python3
"""Build a fixed shared tree benchmark for Pearson log q(tree) evaluation.

Option B benchmark plan:
- sample candidate trees from a small set of strong reference checkpoints
- merge them into one candidate pool
- deduplicate by topology
- stratify by true log_score into high / medium / low bands
- sample a fixed number of trees per band
- persist both candidate pool and final frozen benchmark to disk

The frozen benchmark stores the action sequence for each selected tree so later
evaluation scripts can reconstruct the exact tree deterministically.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from grpo_experiments.eval_utils import (
    choose_device,
    load_generator,
    resolve_run_artifacts,
    save_json,
    set_seed,
)
from grpo_experiments.core.policy_replay import trajectory_actions
from src.gfn.rollout_worker_phylo import RolloutWorker


DEFAULT_SOURCES = [
    (
        "phylgfn_r64",
        "/home/armaan/phylogfn/grpo_experiments/runs/ips_replay_ablation/topo/"
        "20260603_112929_ablation_phylgfn_r64_phylgfn",
    ),
    (
        "hyb_ips_p005",
        "/home/armaan/phylogfn/grpo_experiments/runs/ips_replay_ablation/topo/"
        "20260603_115451_ablation_hyb_ips_pfloor_005_hybrid_ips_grpo",
    ),
    (
        "hyb_ips_p002",
        "/home/armaan/phylogfn/grpo_experiments/runs/ips_replay_ablation/topo/"
        "20260603_115912_ablation_hyb_ips_pfloor_002_hybrid_ips_grpo",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed pooled tree benchmark for Pearson evaluation.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Source in LABEL=RUN_DIR form. "
            "If omitted, uses the default PhyloGFN / hyb_ips p005 / hyb_ips p002 runs."
        ),
    )
    parser.add_argument(
        "--samples-per-run",
        type=int,
        default=1000,
        help="Candidate trees sampled from each source checkpoint.",
    )
    parser.add_argument(
        "--sampling-batch-size",
        type=int,
        default=64,
        help="Rollout batch size while sampling candidate trees.",
    )
    parser.add_argument(
        "--trees-per-band",
        type=int,
        default=100,
        help="Final frozen benchmark size per score band (low / medium / high).",
    )
    parser.add_argument(
        "--dedup-representative",
        choices=["best", "random"],
        default="random",
        help="How to choose one representative when many candidates share a topology.",
    )
    parser.add_argument(
        "--dedup-key",
        choices=["topology_id", "signature"],
        default="signature",
        help=(
            "Identifier used to deduplicate the candidate pool. "
            "Use signature for small datasets with low topology diversity; "
            "use topology_id for larger datasets when topology-unique benchmarking is preferred."
        ),
    )
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("grpo_experiments/eval_benchmarks/topology_pooled_ds1_r64"),
        help="Directory for candidate_pool.json / benchmark.json / summary.json",
    )
    return parser.parse_args()


def parse_sources(args_sources: list[str]) -> list[tuple[str, str]]:
    if not args_sources:
        return DEFAULT_SOURCES[:]
    sources: list[tuple[str, str]] = []
    for item in args_sources:
        if "=" not in item:
            raise ValueError(f"--source must be LABEL=RUN_DIR, got: {item}")
        label, run_dir = item.split("=", 1)
        label = label.strip()
        run_dir = run_dir.strip()
        if not label or not run_dir:
            raise ValueError(f"Malformed --source value: {item}")
        sources.append((label, run_dir))
    return sources


def sample_candidate_records(
    label: str,
    run_dir: str,
    *,
    device: str,
    samples_per_run: int,
    sampling_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = resolve_run_artifacts(run_dir, label=label)
    _, env, generator = load_generator(artifacts, device)
    rollout_worker = RolloutWorker(env)

    records: list[dict[str, Any]] = []
    sample_index = 0
    while sample_index < samples_per_run:
        current_batch = min(sampling_batch_size, samples_per_run - sample_index)
        data, trajectories = rollout_worker.rollout(
            generator,
            current_batch,
            generate_full_trajectories=True,
        )
        batch_log_scores = data["log_scores"].detach().cpu().numpy()
        batch_log_rewards = data["log_rewards"].detach().cpu().numpy()

        for idx, (trajectory, log_score, log_reward) in enumerate(
            zip(trajectories, batch_log_scores, batch_log_rewards)
        ):
            tree = trajectory.current_state.subtrees[0]
            records.append(
                {
                    "candidate_index": sample_index + idx,
                    "source_label": label,
                    "source_run_dir": str(artifacts.root),
                    "source_method": artifacts.method,
                    "checkpoint_path": str(artifacts.checkpoint_path),
                    "topology_id": tree.tree_topology_id,
                    "signature": tree.signature,
                    "log_score": float(log_score),
                    "log_reward": float(log_reward),
                    "trajectory_length": int(len(trajectory.actions)),
                    "actions": trajectory_actions(trajectory),
                }
            )
        sample_index += current_batch

    stats = {
        "label": label,
        "run_dir": str(artifacts.root),
        "method": artifacts.method,
        "samples": len(records),
        "unique_topologies_raw": len({row["topology_id"] for row in records}),
        "log_score_min": float(min(row["log_score"] for row in records)),
        "log_score_max": float(max(row["log_score"] for row in records)),
        "log_score_mean": float(np.mean([row["log_score"] for row in records])),
    }

    del generator
    del env
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return records, stats


def choose_topology_representatives(
    candidates: list[dict[str, Any]],
    *,
    dedup_key: str,
    representative: str,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row[dedup_key])].append(row)

    unique_rows: list[dict[str, Any]] = []
    duplicate_sizes: dict[str, int] = {}
    for dedup_value, rows in grouped.items():
        duplicate_sizes[dedup_value] = len(rows)
        if representative == "best":
            chosen = max(rows, key=lambda row: (row["log_score"], row["source_label"], -row["candidate_index"]))
        else:
            chosen = rows[rng.randrange(len(rows))]
        chosen = dict(chosen)
        chosen["dedup_key"] = dedup_key
        chosen["dedup_value"] = dedup_value
        chosen["duplicate_count"] = int(len(rows))
        chosen["duplicate_source_counts"] = dict(Counter(row["source_label"] for row in rows))
        unique_rows.append(chosen)

    return unique_rows, duplicate_sizes


def split_score_bands(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if len(rows) < 3:
        raise ValueError("Need at least 3 unique rows to split into score bands.")
    rows_sorted = sorted(rows, key=lambda row: row["log_score"])
    low_rows, medium_rows, high_rows = np.array_split(np.asarray(rows_sorted, dtype=object), 3)
    return {
        "low": low_rows.tolist(),
        "medium": medium_rows.tolist(),
        "high": high_rows.tolist(),
    }


def sample_final_benchmark(
    score_bands: dict[str, list[dict[str, Any]]],
    *,
    trees_per_band: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    benchmark_rows: list[dict[str, Any]] = []
    benchmark_index = 0
    for band_name in ("low", "medium", "high"):
        band_rows = score_bands[band_name]
        if len(band_rows) < trees_per_band:
            raise ValueError(
                f"Band {band_name!r} has only {len(band_rows)} unique trees, "
                f"cannot sample {trees_per_band}."
            )
        selected = rng.sample(band_rows, trees_per_band)
        selected = sorted(selected, key=lambda row: row["log_score"])
        for row in selected:
            out = dict(row)
            out["benchmark_index"] = benchmark_index
            out["score_band"] = band_name
            benchmark_rows.append(out)
            benchmark_index += 1
    return benchmark_rows


def summarize_bands(score_bands: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for band_name, rows in score_bands.items():
        scores = [row["log_score"] for row in rows]
        out[band_name] = {
            "count": len(rows),
            "log_score_min": float(min(scores)),
            "log_score_max": float(max(scores)),
            "log_score_mean": float(np.mean(scores)),
        }
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = choose_device(args.device)
    sources = parse_sources(args.source)

    print(f"device: {device}")
    print(f"sources: {len(sources)}")
    for label, run_dir in sources:
        print(f"  - {label}: {run_dir}")

    all_candidates: list[dict[str, Any]] = []
    source_summaries = []
    for label, run_dir in sources:
        print(f"\nsampling candidates from {label}")
        rows, stats = sample_candidate_records(
            label,
            run_dir,
            device=device,
            samples_per_run=args.samples_per_run,
            sampling_batch_size=args.sampling_batch_size,
        )
        all_candidates.extend(rows)
        source_summaries.append(stats)
        print(
            f"  samples={stats['samples']} "
            f"unique_topologies_raw={stats['unique_topologies_raw']} "
            f"log_score_mean={stats['log_score_mean']:.3f}"
        )

    unique_candidates, duplicate_sizes = choose_topology_representatives(
        all_candidates,
        dedup_key=args.dedup_key,
        representative=args.dedup_representative,
        rng=rng,
    )
    score_bands = split_score_bands(unique_candidates)
    benchmark_rows = sample_final_benchmark(
        score_bands,
        trees_per_band=args.trees_per_band,
        rng=rng,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_pool_path = args.output_dir / "candidate_pool.json"
    benchmark_path = args.output_dir / "benchmark.json"
    summary_path = args.output_dir / "summary.json"

    candidate_payload = {
        "metadata": {
            "seed": args.seed,
            "device": device,
            "samples_per_run": args.samples_per_run,
            "sampling_batch_size": args.sampling_batch_size,
            "dedup_key": args.dedup_key,
            "dedup_representative": args.dedup_representative,
            "sources": [{"label": label, "run_dir": run_dir} for label, run_dir in sources],
        },
        "candidates": unique_candidates,
    }
    benchmark_payload = {
        "metadata": {
            "seed": args.seed,
            "device": device,
            "trees_per_band": args.trees_per_band,
            "bands": ["low", "medium", "high"],
            "dedup_key": args.dedup_key,
            "sources": [{"label": label, "run_dir": run_dir} for label, run_dir in sources],
        },
        "benchmark": benchmark_rows,
    }
    summary_payload = {
        "seed": args.seed,
        "device": device,
        "samples_per_run": args.samples_per_run,
        "sampling_batch_size": args.sampling_batch_size,
        "trees_per_band": args.trees_per_band,
        "dedup_key": args.dedup_key,
        "dedup_representative": args.dedup_representative,
        "sources": source_summaries,
        "candidate_count_raw": len(all_candidates),
        "candidate_count_unique": len(unique_candidates),
        "duplicate_histogram": dict(Counter(duplicate_sizes.values())),
        "score_bands": summarize_bands(score_bands),
        "benchmark_count": len(benchmark_rows),
        "benchmark_band_counts": dict(Counter(row["score_band"] for row in benchmark_rows)),
    }

    save_json(candidate_pool_path, candidate_payload)
    save_json(benchmark_path, benchmark_payload)
    save_json(summary_path, summary_payload)

    print("\nSaved benchmark artifacts:")
    print(f"  candidate pool: {candidate_pool_path}")
    print(f"  frozen benchmark: {benchmark_path}")
    print(f"  summary: {summary_path}")
    print(
        f"  raw candidates={summary_payload['candidate_count_raw']} "
        f"unique_candidates={summary_payload['candidate_count_unique']} "
        f"final_benchmark={summary_payload['benchmark_count']}"
    )


if __name__ == "__main__":
    main()
