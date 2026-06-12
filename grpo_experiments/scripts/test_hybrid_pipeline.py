#!/usr/bin/env python3
"""Run and validate a tiny end-to-end hybrid policy-IS pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _latest_run_dir(output_root: Path, run_name: str) -> Path:
    matches = sorted(
        output_root.glob(f"*_{run_name}_hybrid_grpo"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            f"No run directory found under {output_root} for run_name={run_name!r}."
        )
    return matches[-1]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run_and_validate(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"hybrid_pipeline_test_{int(time.time())}"

    device = args.device
    if args.cuda_device is not None:
        device = f"cuda:{args.cuda_device}"

    cmd = [
        args.python,
        "-m",
        "grpo_experiments.hybrid_grpo.train",
        "--cfg",
        args.cfg,
        "--dataset",
        args.dataset,
        "--output",
        str(output_root),
        "--run-name",
        run_name,
        "--seed",
        str(args.seed),
        "--device",
        device,
        "--resample-rounds",
        str(args.resample_rounds),
        "--update-cycles",
        str(args.update_cycles),
        "--fresh-buffer-size",
        str(args.fresh_buffer_size),
        "--replay-sample-size",
        str(args.replay_sample_size),
        "--best-tree-buffer-size",
        str(args.best_tree_buffer_size),
        "--rollout-chunk-size",
        str(args.rollout_chunk_size),
        "--print-every",
        str(args.print_every),
    ]
    if args.replay_warmstart_samples > 0:
        cmd.extend(
            [
                "--replay-warmstart-samples",
                str(args.replay_warmstart_samples),
            ]
        )
    if args.best_trees_topology_only:
        cmd.append("--best-trees-topology-only")

    print(">>> running tiny hybrid pipeline")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    run_dir = _latest_run_dir(output_root, run_name)
    print(f">>> validating run_dir={run_dir}")

    required = [
        "metrics.jsonl",
        "epoch_summaries.json",
        "final_checkpoint.pt",
        "experiment_config.json",
        "resolved_config.yaml",
        "training_state.json",
    ]
    for rel in required:
        _assert((run_dir / rel).exists(), f"Missing expected artifact: {run_dir / rel}")

    rows = _read_jsonl(run_dir / "metrics.jsonl")
    expected_rows = args.resample_rounds * args.update_cycles
    _assert(
        len(rows) == expected_rows,
        f"Expected {expected_rows} metric rows, found {len(rows)}.",
    )

    replay_seen = False
    fresh_seen = False
    for idx, row in enumerate(rows):
        _assert(row.get("training_mode") == "policy_is", f"row {idx}: training_mode mismatch")
        _assert(row.get("method") == "hybrid_grpo", f"row {idx}: method mismatch")
        _assert("mean_importance_ratio" in row, f"row {idx}: missing IS metrics")
        _assert("found_in_replay_buffer" in row, f"row {idx}: missing found_in_replay_buffer")
        _assert("replay_replaced" in row, f"row {idx}: missing replay_replaced")

        batch_size = int(row["batch_size"])
        fresh_count = int(row["fresh_count"])
        replay_count = int(row["replay_count"])
        _assert(
            fresh_count + replay_count == batch_size,
            f"row {idx}: fresh+replay != batch_size ({fresh_count}+{replay_count}!={batch_size})",
        )

        replay_frac = float(row["replay_fraction"])
        _assert(
            0.0 <= replay_frac <= 1.0,
            f"row {idx}: replay_fraction out of range: {replay_frac}",
        )

        found_in_buffer = int(row["found_in_replay_buffer"])
        replay_replaced = int(row["replay_replaced"])
        _assert(found_in_buffer >= 0, f"row {idx}: found_in_replay_buffer must be >= 0")
        _assert(replay_replaced >= 0, f"row {idx}: replay_replaced must be >= 0")
        _assert(
            replay_replaced <= found_in_buffer,
            f"row {idx}: replay_replaced cannot exceed found_in_replay_buffer",
        )

        if replay_count > 0:
            replay_seen = True
        if fresh_count > 0:
            fresh_seen = True

    if args.replay_sample_size > 0:
        _assert(replay_seen, "Expected replay samples, but replay_count was always 0.")
    if args.fresh_buffer_size > 0:
        _assert(fresh_seen, "Expected fresh samples, but fresh_count was always 0.")

    with (run_dir / "epoch_summaries.json").open() as handle:
        summaries = json.load(handle)
    _assert(
        isinstance(summaries, list) and len(summaries) == args.resample_rounds,
        (
            "epoch_summaries length mismatch: "
            f"expected {args.resample_rounds}, got {len(summaries) if isinstance(summaries, list) else 'non-list'}"
        ),
    )

    print(">>> hybrid pipeline sanity check passed")
    print(f"rows={len(rows)}  rounds={args.resample_rounds}  cycles={args.update_cycles}")
    print(f"latest_mean_loss={rows[-1]['loss']:.6f}  latest_replay_fraction={rows[-1]['replay_fraction']:.3f}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a tiny hybrid GRPO policy-IS run and validate core artifacts/metrics."
    )
    p.add_argument("--python", default=sys.executable, help="Python executable for launching train module.")
    p.add_argument(
        "--cfg",
        default=(
            "src/configs/benchmark_dna_cfgs/discrete_branch_lengths/"
            "cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml"
        ),
    )
    p.add_argument("--dataset", default="dataset/benchmark_datasets/DS1_reduced.pickle")
    p.add_argument("--output-root", default="grpo_experiments/runs/hybrid_pipeline_tests")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cpu",
        help="Training device string passed through to trainer (e.g. cpu, cuda:0, cuda:1).",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=4,
        help="Convenience override for CUDA index; if set, uses --device cuda:<index>.",
    )

    p.add_argument("--resample-rounds", type=int, default=50000)
    p.add_argument("--update-cycles", type=int, default=2)
    p.add_argument("--fresh-buffer-size", type=int, default=256)
    p.add_argument("--replay-sample-size", type=int, default=256)
    p.add_argument("--best-tree-buffer-size", type=int, default=512)
    p.add_argument("--replay-warmstart-samples", type=int, default=64)
    p.add_argument("--rollout-chunk-size", type=int, default=256)
    p.add_argument("--is-ratio-clip", type=float, default=0.2)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--best-trees-topology-only", action="store_true")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_dir = run_and_validate(args)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        raise
    print(f"[OK] {run_dir}")


if __name__ == "__main__":
    main()
