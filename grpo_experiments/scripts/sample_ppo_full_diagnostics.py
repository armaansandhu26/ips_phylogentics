#!/usr/bin/env python3
"""Sample a full-model PPO checkpoint with path-probability diagnostics.

The compact NPZ output is compatible with
``plot_full_checkpoint_vs_reward_reference.py``. For PPO, the reverse-path
probability is the environment's fixed backward policy:

    implied log P(x) = log P_F(tau) - log P_B(tau).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from grpo_experiments.eval_utils import (  # noqa: E402
    choose_device,
    load_generator,
    resolve_run_artifacts,
    set_seed,
)
from src.gfn.rollout_worker_phylo import RolloutWorker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="PPO run directory or checkpoint.",
    )
    parser.add_argument("-n", "--num-trees", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--reward-shift", type=float, default=3600.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <run-dir>/sampled_ppo_full_diagnostics_<n>.npz.",
    )
    parser.add_argument("--print-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_trees <= 0 or args.batch_size <= 0:
        raise ValueError("--num-trees and --batch-size must be positive")

    set_seed(args.seed)
    device = choose_device(args.device)
    artifacts = resolve_run_artifacts(str(args.checkpoint))
    if args.checkpoint_name is not None:
        checkpoint_path = artifacts.root / args.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        artifacts.checkpoint_path = checkpoint_path
    output_path = args.output or (
        artifacts.root / f"sampled_ppo_full_diagnostics_{args.num_trees}.npz"
    )
    if output_path.suffix != ".npz":
        raise ValueError("--output must end in .npz")

    print(f"run: {artifacts.root}")
    print(f"checkpoint: {artifacts.checkpoint_path}")
    print(f"device: {device}")
    print(f"sampling {args.num_trees:,} PPO trajectories")

    cfg, env, generator = load_generator(artifacts, device)
    tree_only = bool(cfg.GFN.MODEL.ONLY_TRAIN_TREE_MODEL)
    if tree_only:
        print("note: tree-only checkpoint (fixed edge lengths); using environment P_B")
    rollout_worker = RolloutWorker(env)

    shifted_score = np.empty(args.num_trees, dtype=np.float32)
    raw_log_likelihood = np.empty(args.num_trees, dtype=np.float32)
    log_pf = np.empty(args.num_trees, dtype=np.float32)
    log_pb = np.empty(args.num_trees, dtype=np.float32)
    topology_index = np.empty(args.num_trees, dtype=np.int16)
    topology_ids: list[str] = []
    topology_lookup: dict[str, int] = {}

    generated = 0
    next_print = args.print_every
    with torch.inference_mode(), np.errstate(all="ignore"):
        while generated < args.num_trees:
            current_batch = min(args.batch_size, args.num_trees - generated)
            batch, trajectories = rollout_worker.rollout(
                generator,
                current_batch,
                generate_full_trajectories=True,
            )
            batch_log_pf = batch["log_paths_pf"].sum(dim=-1)
            batch_log_pb = batch["log_paths_pb"].sum(dim=-1)
            terminal_trees = [
                trajectory.current_state.subtrees[0]
                for trajectory in trajectories
            ]
            # Historical runs stored raw log L(x) on the terminal tree, while
            # the current environment stores 3600 + log L(x). Accept either
            # representation, but reject a mixed batch.
            batch_terminal_score = torch.as_tensor(
                [float(tree.log_score) for tree in terminal_trees],
                dtype=batch_log_pf.dtype,
                device=batch_log_pf.device,
            )
            positive = batch_terminal_score > 0.0
            if bool(positive.any()) and not bool(positive.all()):
                raise ValueError("terminal scores mix raw and shifted representations")
            if bool(positive.all()):
                batch_shifted_score = batch_terminal_score
                batch_raw_score = batch_terminal_score - args.reward_shift
            else:
                batch_raw_score = batch_terminal_score
                batch_shifted_score = batch_terminal_score + args.reward_shift

            for name, tensor in (
                ("log P_F", batch_log_pf),
                ("log P_B", batch_log_pb),
                ("terminal log likelihood", batch_raw_score),
                ("shifted reward", batch_shifted_score),
            ):
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"{name} contains NaN/Inf")
            if bool(torch.any(batch_shifted_score <= 0.0)):
                raise ValueError(
                    f"reward shift {args.reward_shift} does not make every reward positive"
                )

            stop = generated + current_batch
            raw_log_likelihood[generated:stop] = (
                batch_raw_score.detach().cpu().numpy()
            )
            shifted_score[generated:stop] = (
                batch_shifted_score.detach().cpu().numpy()
            )
            log_pf[generated:stop] = batch_log_pf.detach().cpu().numpy()
            log_pb[generated:stop] = batch_log_pb.detach().cpu().numpy()

            for offset, tree in enumerate(terminal_trees):
                topology_id = str(tree.tree_topology_id)
                index = topology_lookup.get(topology_id)
                if index is None:
                    index = len(topology_ids)
                    topology_lookup[topology_id] = index
                    topology_ids.append(topology_id)
                topology_index[generated + offset] = index

            generated = stop
            if args.print_every > 0 and (
                generated >= next_print or generated == args.num_trees
            ):
                print(
                    f"sampled {generated:,}/{args.num_trees:,}; "
                    f"topologies={len(topology_ids)}"
                )
                while next_print <= generated:
                    next_print += args.print_every

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        log_score=shifted_score,
        raw_log_likelihood=raw_log_likelihood,
        log_pf=log_pf,
        log_q_reverse=log_pb,
        topology_index=topology_index,
        topology_ids=np.asarray(topology_ids),
    )
    metadata = {
        "run_dir": str(artifacts.root),
        "method": "ppo",
        "checkpoint_path": str(artifacts.checkpoint_path),
        "device": device,
        "num_trees": int(args.num_trees),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "target": (
            f"R(x) = {args.reward_shift:g} + terminal_log_likelihood"
        ),
        "log_score_shift": float(args.reward_shift),
        "reverse_probability": "fixed environment P_B(tau | x)",
        "only_train_tree_model": bool(cfg.GFN.MODEL.ONLY_TRAIN_TREE_MODEL),
        "observed_topologies": len(topology_ids),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved diagnostics: {output_path}")
    print(f"saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
