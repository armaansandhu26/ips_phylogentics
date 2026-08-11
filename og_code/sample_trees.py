#!/usr/bin/env python3
"""Sample terminal trees from an og_code training run checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.configs.defaults import get_cfg_defaults
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.utils.utils import correct_cfg_data, load_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample terminal trees from an og_code run checkpoint.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Experiment output directory containing config.yaml and checkpoints/.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/benchmark_datasets/DS1_reduced.pickle"),
        help="Dataset pickle used during training.",
    )
    parser.add_argument(
        "-n",
        "--num-trees",
        type=int,
        required=True,
        help="Number of terminal trees to sample.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: latest checkpoints/checkpoint_*.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <run-dir>/sampled_trees_<n>.json).",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Cap PyTorch/BLAS CPU threads (default: config MAX_CPU_THREADS or 2).",
    )
    return parser.parse_args()


def resolve_checkpoint(run_dir: Path, checkpoint_arg: Path | None) -> Path:
    if checkpoint_arg is not None:
        if not checkpoint_arg.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_arg}")
        return checkpoint_arg

    checkpoints = sorted(run_dir.glob("checkpoints/checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found under {run_dir / 'checkpoints'}")
    return checkpoints[-1]


def tree_to_newick(tree: Any) -> str:
    return str(tree.ete_node.write(format=1)).strip()


def serialize_tree(tree_idx: int, tree: Any, log_reward: float) -> dict[str, Any]:
    return {
        "index": tree_idx,
        "tree_topology_id": getattr(tree, "tree_topology_id", None),
        "signature": getattr(tree, "signature", None),
        "log_score": float(tree.log_score),
        "log_reward": float(log_reward),
        "newick": tree_to_newick(tree),
    }


def sample_terminal_trees(
    rollout_worker: RolloutWorker,
    generator,
    *,
    num_trees: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    generated = 0

    with np.errstate(all="ignore"):
        while generated < num_trees:
            current_batch = min(batch_size, num_trees - generated)
            batch, trajectories = rollout_worker.rollout(
                generator,
                current_batch,
                generate_full_trajectories=True,
            )
            log_rewards = batch["log_rewards"].detach().cpu().tolist()
            batch_trees = [traj.current_state.subtrees[0] for traj in trajectories]

            for tree, log_reward in zip(batch_trees, log_rewards):
                records.append(serialize_tree(generated, tree, log_reward))
                generated += 1
                if generated % 100000 == 0:
                    print(f"sampled {generated}/{num_trees}", flush=True)

    return records


def main() -> None:
    args = parse_args()
    if args.num_trees <= 0:
        raise ValueError("--num-trees must be positive")

    run_dir = args.run_dir.resolve()
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config: {cfg_path}")

    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = (Path(__file__).resolve().parent / dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"missing dataset: {dataset_path}")

    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint.resolve() if args.checkpoint else None)
    output_path = args.output or (run_dir / f"sampled_trees_{args.num_trees}.json")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"run: {run_dir}")
    print(f"config: {cfg_path}")
    print(f"dataset: {dataset_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {args.device}")
    print(f"sampling {args.num_trees} trees")

    all_seqs = load_sequences(str(dataset_path))
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(cfg_path))
    cfg.AMP = False
    cfg = correct_cfg_data(all_seqs, 1, cfg)

    from src.utils.cpu_threads import apply_cpu_thread_limit

    applied_threads = apply_cpu_thread_limit(
        explicit=args.cpu_threads,
        yaml_value=int(getattr(cfg.GFN.TRAINING_DATA_LOADER, "MAX_CPU_THREADS", 2)),
    )
    if applied_threads:
        print(f"cpu_threads={applied_threads}")

    env = build_env(cfg, all_seqs)
    env.to(args.device)
    generator = build_gfn(cfg, env, args.device, ddp=False)
    generator.load(str(checkpoint_path))
    generator.eval()

    rollout_worker = RolloutWorker(env)
    trees = sample_terminal_trees(
        rollout_worker,
        generator,
        num_trees=args.num_trees,
        batch_size=args.batch_size,
    )

    payload = {
        "run_dir": str(run_dir),
        "method": "og_code",
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "device": args.device,
        "num_trees": int(args.num_trees),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "only_train_tree_model": bool(cfg.GFN.MODEL.ONLY_TRAIN_TREE_MODEL),
        "trees": trees,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"saved {len(trees)} trees to {output_path}")


if __name__ == "__main__":
    main()
