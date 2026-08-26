#!/usr/bin/env python3
"""Sample a learned-reverse full model and retain IPS reference diagnostics.

The compact NPZ output stores only the quantities needed to compare checkpoint
frequencies with the shifted-linear target:

    log w(tau) = log R(x) + log q_phi(tau | x) - log P_F(tau).

Unlike the general tree sampler, this avoids materializing one million Newick
strings and Python dictionaries in memory.
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
from grpo_experiments.learned_reverse_ips_grpo import (  # noqa: E402
    TabularTerminalReversePolicy,
    rollout_tree_action_paths,
)
from grpo_experiments.phylo_learned_reverse_policy import (  # noqa: E402
    PhyloLearnedReverseConfig,
    PhyloLearnedReversePolicy,
    path_log_probabilities as mlp_path_log_probabilities,
)
from src.gfn.outcome_ids import OutcomeIdCache  # noqa: E402
from src.gfn.rollout_worker_phylo import RolloutWorker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Learned-reverse run directory or forward checkpoint.",
    )
    parser.add_argument("-n", "--num-trees", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument(
        "--reverse-state",
        type=Path,
        default=None,
        help="Default: <run-dir>/learned_reverse_state.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <run-dir>/sampled_full_diagnostics_<n>.npz.",
    )
    parser.add_argument("--print-every", type=int, default=100_000)
    return parser.parse_args()


def load_reverse_policy(
    state_path: Path,
    *,
    env,
    device: str,
) -> tuple[TabularTerminalReversePolicy | PhyloLearnedReversePolicy, dict]:
    if not state_path.exists():
        raise FileNotFoundError(f"missing learned reverse state: {state_path}")
    state = torch.load(state_path, map_location=device, weights_only=False)
    if state.get("algorithm") != "learned_reverse_ips_grpo":
        raise ValueError(f"unexpected reverse-state algorithm: {state.get('algorithm')!r}")
    reverse_policy_type = str(state.get("reverse_policy_type", "tabular"))
    if reverse_policy_type == "tabular":
        policy: TabularTerminalReversePolicy | PhyloLearnedReversePolicy = (
            TabularTerminalReversePolicy(
                state["trajectories"],
                state["terminal_ids"],
                device=device,
            )
        )
        policy.load_state_dict(state["reverse_policy"])
        policy.eval()
        if policy.normalization_error() > 1e-6:
            raise ValueError("loaded reverse policy is not normalized")
        return policy, state

    if reverse_policy_type != "mlp":
        raise ValueError(f"unsupported reverse_policy_type: {reverse_policy_type!r}")

    reverse_config_payload = state.get("reverse_config") or {}
    reverse_config = PhyloLearnedReverseConfig(**reverse_config_payload)
    reverse_config.validate()
    policy = PhyloLearnedReversePolicy(
        len(env.sequences),
        hidden_size=reverse_config.hidden_size,
        num_layers=reverse_config.num_layers,
    ).to(device)
    policy.load_state_dict(state["reverse_policy"])
    policy.eval()
    return policy, state


def main() -> None:
    args = parse_args()
    if args.num_trees <= 0:
        raise ValueError("--num-trees must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    set_seed(args.seed)
    device = choose_device(args.device)
    artifacts = resolve_run_artifacts(str(args.checkpoint))
    if args.checkpoint_name is not None:
        checkpoint_path = artifacts.root / args.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        artifacts.checkpoint_path = checkpoint_path

    reverse_state_path = args.reverse_state or artifacts.root / "learned_reverse_state.pt"
    reverse_policy_type = str(
        artifacts.config.get(
            "reverse_policy_type",
            "mlp",
        )
    )
    output_path = args.output or (
        artifacts.root / f"sampled_full_diagnostics_{args.num_trees}.npz"
    )
    if output_path.suffix != ".npz":
        raise ValueError("--output must end in .npz")

    reward_target = str(artifacts.config.get("reward_target", ""))
    if reward_target != "shifted_linear":
        raise ValueError(
            "this diagnostic currently expects reward_target=shifted_linear; "
            f"found {reward_target!r}"
        )

    print(f"run: {artifacts.root}")
    print(f"checkpoint: {artifacts.checkpoint_path}")
    print(f"reverse state: {reverse_state_path}")
    print(f"device: {device}")
    print(f"reverse policy: {reverse_policy_type}")
    print(f"sampling {args.num_trees:,} full-model terminal trees")

    cfg, env, generator = load_generator(artifacts, device)
    if bool(cfg.GFN.MODEL.ONLY_TRAIN_TREE_MODEL):
        raise ValueError("resolved checkpoint is tree-only, not a full model")
    reverse_policy = None
    reverse_state: dict = {"reverse_policy_type": reverse_policy_type, "update_step": -1}
    if reverse_policy_type != "uniform":
        reverse_policy, reverse_state = load_reverse_policy(
            reverse_state_path,
            env=env,
            device=device,
        )
        reverse_policy_type = str(reverse_state.get("reverse_policy_type", reverse_policy_type))
    rollout_worker = RolloutWorker(env)
    outcome_id_cache = OutcomeIdCache(env)

    log_scores = np.empty(args.num_trees, dtype=np.float32)
    log_pf = np.empty(args.num_trees, dtype=np.float32)
    log_q_reverse = np.empty(args.num_trees, dtype=np.float32)
    topology_index = np.empty(args.num_trees, dtype=np.int32)
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
            paths = rollout_tree_action_paths(batch)
            if reverse_policy_type == "uniform":
                batch_log_q = batch["log_paths_pb"].sum(dim=-1)
            elif reverse_policy_type == "tabular":
                catalog_indices = reverse_policy.catalog_indices(paths)
                batch_log_q = reverse_policy.log_prob(catalog_indices)
            else:
                _, batch_topology_ids = outcome_id_cache.ids_from_rollout_batch(
                    batch,
                    trajectories,
                    level="signature",
                )
                batch_log_q = mlp_path_log_probabilities(
                    reverse_policy,
                    env,
                    paths,
                    terminal_ids=batch_topology_ids,
                    terminal_log_scores=batch["log_scores"].detach().cpu().tolist(),
                )
            batch_log_pf = batch["log_paths_pf"].sum(dim=-1)
            batch_scores = batch["log_scores"]

            if not bool(torch.isfinite(batch_log_pf).all()):
                raise ValueError("sampled forward log probabilities contain NaN/Inf")
            if not bool(torch.isfinite(batch_log_q).all()):
                raise ValueError("sampled reverse log probabilities contain NaN/Inf")
            if not bool(torch.isfinite(batch_scores).all()):
                raise ValueError("sampled shifted scores contain NaN/Inf")
            if bool(torch.any(batch_scores <= 0.0)):
                n_bad = int(torch.sum(batch_scores <= 0.0).item())
                print(
                    f"warning: clamping {n_bad}/{batch_scores.numel()} "
                    "shifted scores for sampling",
                    flush=True,
                )
                batch_scores = batch_scores.clamp(min=1e-8)

            stop = generated + current_batch
            log_scores[generated:stop] = batch_scores.detach().cpu().numpy()
            log_pf[generated:stop] = batch_log_pf.detach().cpu().numpy()
            log_q_reverse[generated:stop] = batch_log_q.detach().cpu().numpy()

            for offset, trajectory in enumerate(trajectories):
                tree = trajectory.current_state.subtrees[0]
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

    log_target_reward = np.log(log_scores.astype(np.float64))
    log_weights = (
        log_target_reward
        + log_q_reverse.astype(np.float64)
        - log_pf.astype(np.float64)
    )
    maximum = float(log_weights.max())
    scaled_weights = np.exp(log_weights - maximum)
    ess = float(scaled_weights.sum() ** 2 / np.square(scaled_weights).sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        log_score=log_scores,
        log_pf=log_pf,
        log_q_reverse=log_q_reverse,
        topology_index=topology_index,
        topology_ids=np.asarray(topology_ids),
    )
    metadata = {
        "run_dir": str(artifacts.root),
        "method": artifacts.method,
        "checkpoint_path": str(artifacts.checkpoint_path),
        "reverse_state_path": str(reverse_state_path),
        "reverse_update_step": int(reverse_state.get("update_step", -1)),
        "device": device,
        "num_trees": int(args.num_trees),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "reward_target": reward_target,
        "target": "R(x) = shifted_score = 3600 + terminal_log_likelihood",
        "log_score_shift": float(getattr(env, "log_score_shift", 3600.0)),
        "only_train_tree_model": bool(cfg.GFN.MODEL.ONLY_TRAIN_TREE_MODEL),
        "reverse_policy_type": reverse_policy_type,
        "observed_topologies": len(topology_ids),
        "importance_ess": ess,
        "importance_ess_fraction": ess / args.num_trees,
        "log_weight_min": float(log_weights.min()),
        "log_weight_max": float(log_weights.max()),
        "log_weight_std": float(log_weights.std()),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved diagnostics: {output_path}")
    print(f"saved metadata: {metadata_path}")
    print(f"reference-weight ESS: {ess:,.1f}/{args.num_trees:,} ({ess / args.num_trees:.6f})")


if __name__ == "__main__":
    main()
