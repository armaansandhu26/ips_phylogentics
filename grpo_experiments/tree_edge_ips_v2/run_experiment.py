from __future__ import annotations

import json
import math
import os
from datetime import datetime

import numpy as np
import torch

from grpo_experiments.core.policy_replay import ReplayBuffer, reevaluate_log_paths_pf, sample_replay_buffer
from grpo_experiments.metrics import batch_diversity_stats
from grpo_experiments.tree_edge_ips_v2.config import TrainConfig
from grpo_experiments.tree_edge_ips_v2.ips_grpo import linear_rewards_from_log_scores
from grpo_experiments.tree_edge_ips_v2.trainer import TreeEdgeIPSGRPOTrainer
from grpo_experiments.utils import (
    append_jsonl,
    apply_training_cpu_limits,
    choose_device,
    get_generator_params,
    load_phylogfn_cfg,
    resolve_rollout_chunk_size,
    set_seed,
)
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_phylo import RolloutWorker


INTEGRATION_STATUS = {
    "status": "connected_on_policy_training",
    "connected_components": [
        "Phylo env and TBGFlowNetGenerator construction",
        "On-policy rollout collection through RolloutWorker",
        "Fixed-action replay for current tree/edge log-probs",
        "Exact IPS with optional backward correction and ESS-target / tempered SNIPS",
        "SNIPS group advantages",
        "Split tree/edge PPO update",
        "Periodic on-policy eval sampling",
        "CPU thread cap, metrics, and checkpoints",
    ],
    "intentional_limitations": [
        "Best-tree replay is not mixed into v2 exact IPS updates.",
        "Strict [rep; one_hot(tree_action)] edge input is not enabled for the existing phylo edge model.",
        "Aux heads are not attached to the transformer tree model.",
        "edge_credit=counterfactual is not implemented (trajectory credit only).",
    ],
    "progress_metrics": {
        "reward_quality": [
            "mean/max/min/std/p05/p50/p95_log_score",
            "mean/max/min_log_reward",
        ],
        "ips_health": [
            "ips_ess_mean / ips_ess_fraction_mean (want ~ target)",
            "ips_active_mean (≈0 => inert/uniform SNIPS; larger => real reweighting)",
            "ips_weight_temperature_mean / ips_solved_temperature_mean",
            "ips_snips_weight_min/max/std",
            "ips_clipped_fraction_mean / ips_legacy_absolute_cap_mean",
            "ips_log_prop_min/max (spread of -log pi)",
            "mean_log_prob_ips_old / mean_log_pb when backward_correction=True",
        ],
        "policy_update": [
            "tree/edge_policy_loss",
            "tree/edge_ratio_mean",
            "tree/edge_clipped_fraction",
            "grad_norm",
            "entropy_bonus",
            "mean_delta_log_pf_tree/edge (new-old)",
        ],
        "diversity": [
            "batch_unique_outcomes",
            "batch_unique_topologies",
            "batch_duplicate_fraction / batch_duplicate_topology_fraction",
        ],
        "eval": [
            "eval_mean/max/p95_log_score",
            "eval_unique_topologies / eval_unique_signatures",
        ],
    },
}


def make_output_dir(config: TrainConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config.run_name or "tree_edge_ips_v2"
    output_dir = os.path.join(config.output_root, f"{timestamp}_{name}")
    os.makedirs(output_dir, exist_ok=False)
    return output_dir


def _edge_value_key(value):
    if isinstance(value, list):
        return tuple(_edge_value_key(x) for x in value)
    if isinstance(value, tuple):
        return tuple(_edge_value_key(x) for x in value)
    if isinstance(value, float):
        return round(value, 8)
    return int(value)


def _canonical_outcome_keys(buffer: ReplayBuffer) -> list[tuple[tuple[int, ...], tuple]]:
    action_tensors = buffer.action_tensors
    actions_set = action_tensors.to_actions_set() if action_tensors is not None else buffer.actions_set
    keys = []
    for actions in actions_set:
        tree_path = tuple(int(action["tree_action"]) for action in actions)
        edge_sequence = tuple(_edge_value_key(action.get("edge_action", 0)) for action in actions)
        keys.append((tree_path, edge_sequence))
    return keys


def _tensor_quantiles(values: torch.Tensor, qs=(0.05, 0.5, 0.95)) -> dict[str, float]:
    cpu = values.detach().float().reshape(-1).cpu()
    quantiles = torch.quantile(cpu, torch.tensor(list(qs), dtype=cpu.dtype))
    return {
        "p05": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "mean": float(cpu.mean().item()),
        "std": float(cpu.std(unbiased=False).item()),
        "min": float(cpu.min().item()),
        "max": float(cpu.max().item()),
    }


def _group_advantages(
    trainer: TreeEdgeIPSGRPOTrainer,
    config: TrainConfig,
    log_scores: torch.Tensor,
    old_log_prob_joint: torch.Tensor,
    outcome_keys: list,
    *,
    reward_c: float,
    reward_scale: float,
) -> tuple[torch.Tensor, dict]:
    all_advantages = []
    group_metrics = []
    batch_size = int(log_scores.shape[0])
    if batch_size % config.group_size != 0:
        raise ValueError(
            f"batch size {batch_size} must be divisible by group_size {config.group_size}."
        )

    for group_idx, start in enumerate(range(0, batch_size, config.group_size)):
        end = start + config.group_size
        group_scores = log_scores[start:end]
        returns = linear_rewards_from_log_scores(
            group_scores,
            reward_c=reward_c,
            reward_scale=reward_scale,
            mode=config.reward_mode,
        )
        advantages, metrics = trainer.compute_group_advantages(
            returns,
            old_log_prob_joint[start:end],
            outcome_ids=outcome_keys[start:end],
        )
        all_advantages.append(advantages)
        prefixed = {f"group{group_idx}_{key}": value for key, value in metrics.items()}
        group_metrics.append(prefixed)

    merged: dict[str, float | str | None] = {}
    numeric_by_key: dict[str, list[float]] = {}
    for metrics in group_metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None:
                if isinstance(value, float) and math.isnan(value):
                    continue
                base_key = key.split("_", 1)[1]
                numeric_by_key.setdefault(base_key, []).append(float(value))
    for key, values in numeric_by_key.items():
        merged[f"ips_{key}_mean"] = float(np.mean(values))
        merged[f"ips_{key}_min"] = float(np.min(values))
        merged[f"ips_{key}_max"] = float(np.max(values))
    advantages_all = torch.cat(all_advantages, dim=0)
    adv_stats = _tensor_quantiles(advantages_all)
    merged.update(
        {
            "advantage_min": adv_stats["min"],
            "advantage_max": adv_stats["max"],
            "advantage_p05": adv_stats["p05"],
            "advantage_p50": adv_stats["p50"],
            "advantage_p95": adv_stats["p95"],
            "advantage_std": adv_stats["std"],
        }
    )
    merged["ips_num_groups"] = float(len(group_metrics))
    return advantages_all, merged


def _save_checkpoint(
    output_dir: str,
    name: str,
    generator,
    trainer: TreeEdgeIPSGRPOTrainer,
    config: TrainConfig,
    update: int,
) -> None:
    generator.save(os.path.join(output_dir, name))
    torch.save(
        {
            "trainer_state_dict": trainer.state_dict(),
            "config": config.to_dict(),
            "update": update,
        },
        os.path.join(output_dir, f"{name}.trainer"),
    )


@torch.no_grad()
def _run_eval_probe(
    rollout_worker: RolloutWorker,
    generator,
    *,
    num_episodes: int,
    batch_size: int,
) -> dict[str, float]:
    """Lightweight on-policy sampling probe for progress tracking."""
    log_scores: list[float] = []
    log_rewards: list[float] = []
    topologies: list[str] = []
    signatures: list[str] = []
    generated = 0
    while generated < num_episodes:
        current = min(batch_size, num_episodes - generated)
        batch, trajectories = rollout_worker.rollout(
            generator,
            current,
            generate_full_trajectories=True,
        )
        batch_log_scores = batch["log_scores"].detach().cpu().tolist()
        batch_log_rewards = batch["log_rewards"].detach().cpu().tolist()
        for traj, ls, lr in zip(trajectories, batch_log_scores, batch_log_rewards):
            tree = traj.current_state.subtrees[0]
            log_scores.append(float(ls))
            log_rewards.append(float(lr))
            topologies.append(str(getattr(tree, "tree_topology_id", "unknown")))
            signatures.append(str(getattr(tree, "signature", topologies[-1])))
            generated += 1

    scores = torch.tensor(log_scores, dtype=torch.float32)
    rewards = torch.tensor(log_rewards, dtype=torch.float32)
    score_stats = _tensor_quantiles(scores)
    reward_stats = _tensor_quantiles(rewards)
    return {
        "eval_episodes": float(num_episodes),
        "eval_mean_log_score": score_stats["mean"],
        "eval_std_log_score": score_stats["std"],
        "eval_min_log_score": score_stats["min"],
        "eval_max_log_score": score_stats["max"],
        "eval_p05_log_score": score_stats["p05"],
        "eval_p50_log_score": score_stats["p50"],
        "eval_p95_log_score": score_stats["p95"],
        "eval_mean_log_reward": reward_stats["mean"],
        "eval_max_log_reward": reward_stats["max"],
        "eval_unique_topologies": float(len(set(topologies))),
        "eval_unique_signatures": float(len(set(signatures))),
        "eval_topology_duplicate_fraction": float(1.0 - (len(set(topologies)) / max(len(topologies), 1))),
        "eval_signature_duplicate_fraction": float(1.0 - (len(set(signatures)) / max(len(signatures), 1))),
    }


def run_experiment(config: TrainConfig) -> str:
    """Run connected on-policy tree/edge IPS-GRPO v2 training."""
    device = choose_device(config.device)
    set_seed(config.seed)
    cfg, all_seqs = load_phylogfn_cfg(config)
    applied_threads = apply_training_cpu_limits(config, cfg)
    output_dir = make_output_dir(config)
    cfg.OUTPUT_PATH = output_dir
    config.save_json(os.path.join(output_dir, "experiment_config.json"))
    with open(os.path.join(output_dir, "resolved_config.yaml"), "w") as f:
        f.write(cfg.dump())
    with open(os.path.join(output_dir, "integration_status.json"), "w") as f:
        json.dump(INTEGRATION_STATUS, f, indent=2)

    env = build_env(cfg, all_seqs)
    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    rollout_worker = RolloutWorker(env)
    params = get_generator_params(generator)
    if not params:
        raise RuntimeError(f"No trainable parameters found in {type(generator).__name__}.")
    trainer = TreeEdgeIPSGRPOTrainer(params, config)

    rollout_chunk = resolve_rollout_chunk_size(config)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    reward_c = float(getattr(cfg.ENV.REWARD, "C", config.reward_c))
    reward_scale = float(getattr(cfg.ENV.REWARD, "SCALE", config.reward_scale))

    print(f"Created IPS-GRPO v2 run folder: {output_dir}")
    print(f"CPU thread cap: {applied_threads if applied_threads > 0 else 'disabled'}")
    print(
        f"device={device} batch={config.batch_size()} groups={config.num_groups} "
        f"group_size={config.group_size} chunk={rollout_chunk} "
        f"detach_edge_rep={config.detach_edge_rep} reward_mode={config.reward_mode}"
    )
    print(
        f"IPS: mode={config.propensity_mode} "
        f"backward_correction={config.backward_correction} "
        f"target_ess={config.ips_target_ess_fraction} "
        f"temp={config.ips_weight_temperature} "
        f"truncate={config.snips_truncate_ratio}"
    )
    print("Training loop: on-policy tempered/ESS-target IPS + split tree/edge PPO.")

    for update in range(config.num_updates):
        buffer = sample_replay_buffer(
            rollout_worker,
            generator,
            buffer_size=config.batch_size(),
            chunk_size=rollout_chunk,
            random_spec=None,
            device=device,
        )
        outcome_keys = _canonical_outcome_keys(buffer)
        old_tree = buffer.log_paths_pf_tree_old
        old_edge = buffer.log_paths_pf_edge_old
        if old_tree is None or old_edge is None:
            raise RuntimeError("v2 training requires split old tree/edge log-probs.")
        old_joint = (old_tree + old_edge).sum(dim=-1)
        old_log_prob_for_ips = old_joint
        log_pb = None
        if config.backward_correction and config.propensity_mode == "exact":
            if buffer.log_paths_pb is None:
                raise RuntimeError("backward_correction=True requires rollout log_paths_pb.")
            log_pb = buffer.log_paths_pb.detach().sum(dim=-1)
            old_log_prob_for_ips = old_joint - log_pb
        advantages, ips_metrics = _group_advantages(
            trainer,
            config,
            buffer.log_scores,
            old_log_prob_for_ips,
            outcome_keys,
            reward_c=reward_c,
            reward_scale=reward_scale,
        )

        log_paths_pf, paths_entropy, log_tree, log_edge = reevaluate_log_paths_pf(
            rollout_worker,
            generator,
            buffer,
            chunk_size=rollout_chunk,
            device=device,
            return_split=True,
        )
        train_info = trainer.update_from_log_probs(
            log_prob_tree=log_tree,
            log_prob_edge=log_edge,
            old_log_prob_tree=old_tree.detach(),
            old_log_prob_edge=old_edge.detach(),
            advantage_tree=advantages,
            entropy_tree=paths_entropy,
        )

        log_rewards = buffer.log_rewards.detach()
        log_scores = buffer.log_scores.detach()
        score_stats = _tensor_quantiles(log_scores)
        reward_stats = _tensor_quantiles(log_rewards)
        with torch.no_grad():
            new_joint = log_paths_pf.sum(dim=-1).detach()
            delta_tree = (log_tree.detach() - old_tree.detach()).sum(dim=-1)
            delta_edge = (log_edge.detach() - old_edge.detach()).sum(dim=-1)
            mean_log_pf_tree = float(log_tree.detach().sum(dim=-1).mean().item())
            mean_log_pf_edge = float(log_edge.detach().sum(dim=-1).mean().item())

        div = batch_diversity_stats(
            [str(key) for key in outcome_keys],
            [str(key[0]) for key in outcome_keys],
        )
        record = {
            "update": update,
            "method": "tree_edge_ips_v2",
            "training_mode": "on_policy_exact_ips",
            "batch_size": config.batch_size(),
            "group_size": config.group_size,
            "num_groups": config.num_groups,
            "reward_mode": config.reward_mode,
            "backward_correction": config.backward_correction,
            "mean_log_reward": reward_stats["mean"],
            "std_log_reward": reward_stats["std"],
            "max_log_reward": reward_stats["max"],
            "min_log_reward": reward_stats["min"],
            "mean_log_score": score_stats["mean"],
            "std_log_score": score_stats["std"],
            "max_log_score": score_stats["max"],
            "min_log_score": score_stats["min"],
            "p05_log_score": score_stats["p05"],
            "p50_log_score": score_stats["p50"],
            "p95_log_score": score_stats["p95"],
            "mean_log_prob_joint_old": float(old_joint.mean().item()),
            "mean_log_prob_ips_old": float(old_log_prob_for_ips.mean().item()),
            "mean_log_pb": float(log_pb.mean().item()) if log_pb is not None else 0.0,
            "mean_log_prob_joint_new": float(new_joint.mean().item()),
            "mean_log_pf_tree": mean_log_pf_tree,
            "mean_log_pf_edge": mean_log_pf_edge,
            "mean_delta_log_pf_tree": float(delta_tree.mean().item()),
            "mean_delta_log_pf_edge": float(delta_edge.mean().item()),
            "mean_entropy": float(paths_entropy.detach().mean().item()),
            **train_info,
            **ips_metrics,
            **div,
        }

        should_eval = config.eval_every > 0 and (
            update == 0 or (update + 1) % config.eval_every == 0
        )
        if should_eval:
            eval_metrics = _run_eval_probe(
                rollout_worker,
                generator,
                num_episodes=config.eval_episodes,
                batch_size=config.eval_batch_size,
            )
            record.update(eval_metrics)

        append_jsonl(metrics_path, record)

        should_print = config.print_every > 0 and (update % config.print_every == 0)
        if should_print:
            ess = float(record.get("ips_ess_mean", math.nan))
            ess_frac = float(record.get("ips_ess_fraction_mean", math.nan))
            active = float(record.get("ips_active_mean", math.nan))
            temp = record.get("ips_solved_temperature_mean")
            if temp is None:
                temp = record.get("ips_weight_temperature_mean", math.nan)
            temp_f = float(temp) if temp is not None else float("nan")
            print(
                f"update={update:05d} "
                f"log_score={record['mean_log_score']:.2f} "
                f"(p50={record['p50_log_score']:.2f} p95={record['p95_log_score']:.2f} max={record['max_log_score']:.2f}) "
                f"ess={ess:.1f}/{config.group_size} ({ess_frac:.2f}) "
                f"ips_active={active:.3f} beta={temp_f:.3f} "
                f"uniq_top={div['batch_unique_topologies']:.0f} "
                f"uniq_out={div['batch_unique_outcomes']:.0f} "
                f"grad={record['grad_norm']:.3f}"
            )
            if should_eval:
                print(
                    f"  eval@{update:05d} "
                    f"mean={record['eval_mean_log_score']:.2f} "
                    f"p95={record['eval_p95_log_score']:.2f} "
                    f"max={record['eval_max_log_score']:.2f} "
                    f"topologies={record['eval_unique_topologies']:.0f} "
                    f"signatures={record['eval_unique_signatures']:.0f}"
                )

        if config.checkpoint_every > 0 and (update + 1) % config.checkpoint_every == 0:
            _save_checkpoint(
                output_dir,
                f"checkpoint_update{update + 1:06d}.pt",
                generator,
                trainer,
                config,
                update + 1,
            )

    _save_checkpoint(output_dir, "final_checkpoint.pt", generator, trainer, config, config.num_updates)
    return output_dir
