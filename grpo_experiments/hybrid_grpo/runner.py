"""Runner for hybrid GRPO: fresh + best-tree replay with policy IS cycles."""

from __future__ import annotations

import json
import math
import os

import numpy as np

from grpo_experiments.core.build_trainer import build_grpo_trainer
from grpo_experiments.core.on_policy_buffer import run_policy_is_grpo_cycles
from grpo_experiments.core.trainer import GRPOTrainer
from grpo_experiments.train_logging import enrich_dual_scale_metrics, format_hybrid_step_line
from grpo_experiments.hybrid_grpo.config import HybridExperimentConfig
from grpo_experiments.hybrid_grpo.replay import (
    BestTreeReplayBuffer,
    reevaluate_log_paths_pf_hybrid,
    sample_hybrid_replay_batch,
)
from grpo_experiments.hybrid_grpo.replay_schedule import effective_replay_sizes
from grpo_experiments.metrics import OutcomeTracker, batch_diversity_stats, extract_outcome_ids
from grpo_experiments.trajectory_log import TrajectoryLogger
from grpo_experiments.resume import (
    load_epoch_summaries,
    load_generator_checkpoint,
    load_metrics_rows,
    make_training_state,
    maybe_load_trainer_state,
    prepare_resume,
    restore_tracker,
    resolve_output_dir,
    save_training_state,
)
from grpo_experiments.utils import (
    append_jsonl,
    build_random_spec,
    choose_device,
    generate_exploration_spec,
    get_generator_params,
    load_phylogfn_cfg,
    set_seed,
)
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_phylo import RolloutWorker


def _build_grpo_trainer(exp_cfg: HybridExperimentConfig, generator, cfg) -> GRPOTrainer:
    params = get_generator_params(generator)
    if not params:
        raise RuntimeError(f"No trainable parameters found in {type(generator).__name__}.")
    n_params = sum(p.numel() for p in params)
    print(
        f"Hybrid-GRPO optimizer: {len(params)} param groups, {n_params:,} parameters  "
        f"clip_eps={exp_cfg.grpo_clip_eps}  update_cycles={exp_cfg.effective_update_cycles}"
    )
    # Hybrid uses update_cycles for inner policy-IS replay; TRL mu (num_iterations) stays 1.
    return build_grpo_trainer(
        params,
        lr=exp_cfg.grpo_lr,
        clip_eps=exp_cfg.grpo_clip_eps,
        clip_eps_high=exp_cfg.grpo_clip_eps_high,
        max_grad_norm=exp_cfg.grpo_max_grad_norm,
        advantage_eps=exp_cfg.grpo_advantage_eps,
        entropy_coef=exp_cfg.entropy_coef,
        reward_c=cfg.ENV.REWARD.C,
        reward_scale=cfg.ENV.REWARD.SCALE,
        num_iterations=1,
    )


def _save_checkpoint_bundle(
    output_dir: str,
    generator,
    trainer,
    checkpoint_name: str,
    resume_state,
) -> None:
    ckpt_path = os.path.join(output_dir, checkpoint_name)
    generator.save(ckpt_path)
    resume_state.checkpoint_path = ckpt_path
    save_training_state(output_dir, resume_state, grpo_trainer=trainer)


def _init_run_state(exp_cfg: HybridExperimentConfig, output_dir: str, cfg):
    training_cfg = cfg.GFN.TRAINING_DATA_LOADER
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    resume = None
    if exp_cfg.resume_from:
        resume, _ = prepare_resume(
            output_dir,
            checkpoint_name=exp_cfg.resume_checkpoint,
            training_mode="policy_is",
            steps_per_epoch=training_cfg.STEPS_PER_EPOCH,
            update_cycles=exp_cfg.effective_update_cycles,
            target_epochs=training_cfg.EPOCHS_NUM,
            target_resample_rounds=exp_cfg.effective_resample_rounds,
        )
    return resume, metrics_path


def _run_hybrid_policy_is(exp_cfg: HybridExperimentConfig, device: str, output_dir: str, cfg, all_seqs) -> str:
    env = build_env(cfg, all_seqs)
    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    rollout_worker = RolloutWorker(env)

    trainer = _build_grpo_trainer(exp_cfg, generator, cfg)

    replay_buffer = BestTreeReplayBuffer(
        exp_cfg.best_tree_buffer_size,
        topology_only=exp_cfg.best_trees_topology_only,
    )
    if exp_cfg.replay_warmstart_samples > 0:
        warm_added = replay_buffer.warm_start_from_policy(
            rollout_worker,
            generator,
            env,
            num_samples=exp_cfg.replay_warmstart_samples,
            chunk_size=exp_cfg.rollout_chunk_size,
            device=device,
        )
        print(f"warmstarted best-tree buffer: +{warm_added} entries (size={len(replay_buffer)})")

    resume, metrics_path = _init_run_state(exp_cfg, output_dir, cfg)
    traj_logger = TrajectoryLogger(
        output_dir,
        enabled=exp_cfg.log_trajectories,
        flush_every=exp_cfg.trajectory_flush_every,
    )
    if resume is not None:
        load_generator_checkpoint(generator, resume.checkpoint_path)
        maybe_load_trainer_state(trainer, resume, output_dir)

    metrics_rows = load_metrics_rows(metrics_path)
    tracker = restore_tracker(resume, metrics_rows) if resume else OutcomeTracker()
    seen_outcomes: set[str] = set(tracker.outcome_counts)
    epoch_summaries = load_epoch_summaries(output_dir) if resume else []
    training_cfg = cfg.GFN.TRAINING_DATA_LOADER
    global_step = resume.global_step if resume else 0
    start_round = resume.start_resample_round if resume else 0

    for resample_round in range(start_round, exp_cfg.effective_resample_rounds):
        exploration_specs = generate_exploration_spec(training_cfg.EXPLORATION, resample_round)
        random_spec = build_random_spec(
            exploration_specs,
            exp_cfg.effective_update_cycles,
            0,
            training_cfg.EXPLORATION.ANNEAL_TYPE,
        )

        fresh_size, replay_size, _ = effective_replay_sizes(
            resample_round,
            exp_cfg.effective_resample_rounds,
            exp_cfg.fresh_buffer_size,
            exp_cfg.replay_sample_size,
            exp_cfg.replay_anneal_start,
            exp_cfg.replay_anneal_end,
            total_batch=exp_cfg.replay_anneal_total_batch,
            buffer_size=exp_cfg.best_tree_buffer_size,
        )

        print(
            f"--- resample round {resample_round}: "
            f"fresh={fresh_size} replay={replay_size} "
            f"(best-buffer-size={len(replay_buffer)}) ---"
        )
        batch = sample_hybrid_replay_batch(
            rollout_worker,
            generator,
            env,
            replay_buffer=replay_buffer,
            fresh_buffer_size=fresh_size,
            replay_sample_size=replay_size,
            chunk_size=exp_cfg.rollout_chunk_size,
            random_spec=random_spec,
            device=device,
        )
        advantages, advantage_metrics = trainer.precompute_advantages(batch.log_scores)

        outcome_ids, topology_ids = extract_outcome_ids(batch.trees, exp_cfg.outcome_level)
        traj_logger.log_batch(
            global_step=global_step,
            resample_round=resample_round,
            trees=batch.trees,
            source_tags=batch.source_tags,
            update_cycle=0,
            extra={"method": exp_cfg.method, "outcome_level": exp_cfg.outcome_level},
        )

        fresh_count = batch.fresh_count
        add_stats = replay_buffer.add_samples_with_stats(
            trees=batch.trees[:fresh_count],
            actions_set=batch.actions_set[:fresh_count],
            log_paths_pf_old=batch.log_paths_pf_old[:fresh_count],
            log_rewards=batch.log_rewards[:fresh_count],
            log_scores=batch.log_scores[:fresh_count],
        )
        added = add_stats.inserted
        found_in_replay_buffer = add_stats.found_in_buffer
        replay_replaced = add_stats.replaced_existing

        tracker.update(outcome_ids, topology_ids)
        seen_outcomes.update(outcome_ids)
        div = batch_diversity_stats(outcome_ids, topology_ids)
        div["cumulative_unique_outcomes"] = float(len(seen_outcomes))
        div.update(tracker.stats())

        round_losses: list[float] = []
        round_log_rewards: list[float] = []
        cycle_begin = resume.start_update_cycle if (resume and resample_round == start_round) else 0
        if cycle_begin > 0:
            print(
                f"warning: skipping to resample round {resample_round} "
                "(hybrid mid-round resume is not supported)"
            )
            cycle_begin = 0

        cycle_train_infos = run_policy_is_grpo_cycles(
            trainer,
            rollout_worker,
            generator,
            batch,
            advantages=advantages,
            advantage_metrics=advantage_metrics,
            outcome_ids=None,
            update_cycles=exp_cfg.effective_update_cycles - cycle_begin,
            chunk_size=exp_cfg.rollout_chunk_size,
            device=device,
            reevaluate_fn=reevaluate_log_paths_pf_hybrid,
        )
        for cycle_offset, train_info in enumerate(cycle_train_infos):
            cycle = cycle_begin + cycle_offset

            mean_log_reward = float(batch.log_rewards.mean().item())
            replay_fraction = float(batch.replay_count / max(batch.size, 1))
            record = enrich_dual_scale_metrics({
                "resample_round": resample_round,
                "update_cycle": cycle,
                "global_step": global_step,
                "method": exp_cfg.method,
                "training_mode": "policy_is",
                "outcome_level": exp_cfg.outcome_level,
                "batch_size": batch.size,
                "fresh_count": batch.fresh_count,
                "replay_count": batch.replay_count,
                "fresh_buffer_size": fresh_size,
                "replay_sample_size": replay_size,
                "replay_fraction": replay_fraction,
                "best_tree_buffer_size": len(replay_buffer),
                "found_in_replay_buffer": found_in_replay_buffer,
                "replay_replaced": replay_replaced,
                "mean_log_reward": mean_log_reward,
                "mean_log_reward_behavior": mean_log_reward,
                **train_info,
                "policy_update_cycles": exp_cfg.effective_update_cycles,
                **div,
            })
            append_jsonl(metrics_path, record)

            if exp_cfg.print_every > 0 and global_step % exp_cfg.print_every == 0:
                print(
                    format_hybrid_step_line(
                        global_step=global_step,
                        resample_round=resample_round,
                        cycle=cycle,
                        train_info=train_info,
                        mean_log_reward=mean_log_reward,
                        replay_fraction=replay_fraction,
                        added=added,
                        found_in_replay_buffer=found_in_replay_buffer,
                        replay_replaced=replay_replaced,
                        div=div,
                        entropy_coef=exp_cfg.entropy_coef,
                        ips=False,
                    )
                )

            round_losses.append(train_info["loss"])
            round_log_rewards.append(mean_log_reward)
            global_step += 1

        summary = {
            "resample_round": resample_round,
            "method": exp_cfg.method,
            "outcome_level": exp_cfg.outcome_level,
            "mean_loss": float(np.mean(round_losses)) if round_losses else math.nan,
            "mean_log_reward": float(np.mean(round_log_rewards)) if round_log_rewards else math.nan,
            "best_tree_buffer_size": len(replay_buffer),
            "cumulative_unique_outcomes": float(len(seen_outcomes)),
            **tracker.stats(),
        }
        epoch_summaries.append(summary)

        if exp_cfg.checkpoint_every > 0 and (resample_round + 1) % exp_cfg.checkpoint_every == 0:
            ckpt_name = f"checkpoint_round{resample_round:04d}.pt"
            state = make_training_state(
                global_step=global_step,
                training_mode="policy_is",
                resample_round=resample_round,
                update_cycle=exp_cfg.effective_update_cycles - 1,
                update_cycles=exp_cfg.effective_update_cycles,
                tracker=tracker,
            )
            _save_checkpoint_bundle(output_dir, generator, trainer, ckpt_name, state)

    with open(os.path.join(output_dir, "epoch_summaries.json"), "w") as f:
        json.dump(epoch_summaries, f, indent=2)

    final_state = make_training_state(
        global_step=global_step,
        training_mode="policy_is",
        resample_round=exp_cfg.effective_resample_rounds - 1,
        update_cycle=exp_cfg.effective_update_cycles - 1,
        update_cycles=exp_cfg.effective_update_cycles,
        tracker=tracker,
    )
    _save_checkpoint_bundle(output_dir, generator, trainer, "final_checkpoint.pt", final_state)
    traj_logger.close()
    print(f"saved to: {output_dir}")
    return output_dir


def run_experiment(exp_cfg: HybridExperimentConfig) -> str:
    device = choose_device(exp_cfg.device)
    set_seed(exp_cfg.seed)
    print(
        f"method={exp_cfg.method}  mode=policy_is_hybrid  device={device}  "
        f"fresh={exp_cfg.fresh_buffer_size} replay={exp_cfg.replay_sample_size}  "
        f"best_tree_buffer={exp_cfg.best_tree_buffer_size}"
    )
    if exp_cfg.replay_anneal_start is not None and exp_cfg.replay_anneal_end is not None:
        print(
            f"  replay_anneal={exp_cfg.replay_anneal_start}->{exp_cfg.replay_anneal_end}  "
            f"total_batch={exp_cfg.replay_anneal_total_batch}"
        )
    print(
        f"  resample_rounds={exp_cfg.effective_resample_rounds}  "
        f"update_cycles={exp_cfg.effective_update_cycles}  "
        f"chunk={exp_cfg.rollout_chunk_size}  "
        f"entropy_coef={exp_cfg.entropy_coef}"
    )

    cfg, all_seqs = load_phylogfn_cfg(exp_cfg)
    output_dir = resolve_output_dir(exp_cfg)
    cfg.OUTPUT_PATH = output_dir

    if exp_cfg.resume_from:
        print(f"resume: continuing in {output_dir}")
    else:
        exp_cfg.save_json(os.path.join(output_dir, "experiment_config.json"))
        with open(os.path.join(output_dir, "resolved_config.yaml"), "w") as f:
            f.write(cfg.dump())

    return _run_hybrid_policy_is(exp_cfg, device, output_dir, cfg, all_seqs)

