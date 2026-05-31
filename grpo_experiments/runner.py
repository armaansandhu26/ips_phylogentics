"""
Training loop for PhyloGFN (TB) vs GRPO comparison experiments.

Logged metrics (metrics.jsonl, one record per step)
---------------------------------------------------
Training
    loss              — TB MSE loss (phylgfn) or GRPO policy-gradient loss (grpo)
    grad_norm         — gradient L2 norm after clipping
    param_norm        — parameter L2 norm
    mean_log_reward   — mean log R(x) in batch (target: log-likelihood + prior)
    mean_log_score    — mean log-likelihood score (without prior)

GRPO-specific (grpo method only)
    pg_loss           — policy gradient loss
    mean_advantage    — mean group-relative advantage
    std_advantage     — std of advantages (should ≈ 1 before clipping)
    mean_log_pf       — mean sum of forward log-probs per trajectory
    grpo_group_size   — effective G for this update

Diversity (both methods — key for PhyloGFN vs GRPO comparison)
    batch_unique_outcomes          — unique outcomes in this batch
    batch_duplicate_fraction       — fraction of resampled outcomes (mode collapse signal)
    batch_unique_topologies        — unique tree topologies in batch
    batch_duplicate_topology_fraction
    cumulative_unique_outcomes     — unique outcomes seen so far in run
    global_duplicate_fraction      — cumulative duplicate rate across full run
    global_unique_topologies
    global_duplicate_topology_fraction

Epoch summaries (epoch_summaries.json)
    mean_loss, mean_log_reward, cumulative_unique_outcomes, global stats
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from grpo_experiments.config import ExperimentConfig
from grpo_experiments.grpo import GRPOTrainer
from grpo_experiments.metrics import OutcomeTracker, batch_diversity_stats, extract_outcome_ids
from grpo_experiments.policy_replay import reevaluate_log_paths_pf, sample_replay_buffer
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
    reconstruct_trees,
    scalar_metric,
    set_seed,
)
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.gfn.training_data_loader import TrainingDataLoader


def _train_phylgfn_step(generator, batch, mini_batch_splits: int) -> dict:
    generator.accumulate_loss(batch, mini_batch_splits)
    info = generator.update_model()
    return {
        "loss": scalar_metric(info["loss"]),
        "grad_norm": scalar_metric(info["grad_norm"]),
        "param_norm": scalar_metric(info["param_norm"]),
    }


def _train_grpo_step(grpo_trainer: GRPOTrainer, batch: dict) -> dict:
    return grpo_trainer.update_on_policy(batch)


def _build_grpo_trainer(exp_cfg: ExperimentConfig, generator) -> GRPOTrainer:
    params = get_generator_params(generator)
    if not params:
        raise RuntimeError(
            f"No trainable parameters found in {type(generator).__name__}."
        )
    n_params = sum(p.numel() for p in params)
    print(f"GRPO optimizer: {len(params)} param groups, {n_params:,} parameters")
    return GRPOTrainer(
        params=params,
        lr=exp_cfg.grpo_lr,
        clip_eps=exp_cfg.grpo_clip_eps,
        beta=exp_cfg.grpo_beta,
        max_grad_norm=exp_cfg.grpo_max_grad_norm,
        advantage_eps=exp_cfg.grpo_advantage_eps,
        is_ratio_clip=exp_cfg.is_ratio_clip,
        is_ratio_max=exp_cfg.is_ratio_max,
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


def _init_run_state(exp_cfg: ExperimentConfig, output_dir: str, training_mode: str, cfg):
    training_cfg = cfg.GFN.TRAINING_DATA_LOADER
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    resume = None
    if exp_cfg.resume_from:
        resume, ckpt_path = prepare_resume(
            output_dir,
            checkpoint_name=exp_cfg.resume_checkpoint,
            training_mode=training_mode,
            steps_per_epoch=training_cfg.STEPS_PER_EPOCH,
            update_cycles=exp_cfg.effective_update_cycles,
            target_epochs=training_cfg.EPOCHS_NUM,
            target_resample_rounds=exp_cfg.effective_resample_rounds,
        )
    return resume, metrics_path


def _run_grpo_on_policy(exp_cfg: ExperimentConfig, device: str, output_dir: str, cfg, all_seqs) -> str:
    env = build_env(cfg, all_seqs)
    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    rollout_worker = RolloutWorker(env)
    data_loader = TrainingDataLoader(
        cfg, env, rollout_worker, os.path.join(output_dir, "best_trees.pt"),
    )

    grpo_trainer = None
    if exp_cfg.method == "grpo":
        grpo_trainer = _build_grpo_trainer(exp_cfg, generator)

    resume, metrics_path = _init_run_state(exp_cfg, output_dir, "on_policy", cfg)
    if resume is not None:
        load_generator_checkpoint(generator, resume.checkpoint_path)
        maybe_load_trainer_state(grpo_trainer, resume, output_dir)

    metrics_rows = load_metrics_rows(metrics_path)
    tracker = restore_tracker(resume, metrics_rows) if resume else OutcomeTracker()
    seen_outcomes: set[str] = set(tracker.outcome_counts)
    epoch_summaries = load_epoch_summaries(output_dir) if resume else []
    training_cfg = cfg.GFN.TRAINING_DATA_LOADER
    global_step = resume.global_step if resume else 0
    start_epoch = resume.start_epoch if resume else 0
    start_step = resume.start_step if resume else 0

    for epoch in range(start_epoch, training_cfg.EPOCHS_NUM):
        exploration_specs = generate_exploration_spec(training_cfg.EXPLORATION, epoch)
        epoch_losses: list[float] = []
        epoch_log_rewards: list[float] = []
        step_begin = start_step if epoch == start_epoch else 0

        for step in range(step_begin, data_loader.steps_per_epoch):
            random_spec = data_loader.generate_random_spec(exploration_specs, step)
            batch, trajectories = data_loader.generate_batch(generator, random_spec)

            trees = reconstruct_trees(env, trajectories, batch["log_scores"])
            outcome_ids, topology_ids = extract_outcome_ids(trees, exp_cfg.outcome_level)
            tracker.update(outcome_ids, topology_ids)
            seen_outcomes.update(outcome_ids)

            div = batch_diversity_stats(outcome_ids, topology_ids)
            div["cumulative_unique_outcomes"] = float(len(seen_outcomes))
            div.update(tracker.stats())

            if exp_cfg.method == "phylgfn":
                train_info = _train_phylgfn_step(
                    generator, batch, training_cfg.MINI_BATCH_SPLITS,
                )
            else:
                train_info = _train_grpo_step(grpo_trainer, batch)

            mean_log_reward = float(batch["log_rewards"].mean().item())
            mean_log_score = float(batch["log_scores"].mean().item())

            record = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "method": exp_cfg.method,
                "training_mode": "on_policy",
                "mean_log_reward": mean_log_reward,
                "mean_log_score": mean_log_score,
                **train_info,
                **div,
            }
            append_jsonl(metrics_path, record)

            if exp_cfg.print_every > 0 and global_step % exp_cfg.print_every == 0:
                adv = ""
                if "mean_advantage" in train_info:
                    adv = (
                        f" adv={train_info['mean_advantage']:.3f}"
                        f"±{train_info.get('std_advantage', 0):.3f}"
                    )
                print(
                    f"step={global_step:04d} loss={train_info['loss']:.4f} "
                    f"log_R={mean_log_reward:.1f} "
                    f"batch_dup={div['batch_duplicate_fraction']:.3f} "
                    f"global_dup={div['global_duplicate_fraction']:.3f} "
                    f"seen={int(div['cumulative_unique_outcomes'])}"
                    f"{adv}"
                )

            epoch_losses.append(train_info["loss"])
            epoch_log_rewards.append(mean_log_reward)
            global_step += 1

        start_step = 0

        if exp_cfg.method == "phylgfn" and hasattr(generator, "loss") and generator.loss != 0:
            generator.update_model()

        summary = {
            "epoch": epoch,
            "method": exp_cfg.method,
            "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else math.nan,
            "mean_log_reward": float(np.mean(epoch_log_rewards)) if epoch_log_rewards else math.nan,
            "cumulative_unique_outcomes": float(len(seen_outcomes)),
            **tracker.stats(),
        }
        epoch_summaries.append(summary)

        print(
            f"--- epoch={epoch:03d} method={exp_cfg.method} "
            f"loss={summary['mean_loss']:.4f} "
            f"log_R={summary['mean_log_reward']:.4f} "
            f"unique_seen={summary['cumulative_unique_outcomes']:.0f} "
            f"global_dup={summary['global_duplicate_fraction']:.3f}"
        )

        if exp_cfg.checkpoint_every > 0 and (epoch + 1) % exp_cfg.checkpoint_every == 0:
            ckpt_name = f"checkpoint_epoch{epoch:04d}.pt"
            state = make_training_state(
                global_step=global_step,
                training_mode="on_policy",
                epoch=epoch,
                step=data_loader.steps_per_epoch - 1,
                steps_per_epoch=data_loader.steps_per_epoch,
                tracker=tracker,
            )
            _save_checkpoint_bundle(output_dir, generator, grpo_trainer, ckpt_name, state)

    with open(os.path.join(output_dir, "epoch_summaries.json"), "w") as f:
        json.dump(epoch_summaries, f, indent=2)

    final_state = make_training_state(
        global_step=global_step,
        training_mode="on_policy",
        epoch=training_cfg.EPOCHS_NUM - 1,
        step=data_loader.steps_per_epoch - 1,
        steps_per_epoch=data_loader.steps_per_epoch,
        tracker=tracker,
    )
    _save_checkpoint_bundle(output_dir, generator, grpo_trainer, "final_checkpoint.pt", final_state)
    print(f"saved to: {output_dir}")
    return output_dir


def _run_grpo_policy_is(exp_cfg: ExperimentConfig, device: str, output_dir: str, cfg, all_seqs) -> str:
    env = build_env(cfg, all_seqs)
    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    rollout_worker = RolloutWorker(env)

    grpo_trainer = _build_grpo_trainer(exp_cfg, generator)

    resume, metrics_path = _init_run_state(exp_cfg, output_dir, "policy_is", cfg)
    if resume is not None:
        load_generator_checkpoint(generator, resume.checkpoint_path)
        maybe_load_trainer_state(grpo_trainer, resume, output_dir)

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

        print(
            f"--- resample round {resample_round}: "
            f"sampling {exp_cfg.effective_buffer_size} trees (behavior policy) ---"
        )
        buffer = sample_replay_buffer(
            rollout_worker,
            generator,
            buffer_size=exp_cfg.effective_buffer_size,
            chunk_size=exp_cfg.rollout_chunk_size,
            random_spec=random_spec,
            device=device,
        )
        log_pf_old = buffer.log_pf_old
        advantages = grpo_trainer.compute_advantages(buffer.log_rewards)

        trees = reconstruct_trees(env, buffer.trajectories, buffer.log_scores)
        outcome_ids, topology_ids = extract_outcome_ids(trees, exp_cfg.outcome_level)
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
                "(policy-IS mid-round resume is not supported)"
            )
            cycle_begin = 0

        for cycle in range(cycle_begin, exp_cfg.effective_update_cycles):
            log_paths_pf = reevaluate_log_paths_pf(
                rollout_worker,
                generator,
                buffer,
                chunk_size=exp_cfg.rollout_chunk_size,
                device=device,
            )
            train_info = grpo_trainer.update(
                log_paths_pf,
                buffer.log_rewards,
                log_pf_old=log_pf_old,
                fixed_advantages=advantages,
            )

            mean_log_reward = float(buffer.log_rewards.mean().item())
            record = {
                "resample_round": resample_round,
                "update_cycle": cycle,
                "global_step": global_step,
                "method": exp_cfg.method,
                "training_mode": "policy_is",
                "buffer_size": exp_cfg.effective_buffer_size,
                "mean_log_reward": mean_log_reward,
                "mean_log_reward_behavior": mean_log_reward,
                **train_info,
                **div,
            }
            append_jsonl(metrics_path, record)

            if exp_cfg.print_every > 0 and global_step % exp_cfg.print_every == 0:
                print(
                    f"step={global_step:04d} round={resample_round} cycle={cycle} "
                    f"loss={train_info['loss']:.4f} "
                    f"w={train_info.get('mean_importance_ratio', 1):.3f} "
                    f"log_R={mean_log_reward:.1f} "
                    f"adv={train_info['mean_advantage']:.3f}"
                )

            round_losses.append(train_info["loss"])
            round_log_rewards.append(mean_log_reward)
            global_step += 1

        summary = {
            "resample_round": resample_round,
            "method": exp_cfg.method,
            "mean_loss": float(np.mean(round_losses)) if round_losses else math.nan,
            "mean_log_reward": float(np.mean(round_log_rewards)) if round_log_rewards else math.nan,
            "cumulative_unique_outcomes": float(len(seen_outcomes)),
            **tracker.stats(),
        }
        epoch_summaries.append(summary)
        print(
            f"--- round={resample_round:03d} loss={summary['mean_loss']:.4f} "
            f"log_R={summary['mean_log_reward']:.4f} "
            f"unique_seen={summary['cumulative_unique_outcomes']:.0f} "
            f"global_dup={summary['global_duplicate_fraction']:.3f}"
        )

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
            _save_checkpoint_bundle(output_dir, generator, grpo_trainer, ckpt_name, state)

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
    _save_checkpoint_bundle(output_dir, generator, grpo_trainer, "final_checkpoint.pt", final_state)
    print(f"saved to: {output_dir}")
    return output_dir


def run_experiment(exp_cfg: ExperimentConfig) -> str:
    device = choose_device(exp_cfg.device)
    set_seed(exp_cfg.seed)
    if exp_cfg.method == "grpo" and exp_cfg.enable_policy_is:
        mode = "policy_is"
        print(
            f"method={exp_cfg.method}  mode={mode}  device={device}  "
            f"buffer={exp_cfg.effective_buffer_size}  "
            f"resample_rounds={exp_cfg.effective_resample_rounds}  "
            f"update_cycles={exp_cfg.effective_update_cycles}  "
            f"chunk={exp_cfg.rollout_chunk_size}"
        )
    else:
        print(f"method={exp_cfg.method}  device={device}  G={exp_cfg.grpo_group_size}")

    cfg, all_seqs = load_phylogfn_cfg(exp_cfg)
    output_dir = resolve_output_dir(exp_cfg)
    cfg.OUTPUT_PATH = output_dir

    if exp_cfg.resume_from:
        print(f"resume: continuing in {output_dir}")
    else:
        exp_cfg.save_json(os.path.join(output_dir, "experiment_config.json"))
        with open(os.path.join(output_dir, "resolved_config.yaml"), "w") as f:
            f.write(cfg.dump())

    if exp_cfg.method == "phylgfn":
        return _run_grpo_on_policy(exp_cfg, device, output_dir, cfg, all_seqs)
    if exp_cfg.enable_policy_is:
        return _run_grpo_policy_is(exp_cfg, device, output_dir, cfg, all_seqs)
    return _run_grpo_on_policy(exp_cfg, device, output_dir, cfg, all_seqs)
