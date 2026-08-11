"""
Training loop for IPS-GRPO experiments.

Two modes (config.enable_policy_is):
  - False: on-policy each step (original IPS-GRPO)
  - True:  sample buffer -> IPS advantages frozen -> inner cycles with pi_new/pi_old
"""

from __future__ import annotations

import json
import math
import os
import pickle

import numpy as np

from grpo_experiments.core.on_policy_buffer import run_on_policy_grpo_step, run_policy_is_grpo_cycles
from grpo_experiments.core.policy_replay import sample_replay_buffer
from grpo_experiments.ips_grpo.config import IPSExperimentConfig
from grpo_experiments.ips_grpo.trainer import IPSGRPOTrainer, TemperedLogIPSGRPOTrainer
from grpo_experiments.ips_grpo.policy_loss_modes import (
    PPO_POLICY_LOSS_MODES,
    SPLIT_CREDIT_POLICY_LOSS_MODES,
    TEMPERED_LOG_IPS_POLICY_LOSS_MODES,
    TERMINAL_POLICY_LOSS_MODES,
)
from grpo_experiments.ips_grpo.trainer_log_ips import IPSLogLossTrainer
from grpo_experiments.metrics import (
    OutcomeIdCache,
    OutcomeTracker,
    batch_diversity_stats,
)
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
    apply_experiment_log_score_discretization,
    build_random_spec,
    choose_device,
    generate_exploration_spec,
    get_generator_params,
    load_phylogfn_cfg,
    resolve_rollout_chunk_size,
    set_seed,
    apply_training_cpu_limits,
)
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.gfn.training_data_loader import TrainingDataLoader


def _build_ips_trainer(exp_cfg: IPSExperimentConfig, params, cfg):
    common = dict(
        params=params,
        lr=exp_cfg.grpo_lr,
        max_grad_norm=exp_cfg.grpo_max_grad_norm,
        ips_prob_floor=exp_cfg.ips_prob_floor,
        reward_c=cfg.ENV.REWARD.C,
        reward_scale=cfg.ENV.REWARD.SCALE,
        entropy_coef=exp_cfg.grpo_entropy_coef,
        num_iterations=exp_cfg.grpo_num_iterations,
        advantage_reward_mode=exp_cfg.advantage_reward_mode,
        ips_propensity_mode=exp_cfg.ips_propensity_mode,
        max_inverse_weight=exp_cfg.max_inverse_weight,
        ips_weight_temperature=exp_cfg.ips_weight_temperature,
        snips_truncate_ratio=exp_cfg.snips_truncate_ratio,
        ips_target_ess_fraction=exp_cfg.ips_target_ess_fraction,
    )
    if exp_cfg.policy_loss_mode in TERMINAL_POLICY_LOSS_MODES:
        print(f"  policy_loss={exp_cfg.policy_loss_mode} (trainer_log_ips.py)")
        return IPSLogLossTrainer(**common, policy_loss_mode=exp_cfg.policy_loss_mode)
    ppo_kwargs = dict(
        advantage_eps=exp_cfg.grpo_advantage_eps,
        clip_eps=exp_cfg.grpo_clip_eps,
        clip_eps_high=exp_cfg.grpo_clip_eps_high,
        policy_loss_mode=exp_cfg.policy_loss_mode,
        tree_loss_weight=exp_cfg.tree_loss_weight,
        edge_loss_weight=exp_cfg.edge_loss_weight,
    )
    if exp_cfg.policy_loss_mode in PPO_POLICY_LOSS_MODES | SPLIT_CREDIT_POLICY_LOSS_MODES:
        loss_src = {
            "ppo": "core/loss.py",
            "split_ppo": "core/loss_split_ppo.py",
            "magnitude_weighted_ppo": "core/loss_magnitude_weighted_ppo.py",
        }[exp_cfg.policy_loss_mode]
        print(f"  policy_loss={exp_cfg.policy_loss_mode} ({loss_src} via IPSGRPOTrainer)")
        return IPSGRPOTrainer(**common, **ppo_kwargs)
    if exp_cfg.policy_loss_mode in TEMPERED_LOG_IPS_POLICY_LOSS_MODES:
        print(
            f"  policy_loss={exp_cfg.policy_loss_mode} "
            "(core/advantages_tempered_log_ips.py + core/loss.py)"
        )
        return TemperedLogIPSGRPOTrainer(
            **common,
            **ppo_kwargs,
            tempered_ips_tau=exp_cfg.tempered_ips_tau,
            tempered_ips_tau_divisor=exp_cfg.tempered_ips_tau_divisor,
        )
    raise ValueError(
        f"Unsupported policy_loss_mode {exp_cfg.policy_loss_mode!r}. "
        f"Choose from: {sorted(PPO_POLICY_LOSS_MODES | SPLIT_CREDIT_POLICY_LOSS_MODES | TEMPERED_LOG_IPS_POLICY_LOSS_MODES | TERMINAL_POLICY_LOSS_MODES)}."
    )


def _save_checkpoint_bundle(
    output_dir: str,
    generator,
    trainer,
    checkpoint_name: str,
    resume_state,
    data_loader: TrainingDataLoader | None = None,
) -> None:
    ckpt_path = os.path.join(output_dir, checkpoint_name)
    temporary_ckpt_path = f"{ckpt_path}.{os.getpid()}.tmp"
    generator.save(temporary_ckpt_path)
    os.replace(temporary_ckpt_path, ckpt_path)
    resume_state.checkpoint_path = ckpt_path
    save_training_state(output_dir, resume_state, grpo_trainer=trainer)
    if data_loader is not None and data_loader.best_state_batch_size > 0:
        best_trees_path = os.path.join(output_dir, "best_trees.pt")
        pickle.dump(data_loader.best_trees, open(best_trees_path, "wb"))


def _init_run_state(exp_cfg: IPSExperimentConfig, output_dir: str, training_mode: str, cfg):
    training_cfg = cfg.GFN.TRAINING_DATA_LOADER
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    resume = None
    if exp_cfg.resume_from:
        resume, _ = prepare_resume(
            output_dir,
            checkpoint_name=exp_cfg.resume_checkpoint,
            training_mode=training_mode,
            steps_per_epoch=training_cfg.STEPS_PER_EPOCH,
            update_cycles=exp_cfg.effective_update_cycles,
            target_epochs=training_cfg.EPOCHS_NUM,
            target_resample_rounds=exp_cfg.effective_resample_rounds,
        )
    return resume, metrics_path


def _run_on_policy(exp_cfg: IPSExperimentConfig, device: str, output_dir: str, cfg, all_seqs) -> str:
    env = build_env(cfg, all_seqs)
    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    rollout_worker = RolloutWorker(env)
    data_loader = TrainingDataLoader(
        cfg, env, rollout_worker, os.path.join(output_dir, "best_trees.pt"),
    )

    params = get_generator_params(generator)
    if not params:
        raise RuntimeError(f"No trainable parameters found in {type(generator).__name__}.")
    print(f"IPS-GRPO (on-policy) optimizer: {len(params)} param groups")

    trainer = _build_ips_trainer(exp_cfg, params, cfg)

    resume, metrics_path = _init_run_state(exp_cfg, output_dir, "on_policy", cfg)
    if resume is not None:
        load_generator_checkpoint(generator, resume.checkpoint_path)
        maybe_load_trainer_state(trainer, resume, output_dir)

    metrics_rows = load_metrics_rows(metrics_path)
    tracker = restore_tracker(resume, metrics_rows) if resume else OutcomeTracker()
    seen_outcomes: set[str] = set(tracker.outcome_counts)
    epoch_summaries = load_epoch_summaries(output_dir) if resume else []
    training_cfg = cfg.GFN.TRAINING_DATA_LOADER
    global_step = resume.global_step if resume else 0
    start_epoch = resume.start_epoch if resume else 0
    start_step = resume.start_step if resume else 0
    generation_state = None
    rollout_chunk = resolve_rollout_chunk_size(exp_cfg)
    # FAST_OUTCOME_ID_CACHE: set PHYLOGFN_FAST_OUTCOME_ID_PARITY=1 to assert
    # exact agreement with the old ETE reconstruction path on every batch.
    outcome_cache = OutcomeIdCache(env)
    check_fast_outcome_parity = os.environ.get("PHYLOGFN_FAST_OUTCOME_ID_PARITY") == "1"
    disable_fast_outcome_cache = os.environ.get("PHYLOGFN_DISABLE_FAST_OUTCOME_ID_CACHE") == "1"

    for epoch in range(start_epoch, training_cfg.EPOCHS_NUM):
        exploration_specs = generate_exploration_spec(training_cfg.EXPLORATION, epoch)
        epoch_losses: list[float] = []
        epoch_log_rewards: list[float] = []
        epoch_batch_unique: list[float] = []
        epoch_log_reward_min = float("inf")
        epoch_log_reward_max = float("-inf")
        step_begin = start_step if epoch == start_epoch else 0

        for step in range(step_begin, data_loader.steps_per_epoch):
            random_spec = data_loader.generate_random_spec(exploration_specs, step)
            batch, trajectories = data_loader.generate_batch(generator, random_spec)
            apply_experiment_log_score_discretization(batch, exp_cfg, cfg)

            outcome_ids, topology_ids = outcome_cache.ids_from_rollout_batch(
                batch,
                trajectories,
                level=exp_cfg.outcome_level,
                disable_fast_cache=disable_fast_outcome_cache,
                check_parity=check_fast_outcome_parity,
            )
            tracker.update(outcome_ids, topology_ids)
            seen_outcomes.update(outcome_ids)

            div = batch_diversity_stats(outcome_ids, topology_ids)
            div["cumulative_unique_outcomes"] = float(len(seen_outcomes))
            div.update(tracker.stats())

            adv_dump_dir = (
                os.path.join(output_dir, "advantage_groups")
                if exp_cfg.dump_advantage_groups
                else None
            )
            extra_update_kwargs = {"outcome_ids": outcome_ids}
            if exp_cfg.ips_propensity_mode == "exact":
                extra_update_kwargs["ips_log_paths_pf"] = batch["log_paths_pf"].detach()

            train_info, generation_state = run_on_policy_grpo_step(
                trainer,
                rollout_worker,
                generator,
                batch,
                trajectories,
                random_spec=random_spec,
                generation_state=generation_state,
                chunk_size=rollout_chunk,
                device=device,
                advantage_dump_dir=adv_dump_dir,
                group_meta={
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "method": exp_cfg.method,
                },
                extra_update_kwargs=extra_update_kwargs,
            )

            mean_log_reward = float(batch["log_rewards"].mean().item())
            mean_log_score = float(batch["log_scores"].mean().item())

            record = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "method": exp_cfg.method,
                "training_mode": "on_policy",
                "outcome_level": exp_cfg.outcome_level,
                "mean_log_reward": mean_log_reward,
                "mean_log_score": mean_log_score,
                **train_info,
                **div,
            }
            append_jsonl(metrics_path, record)

            if exp_cfg.print_every > 0 and global_step % exp_cfg.print_every == 0:
                print(
                    f"step={global_step:04d} loss={train_info['loss']:.4f} "
                    f"log_R={mean_log_reward:.1f} "
                    f"p_hat={train_info.get('ips_prob_mean', 0):.3f} "
                    f"r_tilde={train_info.get('ips_scaled_reward_mean', 0):.1f} "
                    f"adv={train_info['mean_advantage']:.3f}"
                )

            batch_log_rewards = batch["log_rewards"]
            epoch_log_reward_min = min(epoch_log_reward_min, float(batch_log_rewards.min().item()))
            epoch_log_reward_max = max(epoch_log_reward_max, float(batch_log_rewards.max().item()))
            epoch_batch_unique.append(div["batch_unique_outcomes"])

            epoch_losses.append(train_info["loss"])
            epoch_log_rewards.append(mean_log_reward)
            global_step += 1

        start_step = 0

        summary = {
            "epoch": epoch,
            "method": exp_cfg.method,
            "outcome_level": exp_cfg.outcome_level,
            "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else math.nan,
            "mean_log_reward": float(np.mean(epoch_log_rewards)) if epoch_log_rewards else math.nan,
            "min_log_reward": epoch_log_reward_min if epoch_log_rewards else math.nan,
            "max_log_reward": epoch_log_reward_max if epoch_log_rewards else math.nan,
            "mean_batch_unique_outcomes": float(np.mean(epoch_batch_unique)) if epoch_batch_unique else math.nan,
            "cumulative_unique_outcomes": float(len(seen_outcomes)),
            **tracker.stats(),
        }
        epoch_summaries.append(summary)
        print(
            f"--- epoch={epoch:03d} loss={summary['mean_loss']:.4f} "
            f"log_R={summary['mean_log_reward']:.4f} "
            f"log_R_min={summary['min_log_reward']:.4f} "
            f"log_R_max={summary['max_log_reward']:.4f} "
            f"batch_unique={summary['mean_batch_unique_outcomes']:.1f} "
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
            _save_checkpoint_bundle(output_dir, generator, trainer, ckpt_name, state, data_loader)

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
    _save_checkpoint_bundle(output_dir, generator, trainer, "final_checkpoint.pt", final_state, data_loader)
    return output_dir


def _run_policy_is(exp_cfg: IPSExperimentConfig, device: str, output_dir: str, cfg, all_seqs) -> str:
    env = build_env(cfg, all_seqs)
    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    rollout_worker = RolloutWorker(env)

    params = get_generator_params(generator)
    if not params:
        raise RuntimeError(f"No trainable parameters found in {type(generator).__name__}.")
    print(f"IPS-GRPO (+policy IS) optimizer: {len(params)} param groups")

    trainer = _build_ips_trainer(exp_cfg, params, cfg)

    resume, metrics_path = _init_run_state(exp_cfg, output_dir, "policy_is", cfg)
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
    rollout_chunk = resolve_rollout_chunk_size(exp_cfg)
    # FAST_OUTCOME_ID_CACHE: set PHYLOGFN_FAST_OUTCOME_ID_PARITY=1 to assert
    # exact agreement with the old ETE reconstruction path on every buffer.
    outcome_cache = OutcomeIdCache(env)
    check_fast_outcome_parity = os.environ.get("PHYLOGFN_FAST_OUTCOME_ID_PARITY") == "1"
    disable_fast_outcome_cache = os.environ.get("PHYLOGFN_DISABLE_FAST_OUTCOME_ID_CACHE") == "1"

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
            chunk_size=rollout_chunk,
            random_spec=random_spec,
            device=device,
        )
        apply_experiment_log_score_discretization(buffer, exp_cfg, cfg)
        outcome_ids, topology_ids = outcome_cache.ids_from_replay_buffer(
            buffer,
            level=exp_cfg.outcome_level,
            disable_fast_cache=disable_fast_outcome_cache,
            check_parity=check_fast_outcome_parity,
        )
        advantages, ips_metrics = trainer.precompute_advantages(
            buffer.log_scores,
            outcome_ids=outcome_ids,
            log_paths_pf=buffer.log_paths_pf_old,
        )

        tracker.update(outcome_ids, topology_ids)
        seen_outcomes.update(outcome_ids)
        div = batch_diversity_stats(outcome_ids, topology_ids)
        div["cumulative_unique_outcomes"] = float(len(seen_outcomes))
        div.update(tracker.stats())

        round_losses: list[float] = []
        round_log_rewards: list[float] = []
        round_log_reward_min = float(buffer.log_rewards.min().item())
        round_log_reward_max = float(buffer.log_rewards.max().item())
        round_batch_unique = div["batch_unique_outcomes"]
        cycle_begin = resume.start_update_cycle if (resume and resample_round == start_round) else 0
        if cycle_begin > 0:
            print(
                f"warning: skipping to resample round {resample_round} "
                "(policy-IS mid-round resume is not supported)"
            )
            cycle_begin = 0

        cycle_train_infos = run_policy_is_grpo_cycles(
            trainer,
            rollout_worker,
            generator,
            buffer,
            advantages=advantages,
            advantage_metrics=ips_metrics,
            outcome_ids=outcome_ids,
            update_cycles=exp_cfg.effective_update_cycles - cycle_begin,
            chunk_size=rollout_chunk,
            device=device,
        )
        for cycle_offset, train_info in enumerate(cycle_train_infos):
            cycle = cycle_begin + cycle_offset

            mean_log_reward = float(buffer.log_rewards.mean().item())
            record = {
                "resample_round": resample_round,
                "update_cycle": cycle,
                "global_step": global_step,
                "method": exp_cfg.method,
                "training_mode": "policy_is",
                "outcome_level": exp_cfg.outcome_level,
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
                    f"p_hat={train_info.get('ips_prob_mean', 0):.3f} "
                    f"r_tilde={train_info.get('ips_scaled_reward_mean', 0):.1f} "
                    f"adv={train_info['mean_advantage']:.3f}"
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
            "min_log_reward": round_log_reward_min if round_log_rewards else math.nan,
            "max_log_reward": round_log_reward_max if round_log_rewards else math.nan,
            "mean_batch_unique_outcomes": round_batch_unique if round_log_rewards else math.nan,
            "cumulative_unique_outcomes": float(len(seen_outcomes)),
            **tracker.stats(),
        }
        epoch_summaries.append(summary)
        print(
            f"--- round={resample_round:03d} loss={summary['mean_loss']:.4f} "
            f"log_R={summary['mean_log_reward']:.4f} "
            f"log_R_min={summary['min_log_reward']:.4f} "
            f"log_R_max={summary['max_log_reward']:.4f} "
            f"batch_unique={summary['mean_batch_unique_outcomes']:.1f} "
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
    return output_dir


def run_experiment(exp_cfg: IPSExperimentConfig) -> str:
    device = choose_device(exp_cfg.device)
    set_seed(exp_cfg.seed)
    rollout_chunk = resolve_rollout_chunk_size(exp_cfg)
    mode = "policy_is" if exp_cfg.enable_policy_is else "on_policy"
    print(
        f"method={exp_cfg.method}  mode={mode}  device={device}  "
        f"ips_prob_floor={exp_cfg.ips_prob_floor}  "
        f"outcome_level={exp_cfg.outcome_level}  "
        f"policy_loss={exp_cfg.policy_loss_mode}"
    )
    if exp_cfg.enable_policy_is:
        print(
            f"  buffer={exp_cfg.effective_buffer_size}  "
            f"resample_rounds={exp_cfg.effective_resample_rounds}  "
            f"update_cycles={exp_cfg.effective_update_cycles}  "
            f"chunk={rollout_chunk}"
            + (f" (from {exp_cfg.rollout_chunk_size})" if rollout_chunk != exp_cfg.rollout_chunk_size else "")
        )
    else:
        print(f"  G={exp_cfg.grpo_group_size}")

    cfg, all_seqs = load_phylogfn_cfg(exp_cfg)
    apply_training_cpu_limits(exp_cfg, cfg)
    output_dir = resolve_output_dir(exp_cfg)
    cfg.OUTPUT_PATH = output_dir

    if exp_cfg.resume_from:
        print(f"resume: continuing in {output_dir}")
    else:
        exp_cfg.save_json(os.path.join(output_dir, "experiment_config.json"))
        with open(os.path.join(output_dir, "resolved_config.yaml"), "w") as f:
            f.write(cfg.dump())

    if exp_cfg.enable_policy_is:
        return _run_policy_is(exp_cfg, device, output_dir, cfg, all_seqs)
    return _run_on_policy(exp_cfg, device, output_dir, cfg, all_seqs)
