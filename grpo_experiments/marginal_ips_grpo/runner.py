"""On-policy training loop for marginal (backward-corrected) exact IPS-GRPO.

This is a thin copy of ``grpo_experiments.ips_grpo.runner._run_on_policy`` with a
single behavioural change (marked ``MARGINAL CORRECTION`` below): the trajectory
log-probability fed into the exact inverse-propensity weight is
``log P_F(tau) - log P_B(tau|x)`` instead of ``log P_F(tau)``.

Everything else (trainer, PPO loss, rollout, replay, logging, checkpointing) is
imported unchanged from the existing IPS-GRPO code, so this file is fully
additive and reverting is just deleting the folder.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from grpo_experiments.core.on_policy_buffer import run_on_policy_grpo_step
from grpo_experiments.ips_grpo.runner import _build_ips_trainer, _save_checkpoint_bundle, _init_run_state
from grpo_experiments.marginal_ips_grpo.config import MarginalIPSExperimentConfig
from grpo_experiments.metrics import OutcomeIdCache, OutcomeTracker, batch_diversity_stats
from grpo_experiments.resume import (
    load_epoch_summaries,
    load_generator_checkpoint,
    load_metrics_rows,
    make_training_state,
    maybe_load_trainer_state,
    resolve_output_dir,
    restore_tracker,
)
from grpo_experiments.utils import (
    append_jsonl,
    apply_experiment_log_score_discretization,
    apply_training_cpu_limits,
    choose_device,
    generate_exploration_spec,
    get_generator_params,
    load_phylogfn_cfg,
    resolve_rollout_chunk_size,
    set_seed,
)
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.gfn.training_data_loader import TrainingDataLoader


def _exact_ips_log_paths(batch: dict, backward_correction: bool):
    """Trajectory log-prob tensor (B, T) to feed the exact IPS weight.

    MARGINAL CORRECTION: with ``backward_correction`` we return
    ``log P_F - log P_B``. ``scale_rewards_exact_ips`` sums over the step axis, so
    the per-trajectory weight becomes ``exp(-(log P_F(tau) - log P_B(tau|x)))``,
    i.e. an estimate of ``1 / P(x)`` (the marginal object probability) rather than
    ``1 / P(tau)`` (a single ordering).
    """
    log_pf = batch["log_paths_pf"]
    if not backward_correction:
        return log_pf.detach()
    log_pb = batch["log_paths_pb"]
    if log_pb.shape != log_pf.shape:
        raise ValueError(
            f"log_paths_pb shape {tuple(log_pb.shape)} != log_paths_pf shape "
            f"{tuple(log_pf.shape)}; cannot apply backward correction."
        )
    return (log_pf - log_pb).detach()


def _run_on_policy(exp_cfg: MarginalIPSExperimentConfig, device: str, output_dir: str, cfg, all_seqs) -> str:
    if exp_cfg.ips_propensity_mode != "exact" and exp_cfg.backward_correction:
        raise ValueError(
            "backward_correction=True requires ips_propensity_mode='exact' "
            "(the correction only applies to the exact trajectory weight)."
        )

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
    print(
        f"Marginal IPS-GRPO (on-policy) optimizer: {len(params)} param groups  "
        f"backward_correction={exp_cfg.backward_correction}"
    )

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
                # MARGINAL CORRECTION happens here.
                extra_update_kwargs["ips_log_paths_pf"] = _exact_ips_log_paths(
                    batch, exp_cfg.backward_correction
                )

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
                "backward_correction": exp_cfg.backward_correction,
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
                    f"ESS={train_info.get('ips_ess_fraction', 0):.3f} "
                    f"beta={train_info.get('ips_weight_temperature', 0):.3f} "
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
            "backward_correction": exp_cfg.backward_correction,
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


def run_experiment(exp_cfg: MarginalIPSExperimentConfig) -> str:
    if exp_cfg.enable_policy_is:
        raise NotImplementedError(
            "marginal_ips_grpo currently supports on-policy training only. "
            "Use grpo_experiments.ips_grpo for the policy-IS path."
        )

    device = choose_device(exp_cfg.device)
    set_seed(exp_cfg.seed)
    rollout_chunk = resolve_rollout_chunk_size(exp_cfg)
    print(
        f"method={exp_cfg.method}  mode=on_policy  device={device}  "
        f"propensity={exp_cfg.ips_propensity_mode}  "
        f"backward_correction={exp_cfg.backward_correction}  "
        f"reward_mode={exp_cfg.advantage_reward_mode}  "
        f"outcome_level={exp_cfg.outcome_level}  "
        f"policy_loss={exp_cfg.policy_loss_mode}  G={exp_cfg.grpo_group_size}  chunk={rollout_chunk}"
    )

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

    return _run_on_policy(exp_cfg, device, output_dir, cfg, all_seqs)
