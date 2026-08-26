from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from grpo_experiments.core.on_policy_buffer import run_on_policy_grpo_step
from grpo_experiments.core.trainer import GRPOTrainer
from grpo_experiments.metrics import OutcomeTracker
from grpo_experiments.resume import (
    load_epoch_summaries,
    load_generator_checkpoint,
    load_metrics_rows,
    make_training_state,
    prepare_resume,
    resolve_output_dir,
    restore_tracker,
    save_training_state,
)
from grpo_experiments.utils import (
    append_jsonl,
    apply_training_cpu_limits,
    build_output_dir,
    choose_device,
    generate_exploration_spec,
    get_generator_params,
    load_phylogfn_cfg,
    resolve_rollout_chunk_size,
    set_seed,
)
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.outcome_ids import OutcomeIdCache
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.gfn.training_data_loader import TrainingDataLoader

from learned_reverse_ips.advantages import RunningLogWeightNormalizer, learned_reverse_advantages
from learned_reverse_ips.checkpoint import (
    DEFAULT_REVERSE_POLICY_TYPE,
    buffered_best_trees,
    load_learned_reverse_state,
    paired_learned_reverse_state,
    replay_batch_metrics,
    save_best_trees,
    save_learned_reverse_state,
    write_catalog,
)
from learned_reverse_ips.config import LearnedReverseExperimentConfig
from learned_reverse_ips.mlp_policy import (
    PhyloLearnedReverseConfig,
    PhyloLearnedReversePolicy,
    build_reverse_batch,
    path_log_probabilities as mlp_path_log_probabilities,
    update_mlp_reverse_policy,
)
from learned_reverse_ips.reverse_policy import (
    rollout_tree_action_paths,
    trajectory_indices_from_paths,
)


def validate_config(config: LearnedReverseExperimentConfig) -> None:
    if config.replay_batch_size > 0:
        config.disable_replay = False
    if config.enable_policy_is:
        raise ValueError("learned reverse IPS currently supports on-policy training only")
    if config.disable_replay and config.replay_batch_size > 0:
        raise ValueError("replay batch size is > 0 but replay remains disabled")
    if config.effective_replay_batch_size > 0 and config.replay_buffer_size < 1:
        raise ValueError("replay buffer size must be >= 1 when replay is enabled")
    if config.grpo_group_size < 1:
        raise ValueError("on-policy plus replay batch sizes must sum to at least 1")
    if config.grpo_num_iterations != 1:
        raise ValueError("learned reverse update ordering requires --grpo-num-iterations 1")
    if config.policy_loss_mode != "ppo":
        raise ValueError("learned reverse experiment currently requires --policy-loss-mode ppo")
    uniform_reverse = config.reverse_policy_type == "uniform"
    if not uniform_reverse and config.reverse_lr <= 0.0:
        raise ValueError("reverse learning rate must be positive")
    if config.reverse_train_epochs < 0:
        raise ValueError("reverse train epochs must be >= 0")
    if not uniform_reverse and config.reverse_train_epochs < 1:
        raise ValueError("reverse train epochs must be at least 1 for mlp reverse policy")
    if uniform_reverse and config.reverse_train_epochs != 0:
        raise ValueError("uniform reverse policy requires --reverse-train-epochs 0")
    if not uniform_reverse and config.reverse_grad_clip_norm <= 0.0:
        raise ValueError("reverse gradient clipping norm must be positive")
    if config.advantage_normalization not in {"batch", "running"}:
        raise ValueError("advantage normalization must be batch or running")
    if not uniform_reverse and (
        config.reverse_hidden_size < 1 or config.reverse_num_layers < 1
    ):
        raise ValueError("reverse MLP dimensions must be >= 1")
    if config.reward_target not in {"likelihood", "shifted_linear"}:
        raise ValueError("reward target must be likelihood or shifted_linear")
    if config.only_train_tree_model:
        raise ValueError("learned reverse IPS requires the full tree+edge model")


def run_experiment(config: LearnedReverseExperimentConfig) -> str:
    validate_config(config)
    set_seed(config.seed)
    device = choose_device(config.device)
    cfg, all_seqs = load_phylogfn_cfg(config)
    apply_training_cpu_limits(config, cfg)

    env = build_env(cfg, all_seqs)
    num_taxa = len(env.sequences)
    uniform_reverse = config.reverse_policy_type == "uniform"
    reverse_config = None
    policy: PhyloLearnedReversePolicy | None = None
    reverse_optimizer = None
    if not uniform_reverse:
        reverse_config = PhyloLearnedReverseConfig(
            hidden_size=config.reverse_hidden_size,
            num_layers=config.reverse_num_layers,
        )
        reverse_config.validate()
        policy = PhyloLearnedReversePolicy(
            num_taxa,
            hidden_size=reverse_config.hidden_size,
            num_layers=reverse_config.num_layers,
        ).to(device)
        reverse_optimizer = torch.optim.Adam(policy.parameters(), lr=config.reverse_lr)

    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    rollout_worker = RolloutWorker(env)

    output_dir = Path(
        resolve_output_dir(config)
        if config.resume_from
        else build_output_dir(config.output_root, config.method, config.run_name)
    )
    cfg.OUTPUT_PATH = str(output_dir)
    resume = None
    if config.resume_from:
        resume, checkpoint_path = prepare_resume(
            output_dir,
            checkpoint_name=config.resume_checkpoint,
            training_mode="on_policy",
            steps_per_epoch=config.steps_per_epoch,
            update_cycles=1,
            target_epochs=config.epochs,
            target_resample_rounds=0,
        )
    config.save_json(str(output_dir / "experiment_config.json"))
    (output_dir / "resolved_config.yaml").write_text(cfg.dump(), encoding="utf-8")
    write_catalog(
        output_dir,
        None,
        None,
        outcome_level=config.outcome_level,
        reverse_policy_type=config.reverse_policy_type,
    )

    data_loader = TrainingDataLoader(
        cfg,
        env,
        rollout_worker,
        str(output_dir / "best_trees.pt"),
    )
    outcome_id_cache = OutcomeIdCache(env)
    params = get_generator_params(generator)
    forward_trainer = GRPOTrainer(
        params=params,
        lr=config.grpo_lr,
        max_grad_norm=config.grpo_max_grad_norm,
        advantage_eps=config.grpo_advantage_eps,
        clip_eps=config.grpo_clip_eps,
        clip_eps_high=config.grpo_clip_eps_high,
        entropy_coef=config.grpo_entropy_coef,
        num_iterations=1,
        reward_c=cfg.ENV.REWARD.C,
        reward_scale=cfg.ENV.REWARD.SCALE,
        policy_loss_mode="ppo",
    )
    normalizer = (
        RunningLogWeightNormalizer(
            decay=config.running_scale_decay,
            advantage_clip=config.running_advantage_clip,
            log_ratio_clip=config.running_log_ratio_clip,
        )
        if config.advantage_normalization == "running"
        else None
    )

    if resume is not None:
        load_generator_checkpoint(generator, resume.checkpoint_path)
        reverse_state_path = paired_learned_reverse_state(Path(resume.checkpoint_path))
        load_learned_reverse_state(
            reverse_state_path,
            policy=policy,
            reverse_optimizer=reverse_optimizer,
            forward_trainer=forward_trainer,
            normalizer=normalizer,
            device=device,
        )
        print(f"restored learned-reverse state from {reverse_state_path.name}")

    print(f"run_dir={output_dir}")
    replay_enabled = config.effective_replay_batch_size > 0
    print(
        f"method={config.method} device={device} "
        f"reverse_policy_type={config.reverse_policy_type} "
        f"outcome_level={config.outcome_level} "
        f"reward_target={config.reward_target} "
        f"full_model={not cfg.GFN.MODEL.ONLY_TRAIN_TREE_MODEL} "
        f"G_on={config.on_policy_batch_size} "
        f"G_replay={config.effective_replay_batch_size} "
        f"G_total={config.grpo_group_size} "
        f"replay_buffer={config.replay_buffer_size if replay_enabled else 0}"
    )
    if uniform_reverse:
        print("weight=R(x)*P_B(tau|x)/P_F(tau) with frozen uniform P_B; no reverse MLE update")
    else:
        print("weight=R(x)*q_phi(tau|x)/P_F(tau); reverse update happens after PPO")
    if replay_enabled:
        print(
            "best-tree replay enabled: first "
            f"{data_loader.best_state_batch_size} trees per micro-batch are replayed "
            f"from {len(buffered_best_trees(data_loader))} buffered trees"
        )

    metrics_path = str(output_dir / "metrics.jsonl")
    final_logger = None
    try:
        from final.logging.run_logger import FinalRunLogger, learned_reverse_extra_metrics

        final_logger = FinalRunLogger.maybe_create(
            output_dir, method=config.method, seed=config.seed
        )
    except ImportError:
        pass
    metrics_rows = load_metrics_rows(metrics_path) if resume is not None else []
    epoch_summaries: list[dict[str, float | int | str]] = (
        load_epoch_summaries(output_dir) if resume is not None else []
    )
    global_step = resume.global_step if resume is not None else 0
    start_epoch = resume.start_epoch if resume is not None else 0
    start_step = resume.start_step if resume is not None else 0
    generation_state = None
    tracker = (
        restore_tracker(resume, metrics_rows) if resume is not None else OutcomeTracker()
    )
    cumulative_outcomes: Counter[str] = Counter(tracker.outcome_counts)
    rollout_chunk = resolve_rollout_chunk_size(config)
    log_score_shift = float(getattr(cfg.ENV, "LOG_SCORE_SHIFT", 0.0))

    for epoch in range(start_epoch, cfg.GFN.TRAINING_DATA_LOADER.EPOCHS_NUM):
        exploration = generate_exploration_spec(
            cfg.GFN.TRAINING_DATA_LOADER.EXPLORATION, epoch
        )
        epoch_losses: list[float] = []
        epoch_reverse_losses: list[float] = []
        epoch_ess: list[float] = []

        step_begin = start_step if epoch == start_epoch else 0
        for step in range(step_begin, data_loader.steps_per_epoch):
            random_spec = data_loader.generate_random_spec(exploration, step)
            batch, rollout_trajectories = data_loader.generate_batch(
                generator, random_spec
            )
            action_paths = rollout_tree_action_paths(batch)
            batch_outcome_ids, batch_topology_ids = outcome_id_cache.ids_from_rollout_batch(
                batch,
                rollout_trajectories,
                level=config.outcome_level,
            )
            cumulative_outcomes.update(batch_outcome_ids)
            terminal_log_scores = batch["log_scores"].detach().cpu().tolist()

            trajectory_indices = trajectory_indices_from_paths(
                action_paths, device=device
            )
            forward_log_probabilities = batch["log_paths_pf"].detach().sum(dim=-1)
            if uniform_reverse:
                reverse_log_probabilities = batch["log_paths_pb"].detach().sum(dim=-1)
            else:
                with torch.inference_mode():
                    reverse_log_probabilities = mlp_path_log_probabilities(
                        policy,
                        env,
                        action_paths,
                        terminal_ids=batch_topology_ids,
                        terminal_log_scores=terminal_log_scores,
                    )
            advantages, advantage_metrics = learned_reverse_advantages(
                batch["log_scores"],
                forward_log_probabilities,
                reverse_log_probabilities,
                reward_target=config.reward_target,
                reward_c=float(cfg.ENV.REWARD.C),
                reward_scale=float(cfg.ENV.REWARD.SCALE),
                normalizer=normalizer,
                advantage_eps=config.grpo_advantage_eps,
                terminal_ids=batch_outcome_ids,
                trajectory_indices=trajectory_indices,
            )
            with torch.inference_mode():
                from learned_reverse_ips.advantages import terminal_log_rewards_from_scores

                log_rewards_target = terminal_log_rewards_from_scores(
                    batch["log_scores"],
                    reward_target=config.reward_target,
                    reward_c=float(cfg.ENV.REWARD.C),
                    reward_scale=float(cfg.ENV.REWARD.SCALE),
                )
                log_weights = (
                    log_rewards_target
                    + reverse_log_probabilities.detach()
                    - forward_log_probabilities.detach()
                )
                scaled = torch.exp(log_weights - log_weights.max())
                advantage_metrics["max_normalized_weight"] = float(
                    (scaled / scaled.sum().clamp(min=1e-8)).max().item()
                )
                advantage_metrics["log_importance_weight_std"] = float(
                    log_weights.std(unbiased=False).item()
                )

            forward_metrics, generation_state = run_on_policy_grpo_step(
                forward_trainer,
                rollout_worker,
                generator,
                batch,
                rollout_trajectories,
                random_spec=random_spec,
                generation_state=generation_state,
                chunk_size=rollout_chunk,
                device=device,
                extra_update_kwargs={
                    "fixed_advantages": advantages,
                    "fixed_advantage_metrics": advantage_metrics,
                },
                group_meta={
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "method": config.method,
                },
            )
            reverse_metrics: dict[str, float] = {}
            if not uniform_reverse:
                reverse_batch = build_reverse_batch(
                    env,
                    action_paths,
                    terminal_ids=batch_topology_ids,
                    terminal_log_scores=terminal_log_scores,
                    device=device,
                )
                reverse_metrics = update_mlp_reverse_policy(
                    policy,
                    reverse_optimizer,
                    reverse_batch,
                    train_epochs=config.reverse_train_epochs,
                    grad_clip_norm=config.reverse_grad_clip_norm,
                )
            else:
                reverse_metrics = {"reverse_loss": 0.0}
            replay_tree_count = data_loader.best_state_batch_size
            record = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "method": config.method,
                "training_mode": "on_policy+replay" if replay_enabled else "on_policy",
                "outcome_level": config.outcome_level,
                "reward_target": config.reward_target,
                "on_policy_batch_size": config.on_policy_batch_size,
                "replay_batch_size": config.effective_replay_batch_size,
                "grpo_group_size": config.grpo_group_size,
                "best_trees_buffer_size": len(buffered_best_trees(data_loader)),
                "batch_unique_outcomes": len(set(batch_outcome_ids)),
                "batch_unique_topologies": len(set(batch_topology_ids)),
                "cumulative_unique_outcomes": len(cumulative_outcomes),
                "global_duplicate_fraction": (
                    1.0
                    - len(cumulative_outcomes)
                    / max(sum(cumulative_outcomes.values()), 1)
                ),
                "mean_log_reward": float(batch["log_rewards"].mean().item()),
                "mean_log_score": float(batch["log_scores"].mean().item()),
                **replay_batch_metrics(batch, replay_tree_count),
                **forward_metrics,
                **reverse_metrics,
            }
            if final_logger is not None:
                record.update(
                    learned_reverse_extra_metrics(record, log_w=log_weights)
                )
                final_logger.on_step(
                    record,
                    log_rewards=batch["log_rewards"],
                    log_paths_pf=batch["log_paths_pf"],
                    outcome_ids=batch_outcome_ids,
                    topology_ids=batch_topology_ids,
                    lr=float(forward_trainer.optimizer.param_groups[0]["lr"]),
                )
            append_jsonl(metrics_path, record)
            epoch_losses.append(float(record["loss"]))
            epoch_reverse_losses.append(float(record["reverse_loss"]))
            epoch_ess.append(float(record["ips_ess_fraction"]))

            if config.print_every > 0 and global_step % config.print_every == 0:
                replay_msg = ""
                if replay_enabled and "mean_log_score_on_policy" in record:
                    replay_msg = (
                        f" logL_replay={record['mean_log_score_replay'] - log_score_shift:.0f}"
                        f" logL_on={record['mean_log_score_on_policy'] - log_score_shift:.0f}"
                        f" best_buf={record['best_trees_buffer_size']}"
                    )
                print(
                    f"step={global_step:05d} loss={record['loss']:.4f} "
                    f"reverse_nll={record['reverse_loss']:.4f} "
                    f"ESS={record['ips_ess_fraction']:.3f} "
                    f"batch_outcomes={record['batch_unique_outcomes']} "
                    f"batch_topologies={record['batch_unique_topologies']} "
                    f"seen={record['cumulative_unique_outcomes']}"
                    f"{replay_msg}"
                )
            global_step += 1

        summary = {
            "epoch": epoch,
            "method": config.method,
            "mean_loss": float(np.mean(epoch_losses)),
            "mean_reverse_loss": float(np.mean(epoch_reverse_losses)),
            "mean_ips_ess_fraction": float(np.mean(epoch_ess)),
            "cumulative_unique_outcomes": len(cumulative_outcomes),
        }
        epoch_summaries.append(summary)

        if config.checkpoint_every > 0 and (epoch + 1) % config.checkpoint_every == 0:
            checkpoint_name = f"checkpoint_epoch{epoch:04d}.pt"
            checkpoint_path = output_dir / checkpoint_name
            generator.save(str(checkpoint_path))
            save_learned_reverse_state(
                output_dir / f"learned_reverse_epoch{epoch:04d}.pt",
                policy=policy,
                optimizer=reverse_optimizer,
                forward_trainer=forward_trainer,
                normalizer=normalizer,
                update_step=global_step,
                reverse_config=reverse_config,
                reverse_policy_type=config.reverse_policy_type,
            )
            outcome_tracker = OutcomeTracker()
            outcome_tracker.outcome_counts.update(cumulative_outcomes)
            outcome_tracker.total = sum(cumulative_outcomes.values())
            save_training_state(
                output_dir,
                make_training_state(
                    global_step=global_step,
                    training_mode="on_policy",
                    epoch=epoch,
                    step=step,
                    steps_per_epoch=data_loader.steps_per_epoch,
                    checkpoint_path=str(checkpoint_path),
                    tracker=outcome_tracker,
                ),
                grpo_trainer=forward_trainer,
            )
            save_best_trees(output_dir, data_loader)

    generator.save(str(output_dir / "final_checkpoint.pt"))
    save_best_trees(output_dir, data_loader)
    save_learned_reverse_state(
        output_dir / "learned_reverse_state.pt",
        policy=policy,
        optimizer=reverse_optimizer,
        forward_trainer=forward_trainer,
        normalizer=normalizer,
        update_step=global_step,
        reverse_config=reverse_config,
        reverse_policy_type=config.reverse_policy_type,
    )
    (output_dir / "epoch_summaries.json").write_text(
        json.dumps(epoch_summaries, indent=2), encoding="utf-8"
    )
    outcome_tracker = OutcomeTracker()
    outcome_tracker.outcome_counts.update(cumulative_outcomes)
    outcome_tracker.total = sum(cumulative_outcomes.values())
    save_training_state(
        output_dir,
        make_training_state(
            global_step=global_step,
            training_mode="on_policy",
            epoch=cfg.GFN.TRAINING_DATA_LOADER.EPOCHS_NUM - 1,
            step=data_loader.steps_per_epoch - 1,
            steps_per_epoch=data_loader.steps_per_epoch,
            checkpoint_path=str(output_dir / "final_checkpoint.pt"),
            tracker=outcome_tracker,
        ),
        grpo_trainer=forward_trainer,
    )
    print(f"completed: {output_dir}")

    if final_logger is not None:
        final_logger.finalize()

    if config.post_train:
        from learned_reverse_ips.post_train import run_post_train_pipeline

        sample_batch_size = (
            config.post_train_sample_batch_size or config.on_policy_batch_size
        )
        run_post_train_pipeline(
            output_dir,
            num_trees=config.post_train_sample_size,
            sample_batch_size=sample_batch_size,
            seed=config.seed,
            device=device,
        )

    return str(output_dir)
