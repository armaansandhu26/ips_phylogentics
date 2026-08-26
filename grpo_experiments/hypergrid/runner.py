"""Training loop for Hyper-Grid GRPO and count IPS."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from final.toy.eval_sampling import evaluate_terminal_distribution
from final.toy.hypergrid_env import HyperGridDataset
from final.toy.hypergrid_policy import HyperGridPolicy
from final.toy.hypergrid_rollout import rollout_batch, sample_terminals
from final.toy.hypergrid_reverse_policy import (
    HyperGridReversePolicy,
    build_reverse_batch,
    forward_action_paths_from_batch,
    path_log_probabilities,
    update_reverse_policy,
)
from final.toy.hypergrid_tb import HyperGridTBTrainer
from learned_reverse_ips.advantages import RunningLogWeightNormalizer, learned_reverse_advantages
from final.logging.wandb_logger import FinalWandbLogger, WandbSettings
from grpo_experiments.core.trainer import GRPOTrainer
from grpo_experiments.hypergrid.config import HypergridExperimentConfig, save_resolved_config
from grpo_experiments.ips_grpo.trainer import IPSGRPOTrainer
from grpo_experiments.metrics import batch_diversity_stats
from grpo_experiments.utils import append_jsonl, set_seed


def _build_output_dir(cfg: HypergridExperimentConfig) -> Path:
    if cfg.resume_from is not None:
        return cfg.resume_from
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{stamp}_{cfg.method}_hypergrid"
    if cfg.run_name:
        label = f"{stamp}_{cfg.run_name}_{cfg.method}"
    out = cfg.output / label
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_trainer(cfg: HypergridExperimentConfig, params) -> GRPOTrainer | HyperGridTBTrainer:
    if cfg.method == "trajectory_balance":
        return HyperGridTBTrainer(
            params,
            lr=cfg.lr,
            max_grad_norm=cfg.max_grad_norm,
        )
    common = dict(
        params=params,
        lr=cfg.lr,
        clip_eps=cfg.clip_eps,
        max_grad_norm=cfg.max_grad_norm,
        reward_c=0.0,
        reward_scale=1.0,
        entropy_coef=cfg.entropy_coef,
        num_iterations=cfg.num_iterations,
        advantage_reward_mode="exp_linear",
    )
    if cfg.method == "count_ips":
        return IPSGRPOTrainer(
            **common,
            ips_prob_floor=cfg.ips_prob_floor,
            ips_propensity_mode="count",
        )
    if cfg.method == "learned_reverse_ips":
        return GRPOTrainer(**common)
    return GRPOTrainer(**common)


def _save_checkpoint(
    path: Path,
    *,
    policy: HyperGridPolicy,
    trainer: GRPOTrainer | HyperGridTBTrainer,
    global_step: int,
    epoch: int,
    reverse_policy: HyperGridReversePolicy | None = None,
    reverse_optimizer: torch.optim.Optimizer | None = None,
    normalizer: RunningLogWeightNormalizer | None = None,
) -> None:
    payload = {
        "policy": policy.state_dict(),
        "trainer": trainer.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
    }
    if reverse_policy is not None:
        payload["reverse_policy"] = reverse_policy.state_dict()
    if reverse_optimizer is not None:
        payload["reverse_optimizer"] = reverse_optimizer.state_dict()
    if normalizer is not None:
        payload["normalizer"] = normalizer.state_dict()
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_checkpoint(
    path: Path,
    policy: HyperGridPolicy,
    trainer: GRPOTrainer | HyperGridTBTrainer,
    *,
    reverse_policy: HyperGridReversePolicy | None = None,
    reverse_optimizer: torch.optim.Optimizer | None = None,
    normalizer: RunningLogWeightNormalizer | None = None,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["policy"])
    if "trainer" in payload:
        trainer.load_state_dict(payload["trainer"])
    if reverse_policy is not None and "reverse_policy" in payload:
        reverse_policy.load_state_dict(payload["reverse_policy"])
    if reverse_optimizer is not None and "reverse_optimizer" in payload:
        reverse_optimizer.load_state_dict(payload["reverse_optimizer"])
    if normalizer is not None and "normalizer" in payload:
        loaded = payload["normalizer"]
        normalizer.log_first_moment = loaded.get("log_first_moment")
        normalizer.log_second_moment = loaded.get("log_second_moment")
        normalizer.updates = int(loaded.get("updates", 0))
    return payload


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = list(output_dir.glob("checkpoint_epoch*.pt"))
    final = output_dir / "final_checkpoint.pt"
    if final.exists():
        candidates.append(final)
    if not candidates:
        return None

    def _global_step(path: Path) -> int:
        meta = torch.load(path, map_location="cpu", weights_only=False)
        return int(meta.get("global_step", -1))

    return max(candidates, key=_global_step)


def _run_eval(
    policy: HyperGridPolicy,
    dataset: HyperGridDataset,
    *,
    num_samples: int,
    batch_size: int,
    device: str,
) -> dict:
    coords, _ = sample_terminals(
        policy,
        dataset,
        num_samples=num_samples,
        batch_size=min(batch_size, num_samples),
        device=device,
    )
    return evaluate_terminal_distribution(coords.numpy(), dataset)


def _wandb_log_train(wandb_logger: FinalWandbLogger | None, step: int, record: dict) -> None:
    if wandb_logger is None:
        return
    prefixed = {f"train/{k}": v for k, v in record.items() if k != "global_step"}
    wandb_logger.log_metrics(step, prefixed)


def _wandb_log_eval(wandb_logger: FinalWandbLogger | None, step: int, metrics: dict) -> None:
    if wandb_logger is None:
        return
    prefixed = {f"eval/{k}": v for k, v in metrics.items()}
    wandb_logger.log_metrics(step, prefixed)


def run_experiment(
    cfg: HypergridExperimentConfig,
    *,
    wandb_settings: WandbSettings | None = None,
) -> Path:
    set_seed(cfg.seed)
    dataset = HyperGridDataset.load(cfg.dataset)
    spec = dataset.spec
    output_dir = _build_output_dir(cfg)
    save_resolved_config(output_dir, cfg)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    wandb_settings = wandb_settings or WandbSettings()
    if wandb_settings.enabled and not wandb_settings.run_name:
        wandb_settings = WandbSettings(
            enabled=wandb_settings.enabled,
            project=wandb_settings.project,
            entity=wandb_settings.entity,
            run_name=f"{cfg.run_name}_{cfg.method}",
            group=wandb_settings.group or cfg.run_name,
            tags=wandb_settings.tags,
        )
    wandb_logger = FinalWandbLogger.configure(wandb_settings)
    plot_watcher = None
    if wandb_logger is not None:
        wandb_settings.apply_to_env()
        wandb_logger.init(
            {
                "method": cfg.method,
                "suite": cfg.run_name,
                "dataset": str(cfg.dataset),
                "H": spec.H,
                "D": spec.D,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
            }
        )
        plot_watcher = wandb_logger.watch_plot_dirs([plots_dir])
        plot_watcher.__enter__()

    device = cfg.device
    policy = HyperGridPolicy(
        dim=spec.D,
        num_actions=dataset.num_actions,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        H=spec.H,
    ).to(device)
    trainer = _build_trainer(cfg, list(policy.parameters()))
    reverse_policy = None
    reverse_optimizer = None
    normalizer = None
    if cfg.method == "learned_reverse_ips":
        reverse_policy = HyperGridReversePolicy(
            hidden_size=cfg.reverse_hidden_size,
            num_layers=cfg.reverse_num_layers,
            H=spec.H,
        ).to(device)
        reverse_optimizer = torch.optim.Adam(reverse_policy.parameters(), lr=cfg.reverse_lr)
        normalizer = (
            RunningLogWeightNormalizer()
            if cfg.advantage_normalization == "running"
            else None
        )
    metrics_path = output_dir / "metrics.jsonl"
    epoch_summaries: list[dict] = []
    summaries_path = output_dir / "epoch_summaries.json"
    if summaries_path.exists():
        epoch_summaries = json.loads(summaries_path.read_text(encoding="utf-8"))

    start_epoch = 0
    global_step = 0
    seen_outcomes: set[str] = set()
    if cfg.resume_from is not None:
        ckpt = _find_latest_checkpoint(output_dir)
        if ckpt is not None:
            meta = _load_checkpoint(
                ckpt,
                policy,
                trainer,
                reverse_policy=reverse_policy,
                reverse_optimizer=reverse_optimizer,
                normalizer=normalizer,
            )
            global_step = int(meta.get("global_step", 0))
            start_epoch = int(meta.get("epoch", 0)) + 1
            print(
                f"resume: {output_dir.name} from {ckpt.name} epoch={start_epoch} "
                f"global_step={global_step} target_epochs={cfg.epochs}"
            )
        if metrics_path.exists():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                outcome_id = row.get("outcome_id")
                if outcome_id:
                    seen_outcomes.add(str(outcome_id))
            if not seen_outcomes:
                last_cum = 0.0
                for line in metrics_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    last_cum = float(json.loads(line).get("cumulative_unique_outcomes", last_cum))
                if last_cum > 0:
                    print(f"resume: restored cumulative_unique_outcomes={last_cum:.0f} (set not persisted)")

    dataset.load_target_probs()

    try:
        for epoch in range(start_epoch, cfg.epochs):
            epoch_losses: list[float] = []
            epoch_l1: list[float] = []
            epoch_modes: list[float] = []

            for step in range(cfg.steps_per_epoch):
                batch = rollout_batch(policy, dataset, batch_size=cfg.batch_size, device=device)
                outcome_ids = batch.outcome_ids
                topology_ids = outcome_ids
                div = batch_diversity_stats(outcome_ids, topology_ids)
                seen_outcomes.update(outcome_ids)

                if cfg.method == "count_ips":
                    train_info = trainer.update(
                        batch.log_paths_pf,
                        batch.log_rewards,
                        log_scores=batch.log_scores,
                        outcome_ids=outcome_ids,
                        mask=batch.mask,
                    )
                elif cfg.method == "trajectory_balance":
                    action_paths = forward_action_paths_from_batch(
                        batch.actions,
                        batch.mask,
                        terminate_action=dataset.terminate_action,
                    )
                    train_info = trainer.update(
                        batch.log_paths_pf,
                        batch.log_rewards,
                        action_paths=action_paths,
                        terminal_coords=batch.terminal_coords,
                        dim=spec.D,
                    )
                elif cfg.method == "learned_reverse_ips":
                    action_paths = forward_action_paths_from_batch(
                        batch.actions,
                        batch.mask,
                        terminate_action=dataset.terminate_action,
                    )
                    forward_log_probabilities = batch.log_paths_pf.sum(dim=-1)
                    reverse_log_probabilities = path_log_probabilities(
                        reverse_policy,
                        action_paths,
                        terminal_coords=batch.terminal_coords,
                        terminal_log_rewards=batch.log_rewards,
                        H=spec.H,
                    )
                    trajectory_key: dict[tuple[int, ...], int] = {}
                    trajectory_indices = []
                    for path in action_paths:
                        key = tuple(path)
                        if key not in trajectory_key:
                            trajectory_key[key] = len(trajectory_key)
                        trajectory_indices.append(trajectory_key[key])
                    trajectory_indices_t = torch.tensor(
                        trajectory_indices, dtype=torch.long, device=device
                    )
                    advantages, advantage_metrics = learned_reverse_advantages(
                        batch.log_rewards,
                        forward_log_probabilities,
                        reverse_log_probabilities,
                        reward_target=cfg.reward_target,
                        reward_c=0.0,
                        reward_scale=1.0,
                        normalizer=normalizer,
                        advantage_eps=1e-8,
                        terminal_ids=outcome_ids,
                        trajectory_indices=trajectory_indices_t,
                    )
                    train_info = trainer.update(
                        batch.log_paths_pf,
                        batch.log_rewards,
                        fixed_advantages=advantages,
                        mask=batch.mask,
                        extra_metrics=advantage_metrics,
                    )
                    reverse_batch = build_reverse_batch(
                        action_paths,
                        terminal_coords=batch.terminal_coords,
                        terminal_log_rewards=batch.log_rewards,
                        H=spec.H,
                        device=device,
                    )
                    reverse_metrics = update_reverse_policy(
                        reverse_policy,
                        reverse_optimizer,
                        reverse_batch,
                        train_epochs=cfg.reverse_train_epochs,
                        grad_clip_norm=cfg.reverse_grad_clip_norm,
                    )
                    train_info.update(reverse_metrics)
                else:
                    train_info = trainer.update(
                        batch.log_paths_pf,
                        batch.log_rewards,
                        log_scores=batch.log_scores,
                        mask=batch.mask,
                    )

                record = {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "method": cfg.method,
                    "mean_log_reward": float(batch.log_rewards.mean().item()),
                    "mean_reward": float(torch.exp(batch.log_rewards).mean().item()),
                    **train_info,
                    **div,
                    "cumulative_unique_outcomes": float(len(seen_outcomes)),
                }
                append_jsonl(str(metrics_path), record)
                _wandb_log_train(wandb_logger, global_step, record)

                if cfg.print_every > 0 and global_step % cfg.print_every == 0:
                    msg = (
                        f"step={global_step:05d} loss={train_info['loss']:.4f} "
                        f"R={record['mean_reward']:.3f} adv={train_info['mean_advantage']:.3f} "
                        f"unique={div['batch_unique_outcomes']:.0f}"
                    )
                    if cfg.method == "count_ips":
                        msg += f" p_hat={train_info.get('ips_prob_mean', 0):.4f}"
                    if cfg.method == "learned_reverse_ips":
                        msg += (
                            f" ESS={train_info.get('ips_ess_fraction', 0):.3f}"
                            f" rev={train_info.get('reverse_loss', 0):.4f}"
                        )
                    if cfg.method == "trajectory_balance":
                        msg += f" logZ={train_info.get('log_Z', 0):.3f}"
                    print(msg)

                epoch_losses.append(train_info["loss"])
                global_step += 1

            eval_metrics: dict = {}
            if cfg.eval_every > 0 and ((epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs):
                raw_eval = _run_eval(
                    policy,
                    dataset,
                    num_samples=cfg.eval_samples,
                    batch_size=cfg.batch_size,
                    device=device,
                )
                eval_metrics = {
                    k: v for k, v in raw_eval.items() if k not in {"sampled_coords", "sampled_rewards"}
                }
                sampled_coords = raw_eval.get("sampled_coords")
                if sampled_coords is not None:
                    np.savez_compressed(
                        output_dir / f"eval_samples_epoch{epoch:04d}.npz",
                        coords=np.asarray(sampled_coords),
                        rewards=np.asarray(raw_eval.get("sampled_rewards", [])),
                    )
                epoch_l1.append(eval_metrics["l1_distance"])
                epoch_modes.append(eval_metrics["num_modes_with_mass"])
                print(
                    f"--- epoch={epoch:03d} eval L1={eval_metrics['l1_distance']:.4f} "
                    f"TV={eval_metrics['total_variation']:.4f} "
                    f"modes={eval_metrics['num_modes_with_mass']:.0f}/4 "
                    f"peak_mass={eval_metrics['peak_mode_mass']:.4f}"
                )
                _wandb_log_eval(wandb_logger, global_step, eval_metrics)

            summary = {
                "epoch": epoch,
                "method": cfg.method,
                "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else math.nan,
                "cumulative_unique_outcomes": float(len(seen_outcomes)),
                **eval_metrics,
            }
            epoch_summaries.append(summary)
            (output_dir / "epoch_summaries.json").write_text(
                json.dumps(epoch_summaries, indent=2) + "\n",
                encoding="utf-8",
            )
            if eval_metrics:
                _write_live_l1_plot(epoch_summaries, plots_dir / "training_l1_live.png")
                _save_checkpoint(
                    output_dir / f"checkpoint_epoch{epoch:04d}.pt",
                    policy=policy,
                    trainer=trainer,
                    global_step=global_step,
                    epoch=epoch,
                    reverse_policy=reverse_policy,
                    reverse_optimizer=reverse_optimizer,
                    normalizer=normalizer,
                )

            if cfg.checkpoint_every > 0 and (epoch + 1) % cfg.checkpoint_every == 0:
                _save_checkpoint(
                    output_dir / f"checkpoint_epoch{epoch:04d}.pt",
                    policy=policy,
                    trainer=trainer,
                    global_step=global_step,
                    epoch=epoch,
                    reverse_policy=reverse_policy,
                    reverse_optimizer=reverse_optimizer,
                    normalizer=normalizer,
                )

        _save_checkpoint(
            output_dir / "final_checkpoint.pt",
            policy=policy,
            trainer=trainer,
            global_step=global_step,
            epoch=cfg.epochs - 1,
            reverse_policy=reverse_policy,
            reverse_optimizer=reverse_optimizer,
            normalizer=normalizer,
        )
        final_eval = _run_eval(
            policy,
            dataset,
            num_samples=cfg.eval_samples,
            batch_size=cfg.batch_size,
            device=device,
        )
        sampled_coords = final_eval.pop("sampled_coords")
        sampled_rewards = final_eval.pop("sampled_rewards")
        final_eval["target_probs_path"] = str(dataset.root / "target_distribution.npz")
        (output_dir / "eval_metrics.json").write_text(
            json.dumps(final_eval, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "epoch_summaries.json").write_text(
            json.dumps(epoch_summaries, indent=2) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            output_dir / f"sampled_terminals_{cfg.eval_samples}.npz",
            coords=sampled_coords,
            rewards=sampled_rewards,
        )
        _wandb_log_eval(wandb_logger, global_step, final_eval)

        from final.toy.plot_comparison import plot_training_diagnostics

        plot_training_diagnostics(
            output_dir,
            method_label=cfg.method,
            out_path=plots_dir / "training_diagnostics.png",
        )
        if plot_watcher is not None:
            plot_watcher.scan_once()
        if wandb_logger is not None:
            wandb_logger.finish()
    finally:
        if plot_watcher is not None:
            plot_watcher.__exit__(None, None, None)

    return output_dir


def _write_live_l1_plot(epoch_summaries: list[dict], out_path: Path) -> None:
    """Write a partial L1 curve during training for wandb plot watcher."""
    epochs = [int(row["epoch"]) for row in epoch_summaries if "l1_distance" in row]
    l1_vals = [float(row["l1_distance"]) for row in epoch_summaries if "l1_distance" in row]
    if len(epochs) < 1:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    ax.plot(epochs, l1_vals, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 distance to target")
    ax.set_ylim(0.0, max(2.2, max(l1_vals) * 1.05))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
