"""Frozen-dataset learned-reverse IPS for the direction/step DAG.

This is the offline counterpart of ``learned_reverse_ips.py``:

1. collect or load a fixed behavior dataset with exact ``log mu(tau)``;
2. fit a terminal-conditioned reverse policy to that fixed dataset;
3. freeze the reverse policy and compute one fixed weight per trajectory,

       w_reverse(tau) = R(x) * q_phi(tau | x) / mu(tau);

4. train a fresh forward policy only by weighted trajectory likelihood.

No forward-policy training rollout is added to the dataset, and neither the
reverse policy nor the IPS weights change during forward training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from exact_probability_ips import _resolve_device
from learned_reverse_ips import (  # noqa: E402
    LearnedReverseConfig,
    LearnedReversePolicy,
    reverse_action_index,
    reverse_action_mask,
    reverse_context,
)
from offline_count_ips import (  # noqa: E402
    FrozenTrajectoryDataset,
    OfflineTrajectory,
    OfflineWeightedIPSTrainer,
    add_common_arguments,
    config_from_args,
    finish_run,
    load_or_collect_dataset,
    normalize_fixed_weights,
    train_offline_policy,
    weighted_terminal_metrics,
)


@dataclass(frozen=True)
class ReverseOfflineBatch:
    contexts: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    episode_indices: torch.Tensor
    num_episodes: int


def materialize_reverse_batch(
    trajectories: Sequence[OfflineTrajectory],
    *,
    budget: int,
    max_step: int,
    device: torch.device,
) -> ReverseOfflineBatch:
    if not trajectories:
        raise ValueError("reverse offline minibatch must be non-empty")
    contexts: list[tuple[float, ...]] = []
    masks: list[tuple[bool, ...]] = []
    actions: list[int] = []
    episode_indices: list[int] = []
    for episode_index, item in enumerate(trajectories):
        child_x = 0
        child_y = 0
        for direction, length in item.trajectory:
            if direction == 0:
                child_x += length
            else:
                child_y += length
            child = type(item.terminal)(child_x, child_y)
            contexts.append(
                reverse_context(child, item.terminal, budget=budget)
            )
            masks.append(reverse_action_mask(child, max_step=max_step))
            actions.append(
                reverse_action_index(
                    direction, length, max_step=max_step
                )
            )
            episode_indices.append(episode_index)
        if child_x != item.terminal.x or child_y != item.terminal.y:
            raise ValueError("reverse trajectory does not reach its terminal")
    return ReverseOfflineBatch(
        contexts=torch.tensor(
            contexts, dtype=torch.float32, device=device
        ),
        masks=torch.tensor(masks, dtype=torch.bool, device=device),
        actions=torch.tensor(actions, dtype=torch.long, device=device),
        episode_indices=torch.tensor(
            episode_indices, dtype=torch.long, device=device
        ),
        num_episodes=len(trajectories),
    )


def reverse_path_log_probabilities(
    policy: LearnedReversePolicy,
    batch: ReverseOfflineBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distribution = policy.dist(batch.contexts, batch.masks)
    edge_log_probabilities = distribution.log_prob(batch.actions)
    path_log_probabilities = torch.zeros(
        batch.num_episodes,
        dtype=torch.float32,
        device=batch.contexts.device,
    )
    path_log_probabilities.scatter_add_(
        0, batch.episode_indices, edge_log_probabilities
    )
    return (
        path_log_probabilities,
        distribution.entropy(),
        distribution.logits.argmax(dim=-1),
    )


def fit_reverse_policy(
    policy: LearnedReversePolicy,
    dataset: FrozenTrajectoryDataset,
    reverse_config: LearnedReverseConfig,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    log_every: int,
) -> list[dict[str, float]]:
    if updates < 1 or batch_size < 1:
        raise ValueError("reverse updates and batch size must be >= 1")
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=reverse_config.lr
    )
    rng = np.random.default_rng(seed)
    parameters = list(policy.parameters())
    history: list[dict[str, float]] = []
    policy.train()
    for update_step in range(1, updates + 1):
        indices = rng.integers(0, len(dataset), size=batch_size)
        trajectories = [
            dataset.trajectories[int(index)] for index in indices
        ]
        batch = materialize_reverse_batch(
            trajectories,
            budget=dataset.budget,
            max_step=dataset.max_step,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        path_log_probabilities, edge_entropy, predictions = (
            reverse_path_log_probabilities(policy, batch)
        )
        loss = -path_log_probabilities.mean()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            parameters, reverse_config.grad_clip_norm
        )
        optimizer.step()
        with torch.no_grad():
            row = {
                "step": float(update_step),
                "reverse_loss": float(loss.item()),
                "reverse_edge_entropy": float(edge_entropy.mean().item()),
                "reverse_edge_accuracy": float(
                    (predictions == batch.actions).float().mean().item()
                ),
                "reverse_grad_norm": float(grad_norm.item()),
                "reverse_param_norm": float(
                    sum(
                        parameter.detach().norm().item() ** 2
                        for parameter in parameters
                    )
                    ** 0.5
                ),
            }
        history.append(row)
        if update_step == 1 or update_step % log_every == 0:
            print(
                f"reverse update {update_step:5d}  "
                f"NLL={row['reverse_loss']:.3f}  "
                f"accuracy={row['reverse_edge_accuracy']:.3f}  "
                f"entropy={row['reverse_edge_entropy']:.3f}"
            )
    policy.eval()
    return history


@torch.inference_mode()
def score_reverse_dataset(
    policy: LearnedReversePolicy,
    dataset: FrozenTrajectoryDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(dataset), batch_size):
        trajectories = dataset.trajectories[start : start + batch_size]
        batch = materialize_reverse_batch(
            trajectories,
            budget=dataset.budget,
            max_step=dataset.max_step,
            device=device,
        )
        path_log_probabilities, _, _ = reverse_path_log_probabilities(
            policy, batch
        )
        values.append(
            path_log_probabilities.cpu().numpy().astype(np.float64)
        )
    return np.concatenate(values)


def learned_reverse_offline_weights(
    dataset: FrozenTrajectoryDataset,
    reverse_log_probabilities: np.ndarray,
    *,
    clip: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if reverse_log_probabilities.shape != (len(dataset),):
        raise ValueError("one reverse log-probability is required per trajectory")
    log_behavior = np.asarray(
        [
            item.behavior_log_probability
            for item in dataset.trajectories
        ],
        dtype=np.float64,
    )
    rewards = np.asarray(
        [item.reward for item in dataset.trajectories],
        dtype=np.float64,
    )
    log_weights = (
        np.log(rewards) + reverse_log_probabilities - log_behavior
    )
    shifted_weights = np.exp(log_weights - float(log_weights.max()))
    implied_terminal_log_probability = (
        log_behavior - reverse_log_probabilities
    )
    within_terminal_std: list[float] = []
    for terminal in dataset.terminal_counts:
        mask = np.asarray(
            [
                item.terminal == terminal
                for item in dataset.trajectories
            ],
            dtype=bool,
        )
        if int(mask.sum()) > 1:
            within_terminal_std.append(
                float(np.std(implied_terminal_log_probability[mask]))
            )
    weights, metrics = normalize_fixed_weights(
        shifted_weights,
        clip=clip,
        extra_metrics={
            "estimator": "learned_reverse",
            "raw_weight_representation": (
                "exp(log_weight - max(log_weight)); common scale is irrelevant"
            ),
            "log_weight_mean": float(log_weights.mean()),
            "log_weight_min": float(log_weights.min()),
            "log_weight_max": float(log_weights.max()),
            "behavior_log_probability_mean": float(log_behavior.mean()),
            "reverse_log_probability_mean": float(
                reverse_log_probabilities.mean()
            ),
            "implied_terminal_log_probability_mean": float(
                implied_terminal_log_probability.mean()
            ),
            "implied_terminal_log_probability_std": float(
                implied_terminal_log_probability.std()
            ),
            "implied_terminal_within_outcome_std": float(
                np.mean(within_terminal_std)
                if within_terminal_std
                else 0.0
            ),
        },
    )
    metrics.update(weighted_terminal_metrics(dataset, weights))
    return weights, metrics


def plot_reverse_fit(
    history: list[dict[str, float]], *, output: Path
) -> None:
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(steps, [row["reverse_loss"] for row in history])
    axes[0].set_title("Reverse trajectory NLL")
    axes[1].plot(
        steps, [row["reverse_edge_accuracy"] for row in history]
    )
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Reverse edge accuracy")
    axes[2].plot(
        steps,
        [row["reverse_edge_entropy"] for row in history],
        label="entropy",
    )
    axes[2].plot(
        steps,
        [row["reverse_grad_norm"] for row in history],
        label="gradient norm",
    )
    axes[2].set_title("Reverse optimization")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("Reverse offline update")
        axis.grid(alpha=0.22)
    fig.suptitle("Offline learned-reverse fit")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--reverse-lr", type=float, default=1e-3)
    parser.add_argument("--reverse-hidden-size", type=int, default=128)
    parser.add_argument("--reverse-num-layers", type=int, default=2)
    parser.add_argument("--reverse-updates", type=int, default=2_000)
    parser.add_argument("--reverse-batch-size", type=int, default=128)
    parser.add_argument("--reverse-grad-clip-norm", type=float, default=1.0)
    args = parser.parse_args()
    config = config_from_args(args)
    reverse_config = LearnedReverseConfig(
        hidden_size=args.reverse_hidden_size,
        num_layers=args.reverse_num_layers,
        lr=args.reverse_lr,
        train_epochs=1,
        grad_clip_norm=args.reverse_grad_clip_norm,
    )
    reverse_config.validate()
    device_name = _resolve_device(args.device)
    device = torch.device(device_name)
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "offline_learned_reverse_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_n{args.dataset_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset, dataset_path = load_or_collect_dataset(
        args, config, device=device_name, run_dir=run_dir
    )
    dataset_summary = dataset.summary()
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(
        "Frozen dataset: "
        f"{len(dataset):,} trajectories, "
        f"{dataset_summary['unique_terminals']}/{config.budget + 1} terminals"
    )
    if dataset_summary["unique_terminals"] < config.budget + 1:
        print(
            "WARNING: the frozen dataset has missing terminal support; "
            "offline IPS cannot recover unseen terminals."
        )

    reverse_policy = LearnedReversePolicy(
        config.max_step,
        reverse_config.hidden_size,
        reverse_config.num_layers,
    ).to(device)
    reverse_history = fit_reverse_policy(
        reverse_policy,
        dataset,
        reverse_config,
        updates=args.reverse_updates,
        batch_size=args.reverse_batch_size,
        seed=config.seed,
        device=device,
        log_every=config.log_every,
    )
    reverse_policy.eval()
    torch.save(
        {
            "reverse_config": asdict(reverse_config),
            "reverse_policy": reverse_policy.state_dict(),
            "updates": args.reverse_updates,
            "algorithm": "offline_learned_reverse_density",
        },
        run_dir / "reverse_checkpoint.pt",
    )
    (run_dir / "reverse_history.json").write_text(
        json.dumps(reverse_history, indent=2), encoding="utf-8"
    )
    reverse_plot = run_dir / "reverse_training.png"
    plot_reverse_fit(reverse_history, output=reverse_plot)
    reverse_log_probabilities = score_reverse_dataset(
        reverse_policy,
        dataset,
        batch_size=args.collection_batch_size,
        device=device,
    )
    clip = args.weight_clip if args.weight_clip > 0.0 else None
    weights, weight_metrics = learned_reverse_offline_weights(
        dataset,
        reverse_log_probabilities,
        clip=clip,
    )
    algorithm = {
        "name": "offline_learned_reverse_ips",
        "display_name": "Offline learned-reverse IPS",
        "raw_weight": "R(x) * q_phi(tau|x) / mu(tau)",
        "forward_loss": "fixed_IPS_weighted_full_trajectory_NLL",
        "reverse_training": "offline_then_frozen",
        "training_rollouts_after_collection": 0,
    }
    trainer = OfflineWeightedIPSTrainer(config, device=device_name)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "reverse_config": asdict(reverse_config),
                "reverse_updates": args.reverse_updates,
                "reverse_batch_size": args.reverse_batch_size,
                "device": str(device),
                "algorithm": algorithm,
                "dataset_path": str(dataset_path),
                "dataset_collection": {
                    "requested_size": args.dataset_size,
                    "collection_batch_size": args.collection_batch_size,
                    "behavior_checkpoint": (
                        str(args.behavior_checkpoint)
                        if args.behavior_checkpoint is not None
                        else None
                    ),
                    "behavior_seed": args.behavior_seed,
                },
                "weight_clip": clip,
                "checkpoint_every": args.checkpoint_every,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history = train_offline_policy(
        trainer,
        dataset,
        weights,
        num_updates=config.num_updates,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        checkpoint_every=args.checkpoint_every or None,
        checkpoint_dir=run_dir / "checkpoints",
        algorithm=algorithm,
    )
    finish_run(
        run_dir=run_dir,
        trainer=trainer,
        dataset=dataset,
        dataset_path=dataset_path,
        history=history,
        weight_metrics=weight_metrics,
        algorithm=algorithm,
        final_samples=args.final_samples,
        extra_summary={
            "reverse_config": asdict(reverse_config),
            "reverse_updates": args.reverse_updates,
            "final_reverse_diagnostics": reverse_history[-1],
            "reverse_checkpoint": "reverse_checkpoint.pt",
            "reverse_training_plot": reverse_plot.name,
        },
    )


if __name__ == "__main__":
    main()
