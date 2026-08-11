"""Train and evaluate the minimal trajectory-balance GFlowNet DAG baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from config import TrainConfig
from gflownet import TrajectoryBalanceGFlowNet
from run_count_ips import _plot_final_counts, _plot_trajectory_diagnostics


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return requested


def _plot_training_curves(
    history: list[dict],
    trainer: TrajectoryBalanceGFlowNet,
    *,
    output: Path,
) -> None:
    steps = [row["step"] for row in history]
    target = trainer.target_reward()
    target_mean_reward = sum(
        target[state] * trainer.reward_by_terminal[state]
        for state in trainer.terminals
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(steps, [row["loss"] for row in history], label="TB loss")
    ax.plot(
        steps,
        [row["tb_residual_abs_mean"] for row in history],
        label="mean |residual|",
    )
    ax.set_yscale("log")
    ax.set_title("Trajectory-balance fit")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, [row["log_z"] for row in history], label="learned log Z")
    ax.axhline(
        float(torch.log(torch.tensor(sum(trainer.reward_by_terminal.values())))),
        linestyle="--",
        color="#00b894",
        label="exact log sum R (diagnostic)",
    )
    ax.set_title("Partition function")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(steps, [row["mean_reward"] for row in history], label="sampled")
    ax.axhline(
        target_mean_reward,
        linestyle="--",
        color="#00b894",
        label="ideal sampler",
    )
    ax.set_title("Mean terminal reward")
    ax.legend()

    ax = axes[1, 1]
    if eval_rows:
        eval_steps = [row["step"] for row in eval_rows]
        ax.plot(
            eval_steps,
            [row["tv_reward_target"] for row in eval_rows],
            "o-",
            label="TV distance",
        )
        ax.plot(
            eval_steps,
            [row["max_abs_prob_error"] for row in eval_rows],
            "s--",
            label="max probability error",
        )
    ax.set_ylim(bottom=0)
    ax.set_title("Distance from reward target")
    ax.legend()

    for ax in axes.flat:
        ax.set_xlabel("update")
        ax.grid(alpha=0.22)
    fig.suptitle("Trajectory-balance GFlowNet training")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--z-lr", type=float, default=1e-2)
    parser.add_argument("--initial-log-z", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--terminal-rewards",
        type=float,
        nargs="+",
        default=None,
        metavar="R",
        help="budget + 1 rewards in increasing terminal x order",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        budget=args.budget,
        max_step=args.max_step,
        terminal_rewards=(
            tuple(args.terminal_rewards) if args.terminal_rewards is not None else None
        ),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        group_size=args.group_size,
        num_groups=args.num_groups,
        num_updates=args.num_updates,
        lr=args.lr,
        entropy_coef=0.0,
        seed=args.seed,
        log_every=args.log_every,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "gflownet_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = TrajectoryBalanceGFlowNet(
        config,
        device=_resolve_device(args.device),
        z_lr=args.z_lr,
        initial_log_z=args.initial_log_z,
    )
    checkpoint_every = args.checkpoint_every or None
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print("Algorithm: trajectory-balance GFlowNet with uniform backward policy")

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "device": str(trainer.device),
                "algorithm": "trajectory_balance_gflownet",
                "objective": "(logZ + logPF - logR - logPB)^2",
                "backward_policy": "uniform_over_valid_parents",
                "z_lr": trainer.z_lr,
                "initial_log_z": args.initial_log_z,
                "checkpoint_every": checkpoint_every,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history = trainer.train(
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=run_dir / "checkpoints",
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    trainer.save(run_dir / "checkpoint.pt", update_step=config.num_updates)

    training_plot = run_dir / "training_curves.png"
    _plot_training_curves(history, trainer, output=training_plot)
    final_evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        final_evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Trajectory-balance GFlowNet vs ideal reward sampling",
    )
    trajectory_plot = run_dir / "trajectory_sampling.png"
    trajectory_sampling = _plot_trajectory_diagnostics(
        history,
        trainer,
        final_evaluation,
        samples=args.final_samples,
        output=trajectory_plot,
        subtitle="Target conditional paths follow the fixed reverse policy",
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary = {
        "environment": trainer.environment_summary(),
        "algorithm": {
            "name": "trajectory_balance_gflownet",
            "objective": "(logZ + logPF - logR - logPB)^2",
            "backward_policy": "uniform_over_valid_parents",
            "learned_log_z": float(trainer.log_z.detach().item()),
        },
        "final_sampling": sampling,
        "trajectory_sampling": trajectory_sampling,
        "best_eval": (
            min(eval_rows, key=lambda row: row["tv_reward_target"])
            if eval_rows
            else None
        ),
        "plots": {
            "training_curves": training_plot.name,
            "sampling_counts": "sampling_counts.png",
            "trajectory_sampling": trajectory_plot.name,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Final counts: {sampling['actual_counts']}")
    print(f"Ideal counts: {sampling['ideal_counts']}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(f"Final R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Learned log Z: {trainer.log_z.detach().item():.4f}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
