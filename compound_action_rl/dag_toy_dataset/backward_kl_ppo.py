"""Backward-reference maximum-entropy PPO for the direction/step DAG.

For a sampled forward trajectory ``tau`` ending at ``x``, optimize the score

    beta * log R(x) + log P_B(tau | x) - log P_F(tau)

where ``P_B`` is a fixed, locally normalized reverse policy.  At beta=1 the
optimal trajectory distribution is ``P_F(tau) proportional to
R(x) P_B(tau | x)``.  Because the reverse probabilities over all paths ending
at any fixed terminal sum to one, the terminal marginal is exactly
``P_F(x) proportional to R(x)`` without terminal enumeration, path counts, or
within-batch output collisions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import TrainConfig
from count_ips import Episode
from dag_env import State, uniform_backward_log_probability
from run_count_ips import _plot_final_counts, _plot_trajectory_diagnostics
from trajectory_ips import FullTrajectoryPPOTrainer, _resolve_device


def backward_kl_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    forward_log_probabilities: Sequence[float],
    backward_log_probabilities: Sequence[float],
    *,
    reward_beta: float = 1.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize sampled trajectory-KL scores for a PPO update."""
    size = len(rewards)
    if (
        size == 0
        or len(outcome_ids) != size
        or len(forward_log_probabilities) != size
        or len(backward_log_probabilities) != size
    ):
        raise ValueError("all trajectory score inputs must have the same non-zero length")
    if reward_beta < 0:
        raise ValueError("reward_beta must be >= 0")

    reward_array = np.asarray(rewards, dtype=np.float64)
    if np.any(reward_array <= 0):
        raise ValueError("backward-KL PPO requires strictly positive rewards")
    forward = np.asarray(forward_log_probabilities, dtype=np.float64)
    backward = np.asarray(backward_log_probabilities, dtype=np.float64)
    scores = reward_beta * np.log(reward_array) + backward - forward
    centered = scores - scores.mean()
    std = float(scores.std())
    advantages = centered if std < eps else centered / (std + eps)

    # The exponentiated score is the unnormalized target/current path ratio.
    # Centering before exponentiation avoids overflow and does not change ESS.
    ratio_weights = np.exp(scores - scores.max())
    ess = float(
        ratio_weights.sum() ** 2 / np.maximum(np.square(ratio_weights).sum(), eps)
    )
    outcome_counts = Counter(outcome_ids)
    forward_probabilities = np.exp(forward)
    metrics = {
        # Compatibility keys used by the shared training loop.
        "ips_prob_mean": float(forward_probabilities.mean()),
        "ips_prob_min": float(forward_probabilities.min()),
        "ips_prob_max": float(forward_probabilities.max()),
        "ips_unique_outcomes": float(len(outcome_counts)),
        "ips_max_outcome_count": float(max(outcome_counts.values())),
        "ips_min_outcome_count": float(min(outcome_counts.values())),
        "ips_scaled_reward_mean": float(scores.mean()),
        "ips_scaled_reward_std": std,
        "ips_ess": ess,
        "ips_ess_fraction": ess / size,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        # Objective-specific diagnostics.
        "reward_beta": float(reward_beta),
        "trajectory_kl_score_mean": float(scores.mean()),
        "trajectory_kl_score_std": std,
        "forward_log_probability_mean": float(forward.mean()),
        "backward_log_probability_mean": float(backward.mean()),
        "log_path_ratio_mean": float((backward - forward).mean()),
    }
    return advantages, metrics


class BackwardKLPPOTrainer(FullTrajectoryPPOTrainer):
    """Full-path PPO on a reverse-reference trajectory KL objective."""

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        reward_beta_start: float = 0.25,
        reward_beta_end: float = 1.0,
        beta_anneal_updates: int | None = None,
    ) -> None:
        super().__init__(config, device=device)
        if self.config.entropy_coef != 0:
            raise ValueError(
                "set entropy_coef=0: -log P_F is already part of the KL objective"
            )
        if reward_beta_start < 0 or reward_beta_end < 0:
            raise ValueError("reward beta values must be >= 0")
        self.reward_beta_start = float(reward_beta_start)
        self.reward_beta_end = float(reward_beta_end)
        self.beta_anneal_updates = (
            self.config.num_updates
            if beta_anneal_updates is None
            else int(beta_anneal_updates)
        )
        if self.beta_anneal_updates < 1:
            raise ValueError("beta_anneal_updates must be >= 1")
        self.current_reward_beta = self.reward_beta_start

    def _on_update_start(self, update_step: int) -> None:
        denominator = max(self.beta_anneal_updates - 1, 1)
        progress = min(max((update_step - 1) / denominator, 0.0), 1.0)
        self.current_reward_beta = self.reward_beta_start + progress * (
            self.reward_beta_end - self.reward_beta_start
        )

    def _group_advantages(self, episodes: list[Episode]) -> float:
        forward_log_probabilities = [
            sum(step.log_prob_joint for step in episode.steps)
            for episode in episodes
        ]
        backward_log_probabilities = [
            uniform_backward_log_probability(
                episode.trajectory,
                max_step=self.config.max_step,
            )
            for episode in episodes
        ]
        advantages, metrics = backward_kl_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            forward_log_probabilities,
            backward_log_probabilities,
            reward_beta=self.current_reward_beta,
            eps=self.config.advantage_eps,
        )
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]

    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "direction_policy": self.direction_policy.state_dict(),
                "step_policy": self.step_policy.state_dict(),
                "update_step": update_step,
                "algorithm": {
                    "name": "backward_kl_ppo",
                    "reward_beta_start": self.reward_beta_start,
                    "reward_beta_end": self.reward_beta_end,
                    "beta_anneal_updates": self.beta_anneal_updates,
                    "current_reward_beta": self.current_reward_beta,
                    "backward_policy": "uniform_over_valid_parents",
                },
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "BackwardKLPPOTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload["algorithm"]
        trainer = cls(
            payload["config"],
            device=device,
            reward_beta_start=algorithm["reward_beta_start"],
            reward_beta_end=algorithm["reward_beta_end"],
            beta_anneal_updates=algorithm["beta_anneal_updates"],
        )
        trainer.current_reward_beta = algorithm["current_reward_beta"]
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer


def _plot_backward_kl_training(
    history: list[dict],
    trainer: BackwardKLPPOTrainer,
    *,
    output: Path,
) -> None:
    steps = [row["step"] for row in history]
    target = trainer.target_reward()
    target_mean_reward = sum(
        target[state] * trainer.reward_by_terminal[state] for state in trainer.terminals
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(steps, [row["mean_reward"] for row in history], label="sampled")
    ax.axhline(target_mean_reward, linestyle="--", color="#00b894", label="ideal")
    ax.set_title("Mean terminal reward")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(
        steps,
        [row["forward_log_probability_mean"] for row in history],
        label="log P_F",
    )
    ax.plot(
        steps,
        [row["backward_log_probability_mean"] for row in history],
        label="log P_B",
    )
    ax.set_title("Mean path log-probabilities")
    ax.legend()

    ax = axes[1, 0]
    if eval_rows:
        eval_steps = [row["step"] for row in eval_rows]
        ax.plot(
            eval_steps,
            [row["tv_reward_target"] for row in eval_rows],
            "o-",
            label="TV distance",
        )
        coverage_ax = ax.twinx()
        coverage_ax.plot(
            eval_steps,
            [row["terminals_hit"] for row in eval_rows],
            "s--",
            color="#e17055",
            label="terminals hit",
        )
        coverage_ax.set_ylabel("Terminals hit", color="#e17055")
    ax.set_ylim(bottom=0)
    ax.set_title("Terminal-distribution convergence")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    ax.plot(steps, [row["reward_beta"] for row in history], label="reward beta")
    score_ax = ax.twinx()
    score_ax.plot(
        steps,
        [row["trajectory_kl_score_std"] for row in history],
        color="#6c5ce7",
        label="score std",
    )
    ax.set_title("Target annealing and score spread")
    ax.set_ylabel("Reward beta")
    score_ax.set_ylabel("Score std", color="#6c5ce7")

    for ax in axes.flat:
        ax.set_xlabel("Update")
        ax.grid(alpha=0.22)
    fig.suptitle("Backward-reference trajectory-KL PPO")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=1_000)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--reward-beta-start", type=float, default=0.25)
    parser.add_argument("--reward-beta-end", type=float, default=1.0)
    parser.add_argument("--beta-anneal-updates", type=int, default=None)
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
        clip_ratio=args.clip_ratio,
        entropy_coef=0.0,
        seed=args.seed,
        log_every=args.log_every,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "backward_kl_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = BackwardKLPPOTrainer(
        config,
        device=_resolve_device(args.device),
        reward_beta_start=args.reward_beta_start,
        reward_beta_end=args.reward_beta_end,
        beta_anneal_updates=args.beta_anneal_updates,
    )
    checkpoint_every = args.checkpoint_every or None
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Algorithm: PPO on beta*log R(x) + log P_B(tau|x) - log P_F(tau); "
        "P_B is uniform over valid parents"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "device": str(trainer.device),
                "algorithm": "backward_kl_ppo",
                "objective": "beta*log R(x) + log P_B(tau|x) - log P_F(tau)",
                "backward_policy": "uniform_over_valid_parents",
                "reward_beta_start": trainer.reward_beta_start,
                "reward_beta_end": trainer.reward_beta_end,
                "beta_anneal_updates": trainer.beta_anneal_updates,
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
    _plot_backward_kl_training(history, trainer, output=training_plot)
    final_evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        final_evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Backward-reference PPO sampling vs ideal reward sampling",
    )
    trajectory_plot = run_dir / "trajectory_sampling.png"
    trajectory_sampling = _plot_trajectory_diagnostics(
        history,
        trainer,
        final_evaluation,
        samples=args.final_samples,
        output=trajectory_plot,
        subtitle="Target conditional paths follow the fixed reverse reference",
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary = {
        "environment": trainer.environment_summary(),
        "algorithm": {
            "name": "backward_kl_ppo",
            "objective": "beta*log R(x) + log P_B(tau|x) - log P_F(tau)",
            "backward_policy": "uniform_over_valid_parents",
            "final_reward_beta": trainer.current_reward_beta,
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
    print(f"Sampling plot: {run_dir / 'sampling_counts.png'}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
