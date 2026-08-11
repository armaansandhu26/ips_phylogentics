"""Count-based IPS with tempered rewards and singleton advantage clipping.

For each independently sampled group of G episodes:

    p_hat(x_i) = count(x_i) / G
    scaled_i   = R(x_i) ** beta / p_hat(x_i)
    advantage  = normalize(scaled)

After normalization, advantages for singleton outcomes (count == 1) are floored
at zero so rare low-reward discoveries are not actively suppressed.

``beta`` is annealed from ``beta_start`` to ``beta_end`` over training.  Early
flat rewards encourage broader terminal coverage; late ``beta -> 1`` restores
the standard count-IPS target ``p proportional to R``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import TrainConfig
from count_ips import CountIPSTrainer, Episode
from run_count_ips import (
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)

ScheduleKind = Literal["linear", "cosine"]


@dataclass(frozen=True)
class TemperingConfig:
    """Reward tempering schedule for count-IPS advantages."""

    beta_start: float = 0.3
    beta_end: float = 1.0
    anneal_updates: int | None = None
    schedule: ScheduleKind = "cosine"
    clip_singleton_negative: bool = True

    def validate(self) -> None:
        if self.beta_start <= 0.0 or self.beta_end <= 0.0:
            raise ValueError("beta values must be > 0")
        if self.anneal_updates is not None and self.anneal_updates < 1:
            raise ValueError("anneal_updates must be >= 1")
        if self.schedule not in ("linear", "cosine"):
            raise ValueError("schedule must be 'linear' or 'cosine'")


def annealed_value(
    start: float,
    end: float,
    *,
    update_step: int,
    anneal_updates: int,
    schedule: ScheduleKind,
) -> float:
    """Interpolate from ``start`` at update 1 to ``end`` at the horizon."""
    if update_step < 1:
        raise ValueError("update_step must be >= 1")
    if anneal_updates < 1:
        raise ValueError("anneal_updates must be >= 1")
    if schedule not in ("linear", "cosine"):
        raise ValueError("schedule must be 'linear' or 'cosine'")

    if anneal_updates == 1:
        progress = 1.0
    else:
        progress = min((update_step - 1) / (anneal_updates - 1), 1.0)
    if schedule == "cosine":
        progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    return float(start + progress * (end - start))


def tempered_count_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    *,
    reward_beta: float = 1.0,
    clip_singleton_negative: bool = True,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute ``R**beta / batch outcome frequency`` advantages with optional clipping."""
    if len(rewards) == 0 or len(rewards) != len(outcome_ids):
        raise ValueError("rewards and outcome_ids must have the same non-zero length")
    if reward_beta <= 0.0:
        raise ValueError("reward_beta must be > 0")

    reward_array = np.asarray(rewards, dtype=np.float64)
    if np.any(reward_array <= 0.0):
        raise ValueError("rewards must be strictly positive for tempered count-IPS")

    counts = Counter(outcome_ids)
    group_size = len(outcome_ids)
    p_hat = np.array([counts[outcome] / group_size for outcome in outcome_ids])
    tempered_rewards = np.power(reward_array, reward_beta)
    scaled = tempered_rewards / p_hat
    centered = scaled - scaled.mean()
    std = float(scaled.std())
    advantages = centered if std < eps else centered / (std + eps)

    clipped_singletons = 0
    if clip_singleton_negative:
        for index, outcome in enumerate(outcome_ids):
            if counts[outcome] == 1 and advantages[index] < 0.0:
                advantages[index] = 0.0
                clipped_singletons += 1

    inverse = 1.0 / p_hat
    ess = float(inverse.sum() ** 2 / np.square(inverse).sum())
    metrics = {
        "ips_prob_mean": float(p_hat.mean()),
        "ips_prob_min": float(p_hat.min()),
        "ips_prob_max": float(p_hat.max()),
        "ips_unique_outcomes": float(len(counts)),
        "ips_max_outcome_count": float(max(counts.values())),
        "ips_min_outcome_count": float(min(counts.values())),
        "ips_scaled_reward_mean": float(scaled.mean()),
        "ips_scaled_reward_std": std,
        "ips_ess": ess,
        "ips_ess_fraction": ess / group_size,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        "reward_beta": float(reward_beta),
        "tempered_reward_mean": float(tempered_rewards.mean()),
        "singleton_negative_clipped": float(clipped_singletons),
    }
    return advantages, metrics


class TemperedCountIPSTrainer(CountIPSTrainer):
    """Count-IPS trainer with annealed reward tempering and singleton clipping."""

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        tempering: TemperingConfig | None = None,
    ) -> None:
        super().__init__(config, device=device)
        self.tempering = tempering or TemperingConfig()
        self.tempering.validate()
        self.anneal_updates = (
            self.config.num_updates
            if self.tempering.anneal_updates is None
            else self.tempering.anneal_updates
        )
        self.current_beta = self.tempering.beta_start

    def _on_update_start(self, update_step: int) -> None:
        self.current_beta = annealed_value(
            self.tempering.beta_start,
            self.tempering.beta_end,
            update_step=update_step,
            anneal_updates=self.anneal_updates,
            schedule=self.tempering.schedule,
        )

    def _group_advantages(self, episodes: list[Episode]) -> float:
        advantages, metrics = tempered_count_ips_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            reward_beta=self.current_beta,
            clip_singleton_negative=self.tempering.clip_singleton_negative,
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
                    "name": "tempered_count_ips",
                    "tempering": asdict(self.tempering),
                    "current_beta": self.current_beta,
                },
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str, *, device: str = "cpu") -> "TemperedCountIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload.get("algorithm", {})
        if algorithm.get("name") != "tempered_count_ips":
            raise ValueError("checkpoint is not a tempered Count-IPS run")
        trainer = cls(
            payload["config"],
            device=device,
            tempering=TemperingConfig(**algorithm["tempering"]),
        )
        trainer.current_beta = float(algorithm["current_beta"])
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer


def _plot_tempering(history: list[dict], output: Path) -> None:
    steps = [row["step"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        steps,
        [row["reward_beta"] for row in history],
        color="#6c5ce7",
        label="reward beta",
    )
    ax.set_xlabel("update")
    ax.set_ylabel("beta")
    ax.set_title("Reward tempering schedule")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--beta-start", type=float, default=0.3)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--anneal-updates", type=int, default=None)
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument(
        "--no-singleton-clip",
        action="store_true",
        help="disable flooring negative advantages on singleton outcomes",
    )
    parser.add_argument("--terminal-rewards", type=float, nargs="+", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
        entropy_coef=args.entropy_coef,
        clip_ratio=args.clip_ratio,
        seed=args.seed,
        log_every=args.log_every,
    )
    tempering = TemperingConfig(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        anneal_updates=args.anneal_updates,
        schedule=args.schedule,
        clip_singleton_negative=not args.no_singleton_clip,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "tempered_count_ips_runs"
        / f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}_seed{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = TemperedCountIPSTrainer(
        config, device=_resolve_device(args.device), tempering=tempering
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Tempering: "
        f"beta {tempering.beta_start:g}->{tempering.beta_end:g}, "
        f"{tempering.schedule} over {trainer.anneal_updates} updates, "
        f"singleton clip={'on' if tempering.clip_singleton_negative else 'off'}"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "tempering_config": asdict(tempering),
                "device": str(trainer.device),
                "checkpoint_every": args.checkpoint_every,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history = trainer.train(
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=run_dir / "checkpoints",
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    trainer.save(run_dir / "checkpoint.pt", update_step=config.num_updates)

    _plot_training_curves(history, trainer, output=run_dir / "training_curves.png")
    _plot_tempering(history, run_dir / "tempering_schedule.png")
    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Tempered Count-IPS vs ideal reward sampling",
    )
    trajectories = _plot_trajectory_diagnostics(
        history,
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "trajectory_sampling.png",
    )
    summary = {
        "environment": trainer.environment_summary(),
        "tempering": asdict(tempering),
        "final_sampling": sampling,
        "trajectory_sampling": trajectories,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Final ideal-line R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
