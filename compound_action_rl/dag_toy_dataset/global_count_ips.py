"""Count-IPS with global outcome frequencies, coverage-gated tempering, and novelty.

Instead of estimating ``p_hat(x)`` from the current batch only, this variant uses
Laplace-smoothed global visit counts accumulated across training:

    p_hat(x_i) = (n_global(x_i) + alpha) / (N_total + alpha * K)

Advantages are built from tempered IPS scores plus a novelty bonus for
under-visited terminals:

    scaled_i = R(x_i) ** beta / p_hat(x_i) + lambda / sqrt(1 + n_global(x_i))
    advantage = normalize(scaled)

``beta`` is coverage-gated: it cannot anneal faster than terminal discovery.
Singleton batch outcomes still have negative advantages floored at zero.

This targets the two failure modes of small-batch count-IPS:
1. batch ``p_hat`` makes every unique outcome equally rare, so reward ranking
   dominates among singletons;
2. reward sharpening outruns coverage before low-reward terminals are seen.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import TrainConfig
from count_ips import CountIPSTrainer, Episode
from dag_env import State
from run_count_ips import (
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)

ScheduleKind = Literal["linear", "cosine"]


@dataclass(frozen=True)
class GlobalCountIPSConfig:
    """Hyperparameters for global-frequency count-IPS."""

    beta_start: float = 0.1
    beta_end: float = 1.0
    anneal_updates: int | None = None
    schedule: ScheduleKind = "cosine"
    smoothing_alpha: float = 1.0
    novelty_coef: float = 1.0
    clip_singleton_negative: bool = True

    def validate(self) -> None:
        if self.beta_start <= 0.0 or self.beta_end <= 0.0:
            raise ValueError("beta values must be > 0")
        if self.beta_end < self.beta_start:
            raise ValueError("beta_end must be >= beta_start")
        if self.anneal_updates is not None and self.anneal_updates < 1:
            raise ValueError("anneal_updates must be >= 1")
        if self.schedule not in ("linear", "cosine"):
            raise ValueError("schedule must be 'linear' or 'cosine'")
        if self.smoothing_alpha <= 0.0:
            raise ValueError("smoothing_alpha must be > 0")
        if self.novelty_coef < 0.0:
            raise ValueError("novelty_coef must be >= 0")


def schedule_progress(
    update_step: int,
    *,
    anneal_updates: int,
    schedule: ScheduleKind,
) -> float:
    if update_step < 1:
        raise ValueError("update_step must be >= 1")
    if anneal_updates < 1:
        raise ValueError("anneal_updates must be >= 1")
    if anneal_updates == 1:
        progress = 1.0
    else:
        progress = min((update_step - 1) / (anneal_updates - 1), 1.0)
    if schedule == "cosine":
        progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    return float(progress)


def coverage_gated_beta(
    *,
    beta_start: float,
    beta_end: float,
    update_step: int,
    anneal_updates: int,
    schedule: ScheduleKind,
    coverage_fraction: float,
) -> tuple[float, float]:
    """Return beta and the effective anneal progress after the coverage gate."""
    if not 0.0 <= coverage_fraction <= 1.0:
        raise ValueError("coverage_fraction must be in [0, 1]")
    progress = schedule_progress(
        update_step, anneal_updates=anneal_updates, schedule=schedule
    )
    effective_progress = min(progress, coverage_fraction)
    beta = beta_start + (beta_end - beta_start) * effective_progress
    return float(beta), float(effective_progress)


def global_smoothed_probabilities(
    outcome_ids: Sequence[object],
    global_counts: Mapping[object, int],
    *,
    num_terminals: int,
    smoothing_alpha: float,
) -> np.ndarray:
    total = float(sum(global_counts.values()))
    denominator = total + smoothing_alpha * num_terminals
    return np.array(
        [
            (float(global_counts.get(outcome, 0)) + smoothing_alpha) / denominator
            for outcome in outcome_ids
        ],
        dtype=np.float64,
    )


def global_count_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    global_counts: Mapping[object, int],
    *,
    num_terminals: int,
    reward_beta: float,
    smoothing_alpha: float = 1.0,
    novelty_coef: float = 1.0,
    clip_singleton_negative: bool = True,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute global-frequency tempered count-IPS advantages."""
    if len(rewards) == 0 or len(rewards) != len(outcome_ids):
        raise ValueError("rewards and outcome_ids must have the same non-zero length")
    if reward_beta <= 0.0:
        raise ValueError("reward_beta must be > 0")
    if num_terminals < 1:
        raise ValueError("num_terminals must be >= 1")

    reward_array = np.asarray(rewards, dtype=np.float64)
    if np.any(reward_array <= 0.0):
        raise ValueError("rewards must be strictly positive for global count-IPS")

    batch_counts = Counter(outcome_ids)
    group_size = len(outcome_ids)
    p_hat = global_smoothed_probabilities(
        outcome_ids,
        global_counts,
        num_terminals=num_terminals,
        smoothing_alpha=smoothing_alpha,
    )
    visit_counts = np.array(
        [float(global_counts.get(outcome, 0)) for outcome in outcome_ids],
        dtype=np.float64,
    )
    tempered_rewards = np.power(reward_array, reward_beta)
    novelty = (
        novelty_coef / np.sqrt(1.0 + visit_counts) if novelty_coef > 0.0 else 0.0
    )
    scaled = tempered_rewards / p_hat + novelty
    centered = scaled - scaled.mean()
    std = float(scaled.std())
    advantages = centered if std < eps else centered / (std + eps)

    clipped_singletons = 0
    if clip_singleton_negative:
        for index, outcome in enumerate(outcome_ids):
            if batch_counts[outcome] == 1 and advantages[index] < 0.0:
                advantages[index] = 0.0
                clipped_singletons += 1

    inverse = 1.0 / np.maximum(p_hat, eps)
    ess = float(inverse.sum() ** 2 / np.maximum(np.square(inverse).sum(), eps))
    return advantages, {
        "ips_prob_mean": float(p_hat.mean()),
        "ips_prob_min": float(p_hat.min()),
        "ips_prob_max": float(p_hat.max()),
        "ips_unique_outcomes": float(len(batch_counts)),
        "ips_max_outcome_count": float(max(batch_counts.values())),
        "ips_min_outcome_count": float(min(batch_counts.values())),
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
        "novelty_mean": float(novelty.mean()) if novelty_coef > 0.0 else 0.0,
        "global_visit_mean": float(visit_counts.mean()),
        "global_visit_min": float(visit_counts.min()),
        "global_visit_max": float(visit_counts.max()),
        "singleton_negative_clipped": float(clipped_singletons),
    }


class GlobalCountIPSTrainer(CountIPSTrainer):
    """Count-IPS trainer using global outcome frequencies and coverage-gated beta."""

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        global_ips: GlobalCountIPSConfig | None = None,
    ) -> None:
        super().__init__(config, device=device)
        self.global_ips = global_ips or GlobalCountIPSConfig()
        self.global_ips.validate()
        self.anneal_updates = (
            self.config.num_updates
            if self.global_ips.anneal_updates is None
            else self.global_ips.anneal_updates
        )
        self._global_terminal_counts: Counter[State] = Counter()
        self.current_beta = self.global_ips.beta_start
        self.current_beta_progress = 0.0
        self.current_coverage_fraction = 0.0

    def _on_update_start(self, update_step: int) -> None:
        self.current_coverage_fraction = len(self._seen_terminals) / len(self.terminals)
        self.current_beta, self.current_beta_progress = coverage_gated_beta(
            beta_start=self.global_ips.beta_start,
            beta_end=self.global_ips.beta_end,
            update_step=update_step,
            anneal_updates=self.anneal_updates,
            schedule=self.global_ips.schedule,
            coverage_fraction=self.current_coverage_fraction,
        )

    def _group_advantages(self, episodes: list[Episode]) -> float:
        for episode in episodes:
            self._global_terminal_counts[episode.terminal] += 1

        advantages, metrics = global_count_ips_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            self._global_terminal_counts,
            num_terminals=len(self.terminals),
            reward_beta=self.current_beta,
            smoothing_alpha=self.global_ips.smoothing_alpha,
            novelty_coef=self.global_ips.novelty_coef,
            clip_singleton_negative=self.global_ips.clip_singleton_negative,
            eps=self.config.advantage_eps,
        )
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        metrics.update(
            {
                "coverage_fraction": float(self.current_coverage_fraction),
                "beta_progress": float(self.current_beta_progress),
                "global_total_visits": float(sum(self._global_terminal_counts.values())),
            }
        )
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
                    "name": "global_count_ips",
                    "global_ips": asdict(self.global_ips),
                    "current_beta": self.current_beta,
                    "current_beta_progress": self.current_beta_progress,
                    "current_coverage_fraction": self.current_coverage_fraction,
                    "global_terminal_counts": {
                        state.signature: int(count)
                        for state, count in self._global_terminal_counts.items()
                    },
                },
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str, *, device: str = "cpu") -> "GlobalCountIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload.get("algorithm", {})
        if algorithm.get("name") != "global_count_ips":
            raise ValueError("checkpoint is not a global Count-IPS run")
        trainer = cls(
            payload["config"],
            device=device,
            global_ips=GlobalCountIPSConfig(**algorithm["global_ips"]),
        )
        trainer.current_beta = float(algorithm["current_beta"])
        trainer.current_beta_progress = float(algorithm["current_beta_progress"])
        trainer.current_coverage_fraction = float(
            algorithm["current_coverage_fraction"]
        )
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        for signature, count in algorithm.get("global_terminal_counts", {}).items():
            x_text, y_text = signature.strip("()").split(",")
            terminal = State(int(x_text), int(y_text))
            trainer._global_terminal_counts[terminal] = int(count)
            trainer._seen_terminals.add(terminal)
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer


def _plot_global_ips_diagnostics(history: list[dict], output: Path) -> None:
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(steps, [row["reward_beta"] for row in history], label="beta", color="#6c5ce7")
    ax.plot(
        steps,
        [row["coverage_fraction"] for row in history],
        label="coverage fraction",
        color="#00b894",
    )
    ax.set_xlabel("update")
    ax.set_ylabel("value")
    ax.set_title("Coverage-gated reward tempering")
    ax.grid(alpha=0.22)
    ax.legend()

    ax = axes[1]
    ax.plot(
        steps,
        [row["global_total_visits"] for row in history],
        label="global visits",
        color="#0984e3",
    )
    ax.plot(
        steps,
        [row["global_unique_outcomes"] for row in history],
        label="unique outcomes",
        color="#e17055",
    )
    ax.set_xlabel("update")
    ax.set_ylabel("count")
    ax.set_title("Global discovery and visits")
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
    parser.add_argument("--beta-start", type=float, default=0.1)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--anneal-updates", type=int, default=None)
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--novelty-coef", type=float, default=1.0)
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
    global_ips = GlobalCountIPSConfig(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        anneal_updates=args.anneal_updates,
        schedule=args.schedule,
        smoothing_alpha=args.smoothing_alpha,
        novelty_coef=args.novelty_coef,
        clip_singleton_negative=not args.no_singleton_clip,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "global_count_ips_runs"
        / f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}_seed{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = GlobalCountIPSTrainer(
        config, device=_resolve_device(args.device), global_ips=global_ips
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Global Count-IPS: "
        f"beta {global_ips.beta_start:g}->{global_ips.beta_end:g} "
        f"(coverage-gated {global_ips.schedule}), "
        f"alpha={global_ips.smoothing_alpha:g}, "
        f"novelty={global_ips.novelty_coef:g}, "
        f"singleton clip={'on' if global_ips.clip_singleton_negative else 'off'}"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "global_ips_config": asdict(global_ips),
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
    _plot_global_ips_diagnostics(history, run_dir / "global_ips_diagnostics.png")
    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Global Count-IPS vs ideal reward sampling",
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
        "global_ips": asdict(global_ips),
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
