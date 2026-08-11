"""Count-IPS PPO with annealed epsilon/temperature exploration.

Training samples each direction and step action from

    q(a | s) = (1 - epsilon) * softmax(logits / temperature)
                + epsilon * Uniform(valid actions).

Both factors of the compound action use the same mixture.  PPO ratios are
computed under ``q`` as well, which keeps the rollout policy and the optimized
policy consistent.  Evaluation bypasses the mixture and samples from the
ordinary learned policy (epsilon=0, temperature=1).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.distributions import Categorical  # noqa: E402

from config import TrainConfig  # noqa: E402
from count_ips import CountIPSTrainer, Episode  # noqa: E402
from run_count_ips import (  # noqa: E402
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)


ScheduleKind = Literal["linear", "cosine"]


@dataclass(frozen=True)
class ExplorationConfig:
    """Exploration schedule applied during training rollouts and PPO updates."""

    epsilon_start: float = 0.30
    epsilon_end: float = 0.02
    temperature_start: float = 2.0
    temperature_end: float = 1.0
    anneal_updates: int | None = None
    schedule: ScheduleKind = "cosine"

    def validate(self) -> None:
        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon_start must be in [0, 1]")
        if not 0.0 <= self.epsilon_end <= 1.0:
            raise ValueError("epsilon_end must be in [0, 1]")
        if self.temperature_start <= 0.0 or self.temperature_end <= 0.0:
            raise ValueError("temperatures must be > 0")
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


def epsilon_temperature_distribution(
    policy_distribution: Categorical,
    mask: torch.Tensor,
    *,
    epsilon: float,
    temperature: float,
) -> Categorical:
    """Return a temperature-scaled policy mixed with valid-action uniform."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    mask = mask.bool()
    logits = policy_distribution.logits
    if mask.shape != logits.shape:
        raise ValueError("mask must have the same shape as distribution logits")
    if torch.any(~mask.any(dim=-1)):
        raise ValueError("every distribution row needs a valid action")

    tempered_logits = torch.where(
        mask,
        logits / temperature,
        torch.full_like(logits, float("-inf")),
    )
    tempered_probabilities = torch.softmax(tempered_logits, dim=-1)
    uniform_probabilities = mask.to(logits.dtype)
    uniform_probabilities = uniform_probabilities / uniform_probabilities.sum(
        dim=-1, keepdim=True
    )
    probabilities = (
        (1.0 - epsilon) * tempered_probabilities
        + epsilon * uniform_probabilities
    )
    return Categorical(probs=probabilities)


class EpsilonGreedyCountIPSTrainer(CountIPSTrainer):
    """Count-IPS trainer using annealed epsilon-uniform policy exploration."""

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        exploration: ExplorationConfig | None = None,
    ) -> None:
        super().__init__(config, device=device)
        self.exploration = exploration or ExplorationConfig()
        self.exploration.validate()
        self.anneal_updates = (
            self.config.num_updates
            if self.exploration.anneal_updates is None
            else self.exploration.anneal_updates
        )
        self.current_epsilon = self.exploration.epsilon_start
        self.current_temperature = self.exploration.temperature_start

    def _on_update_start(self, update_step: int) -> None:
        self.current_epsilon = annealed_value(
            self.exploration.epsilon_start,
            self.exploration.epsilon_end,
            update_step=update_step,
            anneal_updates=self.anneal_updates,
            schedule=self.exploration.schedule,
        )
        self.current_temperature = annealed_value(
            self.exploration.temperature_start,
            self.exploration.temperature_end,
            update_step=update_step,
            anneal_updates=self.anneal_updates,
            schedule=self.exploration.schedule,
        )

    def _action_distribution(
        self,
        distribution: Categorical,
        mask: torch.Tensor,
        *,
        explore: bool,
    ) -> Categorical:
        if not explore:
            return distribution
        return epsilon_temperature_distribution(
            distribution,
            mask,
            epsilon=self.current_epsilon,
            temperature=self.current_temperature,
        )

    def _group_advantages(self, episodes: list[Episode]) -> float:
        ess = super()._group_advantages(episodes)
        self._last_ips_metrics.update(
            {
                "exploration_epsilon": self.current_epsilon,
                "exploration_temperature": self.current_temperature,
            }
        )
        return ess

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
                    "name": "epsilon_greedy_count_ips",
                    "exploration": asdict(self.exploration),
                    "current_epsilon": self.current_epsilon,
                    "current_temperature": self.current_temperature,
                },
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "EpsilonGreedyCountIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload.get("algorithm", {})
        if algorithm.get("name") != "epsilon_greedy_count_ips":
            raise ValueError("checkpoint is not an epsilon-greedy Count-IPS run")
        trainer = cls(
            payload["config"],
            device=device,
            exploration=ExplorationConfig(**algorithm["exploration"]),
        )
        trainer.current_epsilon = float(algorithm["current_epsilon"])
        trainer.current_temperature = float(algorithm["current_temperature"])
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer


def _plot_exploration(history: list[dict], output: Path) -> None:
    steps = [row["step"] for row in history]
    fig, epsilon_axis = plt.subplots(figsize=(8, 4.5))
    temperature_axis = epsilon_axis.twinx()
    epsilon_axis.plot(
        steps,
        [row["exploration_epsilon"] for row in history],
        color="#d63031",
        label="epsilon",
    )
    temperature_axis.plot(
        steps,
        [row["exploration_temperature"] for row in history],
        color="#0984e3",
        label="temperature",
    )
    epsilon_axis.set_xlabel("update")
    epsilon_axis.set_ylabel("epsilon", color="#d63031")
    temperature_axis.set_ylabel("temperature", color="#0984e3")
    epsilon_axis.grid(alpha=0.22)
    fig.suptitle("Training exploration schedule")
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
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--temperature-start", type=float, default=2.0)
    parser.add_argument("--temperature-end", type=float, default=1.0)
    parser.add_argument("--anneal-updates", type=int, default=None)
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="cosine")
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
    exploration = ExplorationConfig(
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        anneal_updates=args.anneal_updates,
        schedule=args.schedule,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "epsilon_greedy_count_ips_runs"
        / f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}_seed{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = EpsilonGreedyCountIPSTrainer(
        config, device=_resolve_device(args.device), exploration=exploration
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Exploration: "
        f"epsilon {exploration.epsilon_start:g}->{exploration.epsilon_end:g}, "
        f"temperature {exploration.temperature_start:g}->{exploration.temperature_end:g}, "
        f"{exploration.schedule} over {trainer.anneal_updates} updates"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "exploration_config": asdict(exploration),
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
    _plot_exploration(history, run_dir / "exploration_schedule.png")
    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Annealed exploratory Count-IPS vs ideal reward sampling",
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
        "exploration": asdict(exploration),
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
