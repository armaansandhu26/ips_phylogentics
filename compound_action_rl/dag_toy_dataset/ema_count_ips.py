"""Count-IPS using a global EMA of batch outcome frequencies.

For each independently sampled group, let ``f_t(o)`` be the fraction of the
group ending at outcome ``o``.  The global probability estimate is updated as

    p_ema_t(o) = (1 - alpha) * p_ema_{t-1}(o) + alpha * f_t(o)

for every outcome seen so far. By default, outcomes absent from the current
group are updated with ``f_t(o) = 0``. The optional stale-absent mode instead
updates only outcomes present in the group and leaves all other values
unchanged. The first-batch initialization mode initializes the EMA directly;
the uniform mode initializes every terminal to ``1 / num_terminals``.

The resulting estimate replaces the batch-local frequency in Count-IPS:

    scaled_i = reward(x_i) / p_ema_t(x_i)

The default mode normalizes these scores into advantages. The optional raw
mode uses ``scaled_i`` directly as the detached PPO episode weight, matching
the reference EMA-REINFORCE setup as closely as the DAG token policy permits.

A separate lifetime counter is retained for diagnostics; EMA values are not
literal visit counts. In stale-absent mode they are also not a normalized
probability distribution.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from config import TrainConfig
from count_ips import CountIPSTrainer, Episode
from dag_env import State
from run_count_ips import (
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)


@dataclass(frozen=True)
class EMACountIPSConfig:
    """Hyperparameters for EMA Count-IPS."""

    alpha: float = 0.1
    initialization: Literal["first_batch", "uniform"] = "first_batch"
    decay_absent_outcomes: bool = True
    tracker_eps: float = 1e-6
    ips_weight_mode: Literal["normalized", "raw"] = "normalized"

    def validate(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.initialization not in ("first_batch", "uniform"):
            raise ValueError("initialization must be 'first_batch' or 'uniform'")
        if self.tracker_eps <= 0.0:
            raise ValueError("tracker_eps must be > 0")
        if self.ips_weight_mode not in ("normalized", "raw"):
            raise ValueError("ips_weight_mode must be 'normalized' or 'raw'")


def update_ema_outcome_frequencies(
    outcome_ids: Sequence[object],
    ema_frequencies: MutableMapping[object, float],
    *,
    alpha: float,
    decay_absent_outcomes: bool = True,
) -> dict[object, float]:
    """Update outcome EMA values in place and return batch frequencies."""
    if len(outcome_ids) == 0:
        raise ValueError("outcome_ids must be non-empty")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    batch_counts = Counter(outcome_ids)
    group_size = len(outcome_ids)
    batch_frequencies = {
        outcome: count / group_size for outcome, count in batch_counts.items()
    }

    if not ema_frequencies:
        ema_frequencies.update(batch_frequencies)
        return batch_frequencies

    outcomes_to_update = (
        set(ema_frequencies) | set(batch_frequencies)
        if decay_absent_outcomes
        else set(batch_frequencies)
    )
    for outcome in outcomes_to_update:
        previous = float(ema_frequencies.get(outcome, 0.0))
        current = float(batch_frequencies.get(outcome, 0.0))
        ema_frequencies[outcome] = (1.0 - alpha) * previous + alpha * current
    return batch_frequencies


def ema_count_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    ema_frequencies: Mapping[object, float],
    *,
    tracker_eps: float = 1e-6,
    normalize: bool = True,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute raw or normalized ``reward / EMA frequency`` episode weights."""
    if len(rewards) == 0 or len(rewards) != len(outcome_ids):
        raise ValueError("rewards and outcome_ids must have the same non-zero length")

    reward_array = np.asarray(rewards, dtype=np.float64)
    p_hat = np.asarray(
        [float(ema_frequencies.get(outcome, 0.0)) for outcome in outcome_ids],
        dtype=np.float64,
    )
    if np.any(p_hat <= 0.0):
        raise ValueError("every sampled outcome must have a positive EMA frequency")
    if tracker_eps <= 0.0:
        raise ValueError("tracker_eps must be > 0")

    pi_estimate = np.maximum(p_hat, tracker_eps)
    scaled = reward_array / pi_estimate
    std = float(scaled.std())
    if normalize:
        centered = scaled - scaled.mean()
        advantages = centered if std < eps else centered / (std + eps)
    else:
        advantages = scaled.copy()

    batch_counts = Counter(outcome_ids)
    inverse = 1.0 / pi_estimate
    ess = float(inverse.sum() ** 2 / np.maximum(np.square(inverse).sum(), eps))
    return advantages, {
        "ips_prob_mean": float(pi_estimate.mean()),
        "ips_prob_min": float(pi_estimate.min()),
        "ips_prob_max": float(pi_estimate.max()),
        "ips_unique_outcomes": float(len(batch_counts)),
        "ips_max_outcome_count": float(max(batch_counts.values())),
        "ips_min_outcome_count": float(min(batch_counts.values())),
        "ips_scaled_reward_mean": float(scaled.mean()),
        "ips_scaled_reward_std": std,
        "ips_ess": ess,
        "ips_ess_fraction": ess / len(outcome_ids),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        "ema_probability_mass": float(sum(ema_frequencies.values())),
        "ema_tracked_outcomes": float(len(ema_frequencies)),
    }


class EMACountIPSTrainer(CountIPSTrainer):
    """Count-IPS trainer backed by global EMA outcome frequencies."""

    probability_label = "p_ema"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        ema_ips: EMACountIPSConfig | None = None,
    ) -> None:
        super().__init__(config, device=device)
        self.ema_ips = ema_ips or EMACountIPSConfig()
        self.ema_ips.validate()
        if self.ema_ips.initialization == "uniform":
            initial_probability = 1.0 / len(self.terminals)
            self._ema_terminal_frequencies = {
                state: initial_probability for state in self.terminals
            }
        else:
            self._ema_terminal_frequencies = {}
        self._lifetime_terminal_counts: Counter[State] = Counter()

    @property
    def ema_terminal_frequencies(self) -> Mapping[State, float]:
        return self._ema_terminal_frequencies

    @property
    def lifetime_terminal_counts(self) -> Mapping[State, int]:
        return self._lifetime_terminal_counts

    def _group_advantages(self, episodes: list[Episode]) -> float:
        outcomes = [episode.terminal for episode in episodes]
        self._lifetime_terminal_counts.update(outcomes)
        update_ema_outcome_frequencies(
            outcomes,
            self._ema_terminal_frequencies,
            alpha=self.ema_ips.alpha,
            decay_absent_outcomes=self.ema_ips.decay_absent_outcomes,
        )
        advantages, metrics = ema_count_ips_advantages(
            [episode.reward for episode in episodes],
            outcomes,
            self._ema_terminal_frequencies,
            tracker_eps=self.ema_ips.tracker_eps,
            normalize=self.ema_ips.ips_weight_mode == "normalized",
            eps=self.config.advantage_eps,
        )
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        metrics.update(
            {
                "ema_alpha": self.ema_ips.alpha,
                "ema_effective_groups": 1.0 / self.ema_ips.alpha,
                "ema_absent_decay": float(self.ema_ips.decay_absent_outcomes),
                "ema_raw_ips_weights": float(
                    self.ema_ips.ips_weight_mode == "raw"
                ),
                "global_total_visits": float(
                    sum(self._lifetime_terminal_counts.values())
                ),
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
                    "name": "ema_count_ips",
                    "ema_ips": asdict(self.ema_ips),
                    "ema_terminal_frequencies": {
                        state.signature: probability
                        for state, probability in self._ema_terminal_frequencies.items()
                    },
                    "lifetime_terminal_counts": {
                        state.signature: count
                        for state, count in self._lifetime_terminal_counts.items()
                    },
                },
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str, *, device: str = "cpu") -> "EMACountIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = payload.get("algorithm", {})
        if algorithm.get("name") != "ema_count_ips":
            raise ValueError("checkpoint is not an EMA Count-IPS run")
        trainer = cls(
            payload["config"],
            device=device,
            ema_ips=EMACountIPSConfig(**algorithm["ema_ips"]),
        )
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer._ema_terminal_frequencies.clear()

        def state_from_signature(signature: str) -> State:
            x_text, y_text = signature.strip("()").split(",")
            return State(int(x_text), int(y_text))

        for signature, probability in algorithm.get(
            "ema_terminal_frequencies", {}
        ).items():
            trainer._ema_terminal_frequencies[state_from_signature(signature)] = float(
                probability
            )
        for signature, count in algorithm.get("lifetime_terminal_counts", {}).items():
            state = state_from_signature(signature)
            trainer._lifetime_terminal_counts[state] = int(count)
            trainer._seen_terminals.add(state)
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        return trainer


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
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tracker-eps", type=float, default=1e-6)
    parser.add_argument(
        "--initialization",
        choices=("first_batch", "uniform"),
        default="first_batch",
        help="initialize the EMA from the first batch or uniformly over terminals",
    )
    parser.add_argument(
        "--keep-absent-stale",
        action="store_true",
        help="update only outcomes present in a batch; leave absent values unchanged",
    )
    parser.add_argument(
        "--ips-weight-mode",
        choices=("normalized", "raw"),
        default="normalized",
        help="z-normalize reward/p or use raw reward/p as the PPO episode weight",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
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
    ema_ips = EMACountIPSConfig(
        alpha=args.alpha,
        initialization=args.initialization,
        decay_absent_outcomes=not args.keep_absent_stale,
        tracker_eps=args.tracker_eps,
        ips_weight_mode=args.ips_weight_mode,
    )
    update_mode = "decay_absent" if ema_ips.decay_absent_outcomes else "stale_absent"
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "ema_count_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}_"
            f"{ema_ips.initialization}_{update_mode}_{ema_ips.ips_weight_mode}_"
            f"seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = EMACountIPSTrainer(
        config, device=_resolve_device(args.device), ema_ips=ema_ips
    )
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        f"EMA Count-IPS: alpha={ema_ips.alpha:g}, "
        f"initialization={ema_ips.initialization}, "
        f"update_mode={update_mode}, "
        f"weight_mode={ema_ips.ips_weight_mode}, "
        f"tracker_eps={ema_ips.tracker_eps:g}, "
        f"effective window~{1.0 / ema_ips.alpha:g} groups"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "ema_ips_config": asdict(ema_ips),
                "device": str(trainer.device),
                "checkpoint_every": args.checkpoint_every,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history: list[dict[str, Any]] = trainer.train(
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
    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="EMA Count-IPS vs ideal reward sampling",
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
        "ema_ips": asdict(ema_ips),
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
