"""Outcome IPS with a learned, normalized terminal propensity model.

This is the outcome-only counterpart to ``learned_reverse_ips.py``.  It does
not model a trajectory or a reverse action.  Instead, a network assigns one
scalar score to every terminal outcome and the scores are normalized across
the complete terminal frontier:

    s_phi(o)       = outcome_network(o)
    p_hat_phi(o)   = softmax({s_phi(o') : o' in O})[o]
    weight(o)      = R(o) / p_safe_phi(o)

The softmax is important.  A standalone sigmoid applied to ``o`` would not
make the estimates sum to one and therefore would not define a marginal
outcome distribution.

The ordering is deliberately lagged.  A frozen outcome model scores the same
G episodes used by PPO, PPO updates the forward policy, and only then is the
outcome model fit by cross-entropy on those episodes:

    rollout -> score with old q -> PPO update -> fit q on rollout outcomes

Consequently an episode is never scored by a model already fit to that
episode.  The model carries outcome-frequency information from earlier
batches, which is the point of this experiment.  An optional uniform mixture
can lower-bound propensities, but is disabled by default:

    p_safe(o) = (1 - epsilon) * p_hat_phi(o) + epsilon / |O|.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
import torch.nn.functional as F  # noqa: E402

from config import TrainConfig
from count_ips import CountIPSTrainer, Episode
from dag_env import State
from exact_probability_ips import _resolve_device
from run_count_ips import (  # noqa: E402
    _plot_final_counts,
    _plot_training_curves,
    _plot_trajectory_diagnostics,
)


@dataclass(frozen=True)
class LearnedOutcomeConfig:
    """Hyperparameters for the marginal outcome density estimator."""

    hidden_size: int = 128
    num_layers: int = 2
    lr: float = 1e-2
    train_epochs: int = 1
    grad_clip_norm: float = 1.0
    uniform_mix: float = 0.0

    def validate(self) -> None:
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("outcome hidden_size and num_layers must be >= 1")
        if self.lr <= 0.0:
            raise ValueError("outcome lr must be > 0")
        if self.train_epochs < 1:
            raise ValueError("outcome train_epochs must be >= 1")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("outcome grad_clip_norm must be > 0")
        if not 0.0 <= self.uniform_mix < 1.0:
            raise ValueError("outcome uniform_mix must be in [0, 1)")


def outcome_context(outcome: State, *, budget: int) -> tuple[float, float]:
    """Return scale-independent coordinates for a terminal outcome."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if outcome.depth != budget:
        raise ValueError("outcome must lie on the terminal frontier")
    scale = float(budget)
    return outcome.x / scale, outcome.y / scale


class LearnedOutcomeDistribution(nn.Module):
    """Energy model normalized over the supplied complete outcome set."""

    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("outcome network dimensions must be >= 1")
        layers: list[nn.Module] = []
        width = 2
        for _ in range(num_layers):
            layers.extend((nn.Linear(width, hidden_size), nn.Tanh()))
            width = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.score_head = nn.Linear(width, 1)

        # Zero scores give exactly 1 / |O| before the first observed batch.
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def scores(self, outcome_contexts: torch.Tensor) -> torch.Tensor:
        if outcome_contexts.ndim != 2 or outcome_contexts.shape[1] != 2:
            raise ValueError("outcome contexts must have shape (outcomes, 2)")
        if outcome_contexts.shape[0] < 2:
            raise ValueError("at least two candidate outcomes are required")
        return self.score_head(self.trunk(outcome_contexts)).squeeze(-1)

    def log_probabilities(self, outcome_contexts: torch.Tensor) -> torch.Tensor:
        """Normalize scores across the complete candidate outcome set."""
        return F.log_softmax(self.scores(outcome_contexts), dim=0)


def learned_outcome_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    outcome_log_probabilities: Sequence[float],
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute count-IPS advantages with learned probabilities replacing counts."""
    size = len(rewards)
    if (
        size == 0
        or len(outcome_ids) != size
        or len(outcome_log_probabilities) != size
    ):
        raise ValueError(
            "rewards, outcomes, and learned probabilities must have equal size"
        )

    reward_array = np.asarray(rewards, dtype=np.float64)
    log_p_hat = np.asarray(outcome_log_probabilities, dtype=np.float64)
    if np.any(~np.isfinite(reward_array)) or np.any(reward_array <= 0.0):
        raise ValueError("rewards must be finite and strictly positive")
    if np.any(~np.isfinite(log_p_hat)) or np.any(log_p_hat > 1e-7):
        raise ValueError("outcome log-probabilities must be finite and <= 0")

    probabilities = np.exp(log_p_hat)
    if np.any(probabilities <= 0.0):
        raise ValueError("outcome probabilities must be strictly positive")
    scaled = reward_array / probabilities
    std = float(scaled.std())
    centered = scaled - scaled.mean()
    advantages = centered if std < eps else centered / (std + eps)

    inverse = 1.0 / probabilities
    ess = float(inverse.sum() ** 2 / np.square(inverse).sum())
    counts = Counter(outcome_ids)
    log_weights = np.log(reward_array) - log_p_hat
    metrics = {
        "ips_prob_mean": float(probabilities.mean()),
        "ips_prob_min": float(probabilities.min()),
        "ips_prob_max": float(probabilities.max()),
        "ips_unique_outcomes": float(len(counts)),
        "ips_max_outcome_count": float(max(counts.values())),
        "ips_min_outcome_count": float(min(counts.values())),
        "ips_scaled_reward_mean": float(scaled.mean()),
        "ips_scaled_reward_std": std,
        "ips_ess": ess,
        "ips_ess_fraction": ess / size,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        "outcome_log_probability_mean": float(log_p_hat.mean()),
        "log_importance_weight_mean": float(log_weights.mean()),
        "log_importance_weight_min": float(log_weights.min()),
        "log_importance_weight_max": float(log_weights.max()),
    }
    return advantages, metrics


class LearnedOutcomeIPSTrainer(CountIPSTrainer):
    """Count-IPS PPO with its batch frequency replaced by lagged learned q."""

    probability_label = "p_safe(o)"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        outcome_config: LearnedOutcomeConfig | None = None,
        forward_lr_decay_after: int | None = None,
        forward_lr_after_decay: float | None = None,
    ) -> None:
        super().__init__(config, device=device)
        self.outcome_config = outcome_config or LearnedOutcomeConfig(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
        )
        self.outcome_config.validate()
        self.outcome_model = LearnedOutcomeDistribution(
            self.outcome_config.hidden_size,
            self.outcome_config.num_layers,
        ).to(self.device)
        self.outcome_optimizer = torch.optim.Adam(
            self.outcome_model.parameters(), lr=self.outcome_config.lr
        )
        self._terminal_contexts = torch.tensor(
            [
                outcome_context(state, budget=self.config.budget)
                for state in self.terminals
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._terminal_indices = {
            state: index for index, state in enumerate(self.terminals)
        }
        if (forward_lr_decay_after is None) != (forward_lr_after_decay is None):
            raise ValueError(
                "forward_lr_decay_after and forward_lr_after_decay must be set together"
            )
        if forward_lr_decay_after is not None and forward_lr_decay_after < 1:
            raise ValueError("forward_lr_decay_after must be >= 1")
        if forward_lr_after_decay is not None and forward_lr_after_decay <= 0:
            raise ValueError("forward_lr_after_decay must be > 0")
        self.forward_lr_decay_after = forward_lr_decay_after
        self.forward_lr_after_decay = forward_lr_after_decay
        self.current_forward_lr = float(self.config.lr)

    def _on_update_start(self, update_step: int) -> None:
        super()._on_update_start(update_step)
        learning_rate = float(self.config.lr)
        if (
            self.forward_lr_decay_after is not None
            and update_step > self.forward_lr_decay_after
        ):
            assert self.forward_lr_after_decay is not None
            learning_rate = float(self.forward_lr_after_decay)
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        self.current_forward_lr = learning_rate

    def _outcome_indices(self, episodes: list[Episode]) -> torch.Tensor:
        try:
            indices = [self._terminal_indices[episode.terminal] for episode in episodes]
        except KeyError as error:
            raise ValueError(
                "episode has an outcome outside the terminal frontier"
            ) from error
        if not indices:
            raise ValueError("episodes must be non-empty")
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def outcome_probabilities(self) -> np.ndarray:
        """Return safe, normalized propensities in trainer terminal order."""
        log_probabilities = self.outcome_model.log_probabilities(
            self._terminal_contexts
        )
        return (
            self._safe_log_probabilities(log_probabilities)
            .exp()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def _safe_log_probabilities(
        self, model_log_probabilities: torch.Tensor
    ) -> torch.Tensor:
        """Mix learned probabilities with uniform support in log space."""
        uniform_mix = self.outcome_config.uniform_mix
        if uniform_mix == 0.0:
            return model_log_probabilities
        uniform_log_probability = -float(np.log(len(self.terminals)))
        return torch.logaddexp(
            model_log_probabilities + float(np.log1p(-uniform_mix)),
            torch.full_like(
                model_log_probabilities,
                uniform_log_probability + float(np.log(uniform_mix)),
            ),
        )

    @torch.inference_mode()
    def outcome_log_probabilities(self, episodes: list[Episode]) -> np.ndarray:
        all_log_probabilities = self.outcome_model.log_probabilities(
            self._terminal_contexts
        )
        safe_log_probabilities = self._safe_log_probabilities(
            all_log_probabilities
        )
        selected = safe_log_probabilities[self._outcome_indices(episodes)]
        return selected.cpu().numpy().astype(np.float64)

    def _group_advantages(self, episodes: list[Episode]) -> float:
        log_probabilities = self.outcome_log_probabilities(episodes)
        advantages, metrics = learned_outcome_ips_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            log_probabilities,
            eps=self.config.advantage_eps,
        )
        metrics["forward_lr"] = self.current_forward_lr
        metrics["outcome_uniform_mix"] = self.outcome_config.uniform_mix
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]

    def _update_outcome_model(self, episodes: list[Episode]) -> dict[str, float]:
        outcome_indices = self._outcome_indices(episodes)
        outcome_counts = Counter(episode.terminal for episode in episodes)
        grad_norm_total = 0.0
        for _ in range(self.outcome_config.train_epochs):
            self.outcome_optimizer.zero_grad(set_to_none=True)
            log_probabilities = self.outcome_model.log_probabilities(
                self._terminal_contexts
            )
            loss = -log_probabilities[outcome_indices].mean()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.outcome_model.parameters(),
                self.outcome_config.grad_clip_norm,
            )
            self.outcome_optimizer.step()
            grad_norm_total += float(grad_norm.item())

        with torch.inference_mode():
            log_probabilities = self.outcome_model.log_probabilities(
                self._terminal_contexts
            )
            probabilities = log_probabilities.exp()
            empirical = torch.bincount(
                outcome_indices, minlength=len(self.terminals)
            ).to(dtype=probabilities.dtype)
            empirical /= outcome_indices.numel()
            positive = empirical > 0
            empirical_entropy = -(
                empirical[positive] * empirical[positive].log()
            ).sum()
            empirical_kl = (
                empirical[positive]
                * (empirical[positive].log() - log_probabilities[positive])
            ).sum()
            parameter_norm = sum(
                parameter.detach().norm().item() ** 2
                for parameter in self.outcome_model.parameters()
            ) ** 0.5
            return {
                "outcome_model_loss": float(
                    -log_probabilities[outcome_indices].mean().item()
                ),
                "outcome_model_entropy": float(
                    -(probabilities * log_probabilities).sum().item()
                ),
                "outcome_model_probability_mass": float(
                    probabilities.sum().item()
                ),
                "outcome_model_min_probability": float(
                    probabilities.min().item()
                ),
                "outcome_model_max_probability": float(
                    probabilities.max().item()
                ),
                "outcome_model_batch_tv": float(
                    (0.5 * torch.abs(probabilities - empirical).sum()).item()
                ),
                "outcome_model_batch_kl": float(empirical_kl.item()),
                "outcome_model_batch_entropy": float(empirical_entropy.item()),
                "outcome_model_grad_norm": (
                    grad_norm_total / self.outcome_config.train_epochs
                ),
                "outcome_model_param_norm": float(parameter_norm),
                "outcome_fit_samples": float(len(episodes)),
                "outcome_fit_unique_outcomes": float(len(outcome_counts)),
                "outcome_fit_min_count": float(min(outcome_counts.values())),
                "outcome_fit_max_count": float(max(outcome_counts.values())),
            }

    def _update_training_groups(
        self, groups: list[list[Episode]]
    ) -> dict[str, float]:
        """Update the forward policy first, then fit q on the identical batch."""
        episodes = [episode for group in groups for episode in group]
        policy_metrics = super()._update_training_groups(groups)
        outcome_metrics = self._update_outcome_model(episodes)
        return {
            **policy_metrics,
            **outcome_metrics,
            "total_rollouts_per_update": float(len(episodes)),
        }

    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "direction_policy": self.direction_policy.state_dict(),
                "step_policy": self.step_policy.state_dict(),
                "outcome_model": self.outcome_model.state_dict(),
                "outcome_config": asdict(self.outcome_config),
                "update_step": update_step,
                "algorithm": {
                    "name": "learned_outcome_ips",
                    "raw_weight": "R(o) / p_safe_phi(o)",
                    "forward_loss": "token_level_ppo",
                    "outcome_update_order": "forward_then_outcome_same_batch",
                    "forward_lr_decay_after": self.forward_lr_decay_after,
                    "forward_lr_after_decay": self.forward_lr_after_decay,
                    "advantage_normalization": "batch",
                },
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: Path | str, *, device: str = "cpu"
    ) -> "LearnedOutcomeIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("algorithm", {}).get("name") != "learned_outcome_ips":
            raise ValueError("checkpoint is not a learned-outcome IPS run")
        outcome_config_payload = payload["outcome_config"]
        outcome_config = (
            LearnedOutcomeConfig(**outcome_config_payload)
            if isinstance(outcome_config_payload, dict)
            else outcome_config_payload
        )
        algorithm = payload.get("algorithm", {})
        trainer = cls(
            payload["config"],
            device=device,
            outcome_config=outcome_config,
            forward_lr_decay_after=algorithm.get("forward_lr_decay_after"),
            forward_lr_after_decay=algorithm.get("forward_lr_after_decay"),
        )
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.outcome_model.load_state_dict(payload["outcome_model"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        trainer.outcome_model.eval()
        return trainer


def _plot_outcome_training(history: list[dict], *, output: Path) -> None:
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    axes[0, 0].plot(
        steps, [row["outcome_model_loss"] for row in history]
    )
    axes[0, 0].set_title("Outcome-model NLL")

    axes[0, 1].plot(
        steps,
        [row["outcome_model_batch_tv"] for row in history],
        label="TV(model, batch)",
    )
    axes[0, 1].plot(
        steps,
        [row["ips_ess_fraction"] for row in history],
        label="IPS ESS / batch",
    )
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_title("Density fit and IPS stability")
    axes[0, 1].legend()

    axes[1, 0].plot(
        steps,
        [row["outcome_model_min_probability"] for row in history],
        label="minimum",
    )
    axes[1, 0].plot(
        steps,
        [row["outcome_model_max_probability"] for row in history],
        label="maximum",
    )
    axes[1, 0].set_title("Learned terminal probabilities")
    axes[1, 0].legend()

    axes[1, 1].plot(
        steps,
        [row["outcome_model_entropy"] for row in history],
        label="model entropy",
    )
    axes[1, 1].plot(
        steps,
        [row["outcome_model_batch_entropy"] for row in history],
        label="batch entropy",
    )
    axes[1, 1].set_title("Outcome-distribution entropy")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Update")
        axis.grid(alpha=0.22)
    fig.suptitle("Learned outcome-propensity diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=2_000)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-decay-after", type=int, default=None)
    parser.add_argument("--lr-after-decay", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--outcome-lr", type=float, default=1e-2)
    parser.add_argument("--outcome-hidden-size", type=int, default=128)
    parser.add_argument("--outcome-num-layers", type=int, default=2)
    parser.add_argument(
        "--outcome-train-epochs",
        type=int,
        default=1,
        help="cross-entropy optimizer steps after each forward-policy update",
    )
    parser.add_argument("--outcome-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--outcome-uniform-mix", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=500)
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
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        log_every=args.log_every,
    )
    outcome_config = LearnedOutcomeConfig(
        hidden_size=args.outcome_hidden_size,
        num_layers=args.outcome_num_layers,
        lr=args.outcome_lr,
        train_epochs=args.outcome_train_epochs,
        grad_clip_norm=args.outcome_grad_clip_norm,
        uniform_mix=args.outcome_uniform_mix,
    )
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "learned_outcome_ips_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = LearnedOutcomeIPSTrainer(
        config,
        device=_resolve_device(args.device),
        outcome_config=outcome_config,
        forward_lr_decay_after=args.lr_decay_after,
        forward_lr_after_decay=args.lr_after_decay,
    )
    checkpoint_every = args.checkpoint_every or None
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Algorithm: R(o) / p_safe_phi(o); q scores the PPO batch before "
        "the forward update and is fit on that same batch afterward; "
        f"uniform_mix={outcome_config.uniform_mix}; no trajectory model"
    )

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "outcome_config": asdict(outcome_config),
                "device": str(trainer.device),
                "algorithm": "learned_outcome_ips",
                "raw_weight": "R(o) / p_safe_phi(o)",
                "forward_loss": "token_level_ppo",
                "outcome_update_order": "forward_then_outcome_same_batch",
                "forward_lr_decay_after": args.lr_decay_after,
                "forward_lr_after_decay": args.lr_after_decay,
                "advantage_normalization": "batch",
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
    _plot_training_curves(
        history,
        trainer,
        output=training_plot,
        propensity_title="Safe learned terminal propensity p_safe_phi(o)",
        suptitle="Learned outcome-propensity IPS training",
    )
    outcome_plot = run_dir / "outcome_model_diagnostics.png"
    _plot_outcome_training(history, output=outcome_plot)
    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_dir / "sampling_counts.png",
        suptitle="Learned outcome-propensity IPS vs ideal reward sampling",
    )
    trajectory_plot = run_dir / "trajectory_sampling.png"
    trajectories = _plot_trajectory_diagnostics(
        history,
        trainer,
        evaluation,
        samples=args.final_samples,
        output=trajectory_plot,
        subtitle=(
            "Trajectories are diagnostics only; the propensity model sees outcomes"
        ),
    )
    eval_rows = [row for row in history if "tv_reward_target" in row]
    summary: dict[str, Any] = {
        "environment": trainer.environment_summary(),
        "algorithm": {
            "name": "learned_outcome_ips",
            "raw_weight": "R(o) / p_safe_phi(o)",
            "forward_loss": "token_level_ppo",
            "outcome_update_order": "forward_then_outcome_same_batch",
            "outcome_uniform_mix": outcome_config.uniform_mix,
            "trajectory_model": None,
        },
        "outcome_config": asdict(outcome_config),
        "final_outcome_model_probabilities": {
            state.signature: float(probability)
            for state, probability in zip(
                trainer.terminals, trainer.outcome_probabilities()
            )
        },
        "final_sampling": sampling,
        "trajectory_sampling": trajectories,
        "best_eval": (
            min(eval_rows, key=lambda row: row["tv_reward_target"])
            if eval_rows
            else None
        ),
        "final_training_diagnostics": {
            key: history[-1][key]
            for key in (
                "ips_ess_fraction",
                "outcome_model_loss",
                "outcome_model_batch_tv",
                "outcome_model_entropy",
                "outcome_model_probability_mass",
            )
        },
        "plots": {
            "training_curves": training_plot.name,
            "outcome_model_diagnostics": outcome_plot.name,
            "sampling_counts": "sampling_counts.png",
            "trajectory_sampling": trajectory_plot.name,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Final ideal-line R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Final TV distance: {sampling['tv_reward_target']:.4f}")
    print(
        "Final outcome diagnostics: "
        f"loss={history[-1]['outcome_model_loss']:.3f}, "
        f"batch_TV={history[-1]['outcome_model_batch_tv']:.3f}, "
        f"IPS_ESS={history[-1]['ips_ess_fraction']:.3f}"
    )
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
