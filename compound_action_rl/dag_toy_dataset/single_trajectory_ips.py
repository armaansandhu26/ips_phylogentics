"""Single-trajectory IPS with a learned backward policy.

Count IPS estimates an outcome propensity from repeats in the current group.
This module instead estimates its inverse from each sampled trajectory:

    inv_p_hat(x; tau) = Q_B(tau | x) / P_F(tau)
    score(tau)         = R(x) * inv_p_hat(x; tau)

``P_F(tau)`` is the exact probability of the sampled path under the frozen
rollout policy. ``Q_B(tau | x)`` is the probability assigned to the same path
by a learned, terminal-conditioned policy that walks from ``x`` to the root.

For any normalized backward policy, not only a perfectly trained one,

    E_{tau ~ P_F(. | x)}[Q_B(tau | x) / P_F(tau)] = 1 / P_F(x).

The backward policy therefore controls estimator variance, not correctness.
It is fitted by maximum likelihood after the forward-policy update, so the
proposal used to score a group is frozen before seeing that group's paths.

The forward update deliberately inherits the token-level PPO loss from
``count_ips.py``. This isolates the new single-trajectory propensity estimator
from unrelated optimizer changes.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import TrainConfig
from count_ips import CountIPSTrainer, Episode
from dag_env import RIGHT, UP, State
from run_count_ips import _plot_final_counts


CHECKPOINT_VERSION = 2


def _run_dir_from_checkpoint_path(path: Path) -> Path:
    return path.parent.parent if path.parent.name == "checkpoints" else path.parent


def _load_checkpoint_history(path: Path, update_step: int) -> list[dict]:
    """Load and de-duplicate consolidated and chunked history through a step."""
    run_dir = _run_dir_from_checkpoint_path(path)
    rows_by_step: dict[int, dict] = {}
    consolidated = run_dir / "history.json"
    sources = [consolidated] if consolidated.is_file() else []
    chunk_dir = run_dir / "history_chunks"
    if chunk_dir.is_dir():
        sources.extend(sorted(chunk_dir.glob("history_updates_*.json")))
    for source in sources:
        rows = json.loads(source.read_text(encoding="utf-8"))
        for row in rows:
            step = int(row.get("step", 0))
            if 0 < step <= update_step:
                rows_by_step[step] = row
    return [rows_by_step[step] for step in sorted(rows_by_step)]


@dataclass(frozen=True)
class BackwardPolicyConfig:
    """Architecture and optimizer settings for the reverse proposal."""

    hidden_size: int = 64
    num_layers: int = 2
    lr: float = 1e-3
    train_epochs: int = 4
    grad_clip_norm: float = 1.0

    def validate(self) -> None:
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("backward hidden_size and num_layers must be >= 1")
        if self.lr <= 0.0:
            raise ValueError("backward lr must be > 0")
        if self.train_epochs < 1:
            raise ValueError("backward train_epochs must be >= 1")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("backward grad_clip_norm must be > 0")


@dataclass(frozen=True)
class BackwardBatch:
    """Flattened reverse decisions from a batch of complete trajectories."""

    contexts: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    episode_indices: torch.Tensor
    num_episodes: int


def backward_context(
    child: State,
    terminal: State,
    *,
    budget: int,
) -> tuple[float, ...]:
    """Encode a reverse state together with its finished outcome."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if child.depth < 1 or child.depth > budget:
        raise ValueError("child must be a non-root state within the budget")
    if terminal.depth != budget:
        raise ValueError("terminal must lie on the terminal frontier")
    if child.x > terminal.x or child.y > terminal.y:
        raise ValueError("child must lie on a monotone path to terminal")

    scale = float(budget)
    return (
        child.x / scale,
        child.y / scale,
        terminal.x / scale,
        terminal.y / scale,
        (terminal.x - child.x) / scale,
        (terminal.y - child.y) / scale,
    )


def backward_action_mask(child: State, *, max_step: int) -> tuple[bool, ...]:
    """Return valid incoming ``(direction, length)`` edges for ``child``."""
    if max_step < 1:
        raise ValueError("max_step must be >= 1")
    if child.depth < 1:
        raise ValueError("the root has no backward action")
    return tuple(
        length <= coordinate
        for coordinate in (child.x, child.y)
        for length in range(1, max_step + 1)
    )


def backward_action_index(
    direction: int,
    length: int,
    *,
    max_step: int,
) -> int:
    """Flatten a forward edge into the corresponding backward action."""
    if direction not in (RIGHT, UP):
        raise ValueError(f"unknown direction {direction}")
    if length < 1 or length > max_step:
        raise ValueError(f"length must be in [1, {max_step}]")
    return direction * max_step + length - 1


class BackwardPolicy(nn.Module):
    """A terminal-conditioned categorical policy over valid parent edges."""

    context_dim = 6

    def __init__(self, max_step: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        if max_step < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("backward network dimensions must be >= 1")

        layers: list[nn.Module] = []
        width = self.context_dim
        for _ in range(num_layers):
            layers.extend((nn.Linear(width, hidden_size), nn.Tanh()))
            width = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, 2 * max_step)

        # Start with a normalized uniform proposal over valid parents. This is
        # useful immediately, before the first reverse-policy fitting step.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def dist(self, contexts: torch.Tensor, masks: torch.Tensor) -> Categorical:
        if contexts.ndim != 2 or contexts.shape[1] != self.context_dim:
            raise ValueError("backward contexts must have shape (edges, 6)")
        if masks.ndim != 2 or masks.shape[0] != contexts.shape[0]:
            raise ValueError("backward masks must have shape (edges, actions)")
        masks = masks.bool()
        if torch.any(~masks.any(dim=-1)):
            raise ValueError("every backward state must have a valid parent")
        logits = self.head(self.trunk(contexts))
        if logits.shape != masks.shape:
            raise ValueError("backward mask width must equal 2 * max_step")
        logits = torch.where(masks, logits, torch.full_like(logits, -1e9))
        return Categorical(logits=logits)


def single_trajectory_ips_advantages(
    rewards: Sequence[float],
    outcome_ids: Sequence[object],
    trajectory_ids: Sequence[object],
    forward_log_probabilities: Sequence[float],
    backward_log_probabilities: Sequence[float],
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return normalized ``R(x) * Q_B(tau|x) / P_F(tau)`` advantages.

    The calculation stays in log space until a max shift makes exponentiation
    safe. A common shift has no effect on z-scored advantages or ESS.
    Outcome and trajectory IDs are used only for diagnostics, never to compute
    the propensity or the advantage.
    """
    size = len(rewards)
    if (
        size == 0
        or len(outcome_ids) != size
        or len(trajectory_ids) != size
        or len(forward_log_probabilities) != size
        or len(backward_log_probabilities) != size
    ):
        raise ValueError("all single-trajectory IPS inputs must have equal size")

    reward_array = np.asarray(rewards, dtype=np.float64)
    log_p_f = np.asarray(forward_log_probabilities, dtype=np.float64)
    log_q_b = np.asarray(backward_log_probabilities, dtype=np.float64)
    if np.any(~np.isfinite(reward_array)) or np.any(reward_array <= 0.0):
        raise ValueError("rewards must be finite and strictly positive")
    if np.any(~np.isfinite(log_p_f)) or np.any(log_p_f > 1e-7):
        raise ValueError("forward log-probabilities must be finite and <= 0")
    if np.any(~np.isfinite(log_q_b)) or np.any(log_q_b > 1e-7):
        raise ValueError("backward log-probabilities must be finite and <= 0")

    log_inverse_propensity = log_q_b - log_p_f
    log_scores = np.log(reward_array) + log_inverse_propensity
    log_shift = float(log_scores.max())
    scores = np.exp(log_scores - log_shift)

    centered = scores - scores.mean()
    score_std = float(scores.std())
    advantages = centered if score_std < eps else centered / (score_std + eps)

    squared_sum = float(np.square(scores).sum())
    ess = float(scores.sum() ** 2 / max(squared_sum, eps))
    outcome_counts = Counter(outcome_ids)
    trajectory_counts = Counter(trajectory_ids)
    implied_propensity = np.exp(
        np.clip(-log_inverse_propensity, -745.0, 700.0)
    )

    return advantages, {
        # Compatibility keys used by CountIPSTrainer's training logger.
        "ips_prob_mean": float(implied_propensity.mean()),
        "ips_prob_min": float(implied_propensity.min()),
        "ips_prob_max": float(implied_propensity.max()),
        "ips_unique_outcomes": float(len(outcome_counts)),
        "ips_max_outcome_count": float(max(outcome_counts.values())),
        "ips_min_outcome_count": float(min(outcome_counts.values())),
        "ips_scaled_reward_mean": float(scores.mean()),
        "ips_scaled_reward_std": score_std,
        "ips_ess": ess,
        "ips_ess_fraction": ess / size,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        # Estimator-specific diagnostics.
        "ips_unique_trajectories": float(len(trajectory_counts)),
        "forward_log_probability_mean": float(log_p_f.mean()),
        "backward_log_probability_mean": float(log_q_b.mean()),
        "log_inverse_outcome_propensity_mean": float(
            log_inverse_propensity.mean()
        ),
        "log_inverse_outcome_propensity_std": float(
            log_inverse_propensity.std()
        ),
        "log_score_mean": float(log_scores.mean()),
        "log_score_min": float(log_scores.min()),
        "log_score_max": float(log_scores.max()),
        "log_score_shift": log_shift,
        "propensity_uses_counts": 0.0,
    }


class SingleTrajectoryIPSTrainer(CountIPSTrainer):
    """Count-IPS PPO with count-free, pathwise outcome propensity estimates."""

    probability_label = "p_f/q_b"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        backward_config: BackwardPolicyConfig | None = None,
    ) -> None:
        super().__init__(config, device=device)
        self.backward_config = backward_config or BackwardPolicyConfig(
            hidden_size=max(16, self.config.hidden_size // 2),
            num_layers=self.config.num_layers,
        )
        self.backward_config.validate()
        self.backward_policy = BackwardPolicy(
            self.config.max_step,
            self.backward_config.hidden_size,
            self.backward_config.num_layers,
        ).to(self.device)
        self.backward_optimizer = torch.optim.Adam(
            self.backward_policy.parameters(),
            lr=self.backward_config.lr,
        )

    def _backward_batch(self, episodes: list[Episode]) -> BackwardBatch:
        contexts: list[tuple[float, ...]] = []
        masks: list[tuple[bool, ...]] = []
        actions: list[int] = []
        episode_indices: list[int] = []

        for episode_index, episode in enumerate(episodes):
            child = State(0, 0)
            for direction, length in episode.trajectory:
                child = (
                    State(child.x + length, child.y)
                    if direction == RIGHT
                    else State(child.x, child.y + length)
                )
                contexts.append(
                    backward_context(
                        child,
                        episode.terminal,
                        budget=self.config.budget,
                    )
                )
                masks.append(
                    backward_action_mask(
                        child,
                        max_step=self.config.max_step,
                    )
                )
                actions.append(
                    backward_action_index(
                        direction,
                        length,
                        max_step=self.config.max_step,
                    )
                )
                episode_indices.append(episode_index)
            if child != episode.terminal:
                raise ValueError("episode trajectory does not reach its terminal")

        if not contexts:
            raise ValueError("episodes must contain at least one transition")
        return BackwardBatch(
            contexts=torch.tensor(
                contexts,
                dtype=torch.float32,
                device=self.device,
            ),
            masks=torch.tensor(masks, dtype=torch.bool, device=self.device),
            actions=torch.tensor(actions, dtype=torch.long, device=self.device),
            episode_indices=torch.tensor(
                episode_indices,
                dtype=torch.long,
                device=self.device,
            ),
            num_episodes=len(episodes),
        )

    def _backward_path_log_probabilities_tensor(
        self,
        batch: BackwardBatch,
    ) -> tuple[torch.Tensor, Categorical]:
        distribution = self.backward_policy.dist(batch.contexts, batch.masks)
        edge_log_probabilities = distribution.log_prob(batch.actions)
        path_log_probabilities = torch.zeros(
            batch.num_episodes,
            dtype=torch.float32,
            device=self.device,
        )
        path_log_probabilities.scatter_add_(
            0,
            batch.episode_indices,
            edge_log_probabilities,
        )
        return path_log_probabilities, distribution

    @torch.inference_mode()
    def backward_path_log_probabilities(
        self,
        episodes: list[Episode],
    ) -> np.ndarray:
        """Evaluate observed paths under the currently frozen proposal."""
        batch = self._backward_batch(episodes)
        log_probabilities, _ = self._backward_path_log_probabilities_tensor(batch)
        return log_probabilities.cpu().numpy().astype(np.float64)

    def _group_advantages(self, episodes: list[Episode]) -> float:
        forward_log_probabilities = [
            sum(step.log_prob_joint for step in episode.steps)
            for episode in episodes
        ]
        backward_log_probabilities = self.backward_path_log_probabilities(episodes)
        advantages, metrics = single_trajectory_ips_advantages(
            [episode.reward for episode in episodes],
            [episode.terminal for episode in episodes],
            [episode.trajectory for episode in episodes],
            forward_log_probabilities,
            backward_log_probabilities,
            eps=self.config.advantage_eps,
        )
        for episode, advantage in zip(episodes, advantages):
            for step in episode.steps:
                step.advantage = float(advantage)
        self._last_ips_metrics = metrics
        return metrics["ips_ess"]

    def _fit_backward_policy(self, episodes: list[Episode]) -> dict[str, float]:
        """Fit ``Q_B`` to forward paths for use on the *next* rollout group."""
        batch = self._backward_batch(episodes)
        grad_norm_total = 0.0
        for _ in range(self.backward_config.train_epochs):
            self.backward_optimizer.zero_grad(set_to_none=True)
            path_log_probabilities, _ = (
                self._backward_path_log_probabilities_tensor(batch)
            )
            loss = -path_log_probabilities.mean()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.backward_policy.parameters(),
                self.backward_config.grad_clip_norm,
            )
            self.backward_optimizer.step()
            grad_norm_total += float(grad_norm.item())

        with torch.inference_mode():
            path_log_probabilities, distribution = (
                self._backward_path_log_probabilities_tensor(batch)
            )
            predictions = distribution.logits.argmax(dim=-1)
            return {
                "backward_loss": float(-path_log_probabilities.mean().item()),
                "backward_edge_accuracy": float(
                    (predictions == batch.actions).float().mean().item()
                ),
                "backward_edge_entropy": float(
                    distribution.entropy().mean().item()
                ),
                "backward_grad_norm": (
                    grad_norm_total / self.backward_config.train_epochs
                ),
            }

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        # The advantages were computed before this method, with Q_B frozen.
        # Updating Q_B after the forward step prevents this batch's fit from
        # leaking into the importance proposal that scored the same batch.
        forward_metrics = super().update(episodes)
        backward_metrics = self._fit_backward_policy(episodes)
        return {**forward_metrics, **backward_metrics}

    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if update_step != self.completed_updates:
            raise ValueError(
                "checkpoint update_step must equal the trainer's completed "
                f"updates ({update_step} != {self.completed_updates})"
            )
        # Persist only the new history rows since the previous checkpoint.
        # Chunking keeps total disk and write I/O linear for long runs. Write
        # the chunk first: if interrupted before the checkpoint replacement,
        # load() filters any ahead-of-checkpoint rows back to update_step.
        persisted_rows = getattr(self, "_persisted_history_rows", 0)
        pending_history = self.training_history[persisted_rows:]
        if pending_history:
            history_dir = _run_dir_from_checkpoint_path(path) / "history_chunks"
            history_dir.mkdir(parents=True, exist_ok=True)
            first_step = int(pending_history[0]["step"])
            last_step = int(pending_history[-1]["step"])
            _atomic_write_json(
                history_dir
                / (
                    f"history_updates_{first_step:06d}_"
                    f"{last_step:06d}.json"
                ),
                pending_history,
            )
            self._persisted_history_rows = len(self.training_history)
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "resumable": True,
            "config": self.config,
            "direction_policy": self.direction_policy.state_dict(),
            "step_policy": self.step_policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "backward_policy": self.backward_policy.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "backward_config": asdict(self.backward_config),
            "update_step": update_step,
            "cumulative_rollouts": self._cumulative_rollouts,
            "history_rows": len(self.training_history),
            "seen_terminals": [
                (state.x, state.y) for state in sorted(self._seen_terminals)
            ],
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            "algorithm": {
                "name": "single_trajectory_ips",
                "inverse_propensity": "Q_B(tau|x) / P_F(tau)",
                "score": "R(x) * Q_B(tau|x) / P_F(tau)",
                "forward_loss": "count_ips_token_level_ppo",
                "backward_update_order": "after_forward_update",
            },
        }
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: str = "cpu",
    ) -> "SingleTrajectoryIPSTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("algorithm", {}).get("name") != "single_trajectory_ips":
            raise ValueError("checkpoint is not a single-trajectory IPS run")
        trainer = cls(
            payload["config"],
            device=device,
            backward_config=BackwardPolicyConfig(**payload["backward_config"]),
        )
        trainer.direction_policy.load_state_dict(payload["direction_policy"])
        trainer.step_policy.load_state_dict(payload["step_policy"])
        trainer.backward_policy.load_state_dict(payload["backward_policy"])
        if "optimizer" in payload:
            trainer.optimizer.load_state_dict(payload["optimizer"])
        if "backward_optimizer" in payload:
            trainer.backward_optimizer.load_state_dict(
                payload["backward_optimizer"]
            )
        trainer._completed_updates = int(payload.get("update_step", 0))
        trainer._training_history = _load_checkpoint_history(
            Path(path),
            trainer._completed_updates,
        )
        # A checkpoint remains sufficient to resume model/optimizer/RNG state
        # if copied without its optional diagnostic history chunks.
        trainer._persisted_history_rows = len(trainer._training_history)
        trainer._cumulative_rollouts = int(
            payload.get(
                "cumulative_rollouts",
                (
                    trainer._training_history[-1]["cumulative_rollouts"]
                    if trainer._training_history
                    else 0
                ),
            )
        )
        trainer._seen_terminals = {
            State(int(x), int(y))
            for x, y in payload.get("seen_terminals", [])
        }
        trainer._checkpoint_resumable = bool(
            payload.get("resumable", False)
            and payload.get("checkpoint_version") == CHECKPOINT_VERSION
            and "optimizer" in payload
            and "rng_state" in payload
        )
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch_cpu"])
            if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        trainer.direction_policy.eval()
        trainer.step_policy.eval()
        trainer.backward_policy.eval()
        return trainer

    @property
    def checkpoint_resumable(self) -> bool:
        """Whether the loaded checkpoint contains complete continuation state."""
        return getattr(self, "_checkpoint_resumable", True)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return requested


def _atomic_write_json(path: Path, payload: object) -> None:
    """Replace a JSON artifact only after its complete contents are on disk."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_resume_checkpoint(path: Path) -> Path:
    """Resolve a checkpoint file or select the newest checkpoint in a run."""
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"resume path does not exist: {path}")

    candidates = list(path.glob("checkpoint_update_*.pt"))
    candidates.extend(path.glob("checkpoints/checkpoint_update_*.pt"))
    if (path / "checkpoint.pt").is_file():
        candidates.append(path / "checkpoint.pt")
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found under: {path}")

    def update_step(checkpoint: Path) -> int:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        return int(payload.get("update_step", -1))

    return max(candidates, key=update_step)


def _run_dir_for_checkpoint(checkpoint: Path) -> Path:
    return _run_dir_from_checkpoint_path(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument(
        "--num-updates",
        type=int,
        default=None,
        help=(
            "total target update number; defaults to 2000 for a new run and "
            "to the checkpoint target when resuming"
        ),
    )
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--backward-lr", type=float, default=1e-3)
    parser.add_argument("--backward-hidden-size", type=int, default=64)
    parser.add_argument("--backward-num-layers", type=int, default=2)
    parser.add_argument("--backward-train-epochs", type=int, default=4)
    parser.add_argument("--backward-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--terminal-rewards", type=float, nargs="+", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--run-dir", type=Path, default=None)
    location.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "checkpoint file, checkpoints directory, or run directory; the "
            "newest complete checkpoint is selected automatically"
        ),
    )
    args = parser.parse_args()

    requested_device = _resolve_device(args.device)
    resumed_from: Path | None = None
    if args.resume is not None:
        resumed_from = _resolve_resume_checkpoint(args.resume.resolve())
        run_dir = _run_dir_for_checkpoint(resumed_from)
        trainer = SingleTrajectoryIPSTrainer.load(
            resumed_from,
            device=requested_device,
        )
        if not trainer.checkpoint_resumable:
            raise RuntimeError(
                "this is a legacy evaluation-only checkpoint and lacks the "
                "forward optimizer, RNG state, or complete history required "
                "for an exact resume; start a new run with the current code"
            )
        target_updates = (
            args.num_updates
            if args.num_updates is not None
            else trainer.config.num_updates
        )
        if target_updates < trainer.completed_updates:
            raise ValueError(
                "--num-updates is a total target and cannot be below the "
                f"checkpoint step ({target_updates} < {trainer.completed_updates})"
            )
        trainer.config = replace(
            trainer.config,
            num_updates=target_updates,
        )
    else:
        target_updates = args.num_updates if args.num_updates is not None else 2_000
        config = TrainConfig(
            budget=args.budget,
            max_step=args.max_step,
            terminal_rewards=(
                tuple(args.terminal_rewards)
                if args.terminal_rewards is not None
                else None
            ),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            group_size=args.group_size,
            num_groups=args.num_groups,
            num_updates=target_updates,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            clip_ratio=args.clip_ratio,
            seed=args.seed,
            log_every=args.log_every,
        )
        backward_config = BackwardPolicyConfig(
            hidden_size=args.backward_hidden_size,
            num_layers=args.backward_num_layers,
            lr=args.backward_lr,
            train_epochs=args.backward_train_epochs,
            grad_clip_norm=args.backward_grad_clip_norm,
        )
        run_dir = args.run_dir or (
            Path(__file__).resolve().parent
            / "data"
            / "single_trajectory_ips_runs"
            / (
                f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
                f"_gs{config.group_size}_seed{config.seed}"
            )
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        trainer = SingleTrajectoryIPSTrainer(
            config,
            device=requested_device,
            backward_config=backward_config,
        )

    trainer.direction_policy.train()
    trainer.step_policy.train()
    trainer.backward_policy.train()
    checkpoint_every = args.checkpoint_every or None
    print(f"Run directory: {run_dir}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        f"Updates: {trainer.completed_updates:,} -> "
        f"{trainer.config.num_updates:,}"
    )
    if resumed_from is not None:
        print(f"Resumed exactly from: {resumed_from}")
    print("Estimator: R(x) * Q_B(tau|x) / P_F(tau); no outcome counts")

    _atomic_write_json(
        run_dir / "config.json",
        {
            "train_config": asdict(trainer.config),
            "backward_config": asdict(trainer.backward_config),
            "device": str(trainer.device),
            "checkpoint_version": CHECKPOINT_VERSION,
            "resumed_from": str(resumed_from) if resumed_from is not None else None,
            "algorithm": {
                "name": "single_trajectory_ips",
                "inverse_propensity": "Q_B(tau|x) / P_F(tau)",
                "score": "R(x) * Q_B(tau|x) / P_F(tau)",
                "forward_loss": "count_ips_token_level_ppo",
                "backward_update_order": "after_forward_update",
            },
        },
    )
    history = trainer.train(
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=run_dir / "checkpoints",
    )
    _atomic_write_json(
        run_dir / "history.json",
        history,
    )
    trainer.save(
        run_dir / "checkpoint.pt",
        update_step=trainer.completed_updates,
    )

    trainer.direction_policy.eval()
    trainer.step_policy.eval()
    trainer.backward_policy.eval()
    evaluation = trainer.evaluate(args.final_samples)
    sampling_plot = run_dir / "sampling_counts.png"
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=sampling_plot,
        suptitle=(
            "Single-trajectory IPS sampling vs ideal reward sampling "
            f"({args.final_samples:,} samples)"
        ),
    )
    summary = {
        "environment": trainer.environment_summary(),
        "algorithm": "single_trajectory_ips",
        "estimator": "R(x) * Q_B(tau|x) / P_F(tau)",
        "propensity_uses_counts": False,
        "completed_updates": trainer.completed_updates,
        "resumed_from": str(resumed_from) if resumed_from is not None else None,
        "final_sampling": sampling,
        "final_evaluation": evaluation,
    }
    _atomic_write_json(
        run_dir / "summary.json",
        summary,
    )
    print(f"Final TV distance: {evaluation['tv_reward_target']:.4f}")
    print(f"{args.final_samples:,}-sample plot: {sampling_plot}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
