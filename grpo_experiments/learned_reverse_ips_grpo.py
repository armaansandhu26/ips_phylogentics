#!/usr/bin/env python3
"""Learned-reverse IPS-GRPO for the five-taxon phylogeny.

This is the phylogenetic counterpart of
``compound_action_rl/dag_toy_dataset/learned_reverse_ips.py``:

    weight(tau) = R(x) * q_phi(tau | x) / P_F(tau).

For five taxa there are only 180 valid sequences of tree-merge actions.  The
environment maps those sequences to 105 structural terminal IDs.  We therefore
use an exact tabular reverse proposal: one learned logit per structural history,
normalized only against the other histories reaching the same topology.

This remains a valid q(tau | x) when branch lengths are learned.  Conditional
on an exact terminal tree and a structural reverse history, its branch-length
actions are determined by the terminal tree; q only has to distribute mass
over the alternative structural histories.  It may share that distribution
between terminal trees with the same topology.  Thus

    sum_{tau -> x} q_phi(tau | x) = 1

for every terminal without relying on sampled outcome counts.

The forward advantages are computed with q frozen before the batch.  The
forward PPO update happens first; q is then fitted by maximum likelihood on
that batch for use by the next update.

Two explicit terminal targets are supported:

* ``likelihood``: the original PhyloGFN target R(x) = exp(log L(x));
* ``shifted_linear``: the og_code shift ablation
  R(x) = 3600 + log L(x) (more generally, the configured positive shifted
  score).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from grpo_experiments.core.on_policy_buffer import run_on_policy_grpo_step
from grpo_experiments.core.trainer import GRPOTrainer
from grpo_experiments.ips_grpo.config import (
    IPSExperimentConfig,
    build_arg_parser,
    config_from_args,
)
from src.gfn.outcome_ids import OutcomeIdCache
from grpo_experiments.metrics import OutcomeTracker
from grpo_experiments.resume import (
    load_epoch_summaries,
    load_generator_checkpoint,
    load_metrics_rows,
    make_training_state,
    prepare_resume,
    resolve_output_dir,
    restore_tracker,
    save_training_state,
)
from grpo_experiments.utils import (
    append_jsonl,
    apply_training_cpu_limits,
    build_output_dir,
    choose_device,
    generate_exploration_spec,
    get_generator_params,
    load_phylogfn_cfg,
    resolve_rollout_chunk_size,
    set_seed,
)
from src.env import build_env
from src.gfn.build import build_gfn
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.gfn.training_data_loader import TrainingDataLoader

from grpo_experiments.phylo_learned_reverse_policy import (
    PhyloLearnedReverseConfig,
    PhyloLearnedReversePolicy,
    build_reverse_batch,
    path_log_probabilities as mlp_path_log_probabilities,
    update_mlp_reverse_policy,
)


METHOD = "learned_reverse_ips_grpo"
_LOG_REWARD_EPS = 1e-8
RewardTarget = Literal["likelihood", "shifted_linear"]
ReversePolicyType = Literal["tabular", "mlp"]


@dataclass
class LearnedReverseExperimentConfig(IPSExperimentConfig):
    """Configuration for the exact-tabular learned reverse experiment."""

    only_train_tree_model: bool = True
    reward_target: RewardTarget = "likelihood"
    reverse_policy_type: ReversePolicyType = "tabular"
    reverse_lr: float = 1e-3
    reverse_train_epochs: int = 4
    reverse_grad_clip_norm: float = 1.0
    reverse_hidden_size: int = 128
    reverse_num_layers: int = 2
    advantage_normalization: str = "running"
    running_scale_decay: float = 0.99
    running_advantage_clip: float = 10.0
    running_log_ratio_clip: float = 20.0

    @property
    def method(self) -> str:
        return METHOD

    @classmethod
    def from_base(
        cls,
        base: IPSExperimentConfig,
        *,
        only_train_tree_model: bool,
        reward_target: RewardTarget,
        reverse_policy_type: ReversePolicyType,
        reverse_lr: float,
        reverse_train_epochs: int,
        reverse_grad_clip_norm: float,
        reverse_hidden_size: int,
        reverse_num_layers: int,
        advantage_normalization: str,
        running_scale_decay: float,
        running_advantage_clip: float,
        running_log_ratio_clip: float,
    ) -> "LearnedReverseExperimentConfig":
        kwargs = {field.name: getattr(base, field.name) for field in fields(IPSExperimentConfig)}
        # LearnedReverseExperimentConfig re-declares this field; avoid duplicate kwargs
        # when IPSExperimentConfig also carries only_train_tree_model (e.g. --full-model).
        kwargs.pop("only_train_tree_model", None)
        return cls(
            **kwargs,
            only_train_tree_model=only_train_tree_model,
            reward_target=reward_target,
            reverse_policy_type=reverse_policy_type,
            reverse_lr=reverse_lr,
            reverse_train_epochs=reverse_train_epochs,
            reverse_grad_clip_norm=reverse_grad_clip_norm,
            reverse_hidden_size=reverse_hidden_size,
            reverse_num_layers=reverse_num_layers,
            advantage_normalization=advantage_normalization,
            running_scale_decay=running_scale_decay,
            running_advantage_clip=running_advantage_clip,
            running_log_ratio_clip=running_log_ratio_clip,
        )


@dataclass
class RunningLogWeightNormalizer:
    """Detached EMA scale for log-space importance weights."""

    decay: float = 0.99
    advantage_clip: float = 10.0
    log_ratio_clip: float = 20.0
    log_first_moment: float | None = None
    log_second_moment: float | None = None
    updates: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("running scale decay must be in [0, 1)")
        if self.advantage_clip <= 0.0:
            raise ValueError("running advantage clip must be positive")
        if self.log_ratio_clip <= 0.0:
            raise ValueError("running log-ratio clip must be positive")

    @staticmethod
    def _log_mean_exp(values: np.ndarray) -> float:
        maximum = float(values.max())
        return maximum + float(np.log(np.exp(values - maximum).mean()))

    def normalize(self, log_weights: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        if log_weights.ndim != 1 or log_weights.size == 0:
            raise ValueError("log_weights must be a non-empty vector")
        if np.any(~np.isfinite(log_weights)):
            raise ValueError("log_weights must be finite")

        batch_log_first = self._log_mean_exp(log_weights)
        batch_log_second = self._log_mean_exp(2.0 * log_weights)
        bootstrap = self.log_first_moment is None or self.log_second_moment is None
        scale_log_first = batch_log_first if bootstrap else float(self.log_first_moment)
        scale_log_second = batch_log_second if bootstrap else float(self.log_second_moment)
        log_rms = 0.5 * scale_log_second

        stabilized = np.exp(
            np.clip(log_weights - log_rms, -self.log_ratio_clip, self.log_ratio_clip)
        )
        baseline = float(
            np.exp(
                np.clip(
                    scale_log_first - log_rms,
                    -self.log_ratio_clip,
                    self.log_ratio_clip,
                )
            )
        )
        centered = stabilized - baseline
        advantages = np.clip(centered, -self.advantage_clip, self.advantage_clip)

        if bootstrap:
            self.log_first_moment = batch_log_first
            self.log_second_moment = batch_log_second
        else:
            log_decay = math.log(self.decay) if self.decay > 0.0 else -math.inf
            log_new = math.log1p(-self.decay)
            self.log_first_moment = float(
                np.logaddexp(
                    log_decay + float(self.log_first_moment),
                    log_new + batch_log_first,
                )
            )
            self.log_second_moment = float(
                np.logaddexp(
                    log_decay + float(self.log_second_moment),
                    log_new + batch_log_second,
                )
            )
        self.updates += 1
        return advantages, {
            "running_scale_bootstrap": float(bootstrap),
            "running_scale_updates": float(self.updates),
            "running_log_weight_rms": float(log_rms),
            "running_scaled_weight_baseline": baseline,
            "running_scaled_weight_mean": float(stabilized.mean()),
            "running_scaled_weight_std": float(stabilized.std()),
            "running_preclip_advantage_min": float(centered.min()),
            "running_preclip_advantage_max": float(centered.max()),
            "running_advantage_clip_fraction": float(np.mean(advantages != centered)),
        }

    def state_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


class TabularTerminalReversePolicy(nn.Module):
    """Exactly normalized q(structural history | terminal topology)."""

    def __init__(
        self,
        trajectories: Sequence[tuple[int, ...]],
        terminal_ids: Sequence[str],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if not trajectories or len(trajectories) != len(terminal_ids):
            raise ValueError("trajectory catalog and terminal IDs must be non-empty and aligned")
        if len(set(trajectories)) != len(trajectories):
            raise ValueError("trajectory catalog contains duplicates")

        self.trajectories = tuple(tuple(int(action) for action in path) for path in trajectories)
        self.terminal_ids = tuple(str(item) for item in terminal_ids)
        self.trajectory_to_index = {
            trajectory: index for index, trajectory in enumerate(self.trajectories)
        }

        grouped: dict[str, list[int]] = defaultdict(list)
        for index, terminal_id in enumerate(self.terminal_ids):
            grouped[terminal_id].append(index)
        self.group_ids = tuple(sorted(grouped))
        self.group_indices = tuple(
            torch.tensor(grouped[group_id], dtype=torch.long, device=device)
            for group_id in self.group_ids
        )
        group_lookup = {group_id: group for group, group_id in enumerate(self.group_ids)}
        self.register_buffer(
            "trajectory_groups",
            torch.tensor(
                [group_lookup[terminal_id] for terminal_id in self.terminal_ids],
                dtype=torch.long,
                device=device,
            ),
        )
        self.logits = nn.Parameter(torch.zeros(len(self.trajectories), device=device))

    def catalog_indices(self, action_paths: Sequence[tuple[int, ...]]) -> torch.Tensor:
        try:
            indices = [self.trajectory_to_index[tuple(path)] for path in action_paths]
        except KeyError as exc:
            raise ValueError(f"rollout produced an unknown tree-action trajectory: {exc.args[0]}") from exc
        return torch.tensor(indices, dtype=torch.long, device=self.logits.device)

    def all_log_probabilities(self) -> torch.Tensor:
        output = torch.empty_like(self.logits)
        for indices in self.group_indices:
            output[indices] = self.logits[indices] - torch.logsumexp(
                self.logits[indices], dim=0
            )
        return output

    def log_prob(self, catalog_indices: torch.Tensor) -> torch.Tensor:
        return self.all_log_probabilities()[catalog_indices]

    def entropy_for_indices(self, catalog_indices: torch.Tensor) -> torch.Tensor:
        log_q = self.all_log_probabilities()
        entropies = torch.empty(len(self.group_indices), device=self.logits.device)
        for group, indices in enumerate(self.group_indices):
            probabilities = log_q[indices].exp()
            entropies[group] = -(probabilities * log_q[indices]).sum()
        return entropies[self.trajectory_groups[catalog_indices]]

    @torch.no_grad()
    def normalization_error(self) -> float:
        log_q = self.all_log_probabilities()
        errors = [
            abs(float(log_q[indices].exp().sum().item()) - 1.0)
            for indices in self.group_indices
        ]
        return max(errors)


def _edge_action(step: int, num_steps: int) -> int | list[int]:
    return 0 if step == num_steps - 1 else [0, 0]


def enumerate_tree_action_catalog(
    env,
) -> tuple[list[tuple[int, ...]], list[str]]:
    """Enumerate every merge-action path and its structural terminal ID.

    Edge action zero is only a convenient way to instantiate each structural
    path.  The returned topology grouping is independent of the sampled edge
    actions used during training.
    """
    num_taxa = len(env.sequences)
    if num_taxa != 5:
        raise ValueError(
            f"this exact tabular experiment requires 5 taxa, found {num_taxa}"
        )
    action_ranges = [
        range(num_trees * (num_trees - 1) // 2)
        for num_trees in range(num_taxa, 1, -1)
    ]
    trajectories: list[tuple[int, ...]] = []
    terminal_ids: list[str] = []
    for action_path in itertools.product(*action_ranges):
        actions = [
            {
                "tree_action": int(tree_action),
                "edge_action": _edge_action(step, len(action_path)),
            }
            for step, tree_action in enumerate(action_path)
        ]
        trajectory = env.actions_to_trajectory(actions)
        tree = trajectory.current_state.subtrees[0]
        trajectories.append(tuple(int(action) for action in action_path))
        terminal_ids.append(str(tree.ete_node.get_topology_id()))
    return trajectories, terminal_ids


def rollout_tree_action_paths(batch: dict[str, Any]) -> list[tuple[int, ...]]:
    action_tensors = batch.get("action_tensors")
    if action_tensors is None:
        raise ValueError("rollout batch is missing action_tensors")
    tree_actions = torch.stack(action_tensors.tree_actions, dim=1)
    return [tuple(int(value) for value in row) for row in tree_actions.cpu().tolist()]


def trajectory_indices_from_paths(
    action_paths: Sequence[tuple[int, ...]],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    lookup: dict[tuple[int, ...], int] = {}
    indices: list[int] = []
    for path in action_paths:
        if path not in lookup:
            lookup[path] = len(lookup)
        indices.append(lookup[path])
    return torch.tensor(indices, dtype=torch.long, device=device)


def terminal_log_rewards_from_scores(
    log_scores: torch.Tensor,
    *,
    reward_target: RewardTarget,
    reward_c: float,
    reward_scale: float,
) -> torch.Tensor:
    """Return log R for the selected phylogenetic target.

    ``likelihood`` preserves the original PhyloGFN exponential target:
    log R = (C + log L) / scale.  ``shifted_linear`` treats that positive
    shifted score as R itself, matching the og_code shift-3600 ablation:
    log R = log((C + shifted log L) / scale).
    """
    if reward_scale == 0.0:
        raise ValueError("reward_scale must be non-zero")
    linear_score = (reward_c + log_scores.detach()) / reward_scale
    if reward_target == "likelihood":
        return linear_score
    if reward_target == "shifted_linear":
        positive = linear_score.clamp(min=_LOG_REWARD_EPS)
        if bool(torch.any(linear_score <= 0.0)):
            n_bad = int(torch.sum(linear_score <= 0.0).item())
            print(
                f"warning: clamping {n_bad}/{linear_score.numel()} "
                "shifted_linear scores to keep log R finite",
                flush=True,
            )
        return positive.log()
    raise ValueError(f"unknown reward target: {reward_target!r}")


def learned_reverse_advantages(
    log_scores: torch.Tensor,
    forward_log_probabilities: torch.Tensor,
    reverse_log_probabilities: torch.Tensor,
    *,
    reward_target: RewardTarget,
    reward_c: float,
    reward_scale: float,
    normalizer: RunningLogWeightNormalizer | None,
    advantage_eps: float,
    terminal_ids: Sequence[str],
    trajectory_indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages from log R + log q_phi - log P_F."""
    if (
        log_scores.ndim != 1
        or forward_log_probabilities.shape != log_scores.shape
        or reverse_log_probabilities.shape != log_scores.shape
    ):
        raise ValueError("scores and path log-probabilities must all have shape (B,)")

    log_rewards = terminal_log_rewards_from_scores(
        log_scores,
        reward_target=reward_target,
        reward_c=reward_c,
        reward_scale=reward_scale,
    )
    log_weights = log_rewards + reverse_log_probabilities.detach() - forward_log_probabilities.detach()
    log_weights_np = log_weights.double().cpu().numpy()

    maximum = float(log_weights_np.max())
    scaled = np.exp(log_weights_np - maximum)
    if normalizer is None:
        centered = scaled - scaled.mean()
        std = float(scaled.std())
        advantages_np = centered if std < advantage_eps else centered / (std + advantage_eps)
        normalization_metrics: dict[str, float] = {}
    else:
        advantages_np, normalization_metrics = normalizer.normalize(log_weights_np)

    squared_sum = float(np.square(scaled).sum())
    ess = float(scaled.sum() ** 2 / max(squared_sum, advantage_eps))
    implied_log_probability = (
        forward_log_probabilities.detach() - reverse_log_probabilities.detach()
    ).double().cpu().numpy()
    by_terminal: dict[str, list[float]] = defaultdict(list)
    for terminal_id, value in zip(terminal_ids, implied_log_probability):
        by_terminal[str(terminal_id)].append(float(value))
    within_terminal_stds = [
        float(np.std(values)) for values in by_terminal.values() if len(values) > 1
    ]

    metrics = {
        "ips_ess": ess,
        "ips_ess_fraction": ess / max(len(scaled), 1),
        "ips_unique_outcomes": float(len(set(terminal_ids))),
        "ips_unique_trajectories": float(trajectory_indices.unique().numel()),
        "forward_log_probability_mean": float(forward_log_probabilities.mean().item()),
        "reverse_log_probability_mean": float(reverse_log_probabilities.mean().item()),
        "implied_terminal_log_probability_mean": float(implied_log_probability.mean()),
        "implied_terminal_log_probability_std": float(implied_log_probability.std()),
        "implied_terminal_within_outcome_std": float(
            np.mean(within_terminal_stds) if within_terminal_stds else 0.0
        ),
        "log_importance_weight_mean": float(log_weights_np.mean()),
        "log_importance_weight_min": float(log_weights_np.min()),
        "log_importance_weight_max": float(log_weights_np.max()),
        "target_log_reward_mean": float(log_rewards.mean().item()),
        "target_log_reward_min": float(log_rewards.min().item()),
        "target_log_reward_max": float(log_rewards.max().item()),
        "advantage_mean": float(advantages_np.mean()),
        "advantage_std": float(advantages_np.std()),
        "advantage_min": float(advantages_np.min()),
        "advantage_max": float(advantages_np.max()),
        "advantage_normalization_is_running": float(normalizer is not None),
    }
    metrics.update(normalization_metrics)
    advantages = torch.tensor(
        advantages_np,
        dtype=log_scores.dtype,
        device=log_scores.device,
    )
    return advantages, metrics


def update_reverse_policy(
    policy: TabularTerminalReversePolicy,
    optimizer: torch.optim.Optimizer,
    trajectory_indices: torch.Tensor,
    *,
    train_epochs: int,
    grad_clip_norm: float,
) -> dict[str, float]:
    """MLE update performed only after the forward-policy update."""
    grad_norm_total = 0.0
    for _ in range(train_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = -policy.log_prob(trajectory_indices).mean()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
        grad_norm_total += float(grad_norm.item())

    with torch.inference_mode():
        log_probabilities = policy.log_prob(trajectory_indices)
        entropy = policy.entropy_for_indices(trajectory_indices)
        return {
            "reverse_loss": float(-log_probabilities.mean().item()),
            "reverse_path_probability_mean": float(log_probabilities.exp().mean().item()),
            "reverse_path_entropy": float(entropy.mean().item()),
            "reverse_grad_norm": grad_norm_total / train_epochs,
            "reverse_param_norm": float(policy.logits.norm().item()),
            "reverse_normalization_error": policy.normalization_error(),
        }


def _paired_learned_reverse_state(checkpoint_path: Path) -> Path:
    root = checkpoint_path.parent
    name = checkpoint_path.name
    if name == "final_checkpoint.pt":
        path = root / "learned_reverse_state.pt"
    elif name.startswith("checkpoint_epoch") and name.endswith(".pt"):
        path = root / f"learned_reverse_epoch{name[len('checkpoint_epoch'):]}"
    else:
        raise ValueError(f"cannot infer learned-reverse state for checkpoint {name}")
    if not path.exists():
        raise FileNotFoundError(f"missing paired learned-reverse state: {path}")
    return path


def _restore_running_normalizer(
    normalizer: RunningLogWeightNormalizer,
    state: dict[str, float | int | None],
) -> None:
    normalizer.log_first_moment = state["log_first_moment"]
    normalizer.log_second_moment = state["log_second_moment"]
    normalizer.updates = int(state["updates"])


def _load_learned_reverse_state(
    path: Path,
    *,
    policy: Union[TabularTerminalReversePolicy, PhyloLearnedReversePolicy],
    reverse_optimizer: torch.optim.Optimizer,
    forward_trainer: GRPOTrainer,
    normalizer: RunningLogWeightNormalizer | None,
    device: str,
) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("algorithm") != METHOD:
        raise ValueError(
            f"unexpected learned-reverse state algorithm: {state.get('algorithm')!r}"
        )
    policy.load_state_dict(state["reverse_policy"])
    reverse_optimizer.load_state_dict(state["reverse_optimizer"])
    forward_trainer.load_state_dict(state["forward_trainer"])
    if normalizer is not None and state.get("running_normalizer") is not None:
        _restore_running_normalizer(normalizer, state["running_normalizer"])
    return state


def _save_learned_reverse_state(
    path: Path,
    *,
    policy: Union[TabularTerminalReversePolicy, PhyloLearnedReversePolicy],
    optimizer: torch.optim.Optimizer,
    forward_trainer: GRPOTrainer,
    normalizer: RunningLogWeightNormalizer | None,
    update_step: int,
    reverse_policy_type: ReversePolicyType,
    reverse_config: PhyloLearnedReverseConfig | None = None,
) -> None:
    payload: dict[str, Any] = {
        "algorithm": METHOD,
        "reverse_policy_type": reverse_policy_type,
        "reverse_policy": policy.state_dict(),
        "reverse_optimizer": optimizer.state_dict(),
        "forward_trainer": forward_trainer.state_dict(),
        "running_normalizer": (
            normalizer.state_dict() if normalizer is not None else None
        ),
        "update_step": update_step,
    }
    if isinstance(policy, TabularTerminalReversePolicy):
        payload["trajectories"] = policy.trajectories
        payload["terminal_ids"] = policy.terminal_ids
    if reverse_config is not None:
        payload["reverse_config"] = asdict(reverse_config)
    torch.save(payload, path)


def _write_catalog(
    output_dir: Path,
    trajectories: Sequence[tuple[int, ...]] | None,
    terminal_ids: Sequence[str] | None,
    *,
    outcome_level: str,
    reverse_policy_type: ReversePolicyType,
) -> None:
    payload: dict[str, Any] = {
        "reverse_policy_type": reverse_policy_type,
        "reverse_conditioning_level": "topology",
        "reported_outcome_level": outcome_level,
    }
    if trajectories is None or terminal_ids is None:
        payload["num_trajectories"] = None
        payload["num_structural_outcomes"] = None
        payload["entries"] = []
    else:
        multiplicities = Counter(terminal_ids)
        payload.update(
            {
                "num_trajectories": len(trajectories),
                "num_structural_outcomes": len(multiplicities),
                "multiplicity_histogram": {
                    str(multiplicity): count
                    for multiplicity, count in sorted(
                        Counter(multiplicities.values()).items()
                    )
                },
                "entries": [
                    {"trajectory": list(trajectory), "terminal_id": terminal_id}
                    for trajectory, terminal_id in zip(trajectories, terminal_ids)
                ],
            }
        )
    (output_dir / "reverse_catalog.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _save_best_trees(output_dir: Path, data_loader: TrainingDataLoader) -> None:
    if data_loader.best_state_batch_size <= 0:
        return
    path = output_dir / "best_trees.pt"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(data_loader.best_trees, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _replay_batch_metrics(
    batch: dict[str, torch.Tensor],
    replay_tree_count: int,
) -> dict[str, float]:
    if replay_tree_count <= 0:
        return {}
    log_scores = batch["log_scores"]
    if replay_tree_count >= log_scores.numel():
        return {
            "replay_tree_fraction": 1.0,
            "mean_log_score_replay": float(log_scores.mean().item()),
        }
    return {
        "replay_tree_fraction": replay_tree_count / log_scores.numel(),
        "mean_log_score_replay": float(log_scores[:replay_tree_count].mean().item()),
        "mean_log_score_on_policy": float(log_scores[replay_tree_count:].mean().item()),
    }


def validate_config(config: LearnedReverseExperimentConfig) -> None:
    if config.replay_batch_size > 0:
        config.disable_replay = False
    if config.enable_policy_is:
        raise ValueError("learned reverse IPS currently supports on-policy training only")
    if config.disable_replay and config.replay_batch_size > 0:
        raise ValueError("replay batch size is > 0 but replay remains disabled")
    if config.effective_replay_batch_size > 0 and config.replay_buffer_size < 1:
        raise ValueError("replay buffer size must be >= 1 when replay is enabled")
    if config.grpo_group_size < 1:
        raise ValueError("on-policy plus replay batch sizes must sum to at least 1")
    if config.grpo_num_iterations != 1:
        raise ValueError("learned reverse update ordering requires --grpo-num-iterations 1")
    if config.policy_loss_mode != "ppo":
        raise ValueError("learned reverse experiment currently requires --policy-loss-mode ppo")
    if config.reverse_lr <= 0.0:
        raise ValueError("reverse learning rate must be positive")
    if config.reverse_train_epochs < 1:
        raise ValueError("reverse train epochs must be at least 1")
    if config.reverse_grad_clip_norm <= 0.0:
        raise ValueError("reverse gradient clipping norm must be positive")
    if config.advantage_normalization not in {"batch", "running"}:
        raise ValueError("advantage normalization must be batch or running")
    if config.reverse_policy_type not in {"tabular", "mlp"}:
        raise ValueError("reverse policy type must be tabular or mlp")
    if config.reverse_hidden_size < 1 or config.reverse_num_layers < 1:
        raise ValueError("reverse MLP dimensions must be >= 1")
    if config.reward_target not in {"likelihood", "shifted_linear"}:
        raise ValueError("reward target must be likelihood or shifted_linear")

def run_experiment(config: LearnedReverseExperimentConfig) -> str:
    validate_config(config)
    set_seed(config.seed)
    device = choose_device(config.device)
    cfg, all_seqs = load_phylogfn_cfg(config)
    apply_training_cpu_limits(config, cfg)
    # Enumerate on CPU before moving the environment and model to the training device.
    env = build_env(cfg, all_seqs)
    num_taxa = len(env.sequences)
    catalog_trajectories: list[tuple[int, ...]] | None = None
    catalog_terminal_ids: list[str] | None = None
    reverse_config: PhyloLearnedReverseConfig | None = None

    if config.reverse_policy_type == "tabular":
        catalog_trajectories, catalog_terminal_ids = enumerate_tree_action_catalog(env)
        policy: Union[TabularTerminalReversePolicy, PhyloLearnedReversePolicy] = (
            TabularTerminalReversePolicy(
                catalog_trajectories, catalog_terminal_ids, device=device
            )
        )
        if len(catalog_trajectories) != 180 or len(policy.group_ids) != 105:
            raise RuntimeError(
                "unexpected five-taxon catalog: "
                f"{len(catalog_trajectories)} trajectories, "
                f"{len(policy.group_ids)} outcomes"
            )
        if policy.normalization_error() > 1e-6:
            raise RuntimeError("reverse proposal failed its initial normalization check")
    else:
        reverse_config = PhyloLearnedReverseConfig(
            hidden_size=config.reverse_hidden_size,
            num_layers=config.reverse_num_layers,
        )
        reverse_config.validate()
        policy = PhyloLearnedReversePolicy(
            num_taxa,
            hidden_size=reverse_config.hidden_size,
            num_layers=reverse_config.num_layers,
        ).to(device)

    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    rollout_worker = RolloutWorker(env)

    output_dir = Path(
        resolve_output_dir(config)
        if config.resume_from
        else build_output_dir(config.output_root, config.method, config.run_name)
    )
    cfg.OUTPUT_PATH = str(output_dir)
    resume = None
    if config.resume_from:
        resume, checkpoint_path = prepare_resume(
            output_dir,
            checkpoint_name=config.resume_checkpoint,
            training_mode="on_policy",
            steps_per_epoch=config.steps_per_epoch,
            update_cycles=1,
            target_epochs=config.epochs,
            target_resample_rounds=0,
        )
    config.save_json(str(output_dir / "experiment_config.json"))
    (output_dir / "resolved_config.yaml").write_text(cfg.dump(), encoding="utf-8")
    _write_catalog(
        output_dir,
        catalog_trajectories,
        catalog_terminal_ids,
        outcome_level=config.outcome_level,
        reverse_policy_type=config.reverse_policy_type,
    )

    data_loader = TrainingDataLoader(
        cfg,
        env,
        rollout_worker,
        str(output_dir / "best_trees.pt"),
    )
    outcome_id_cache = OutcomeIdCache(env)
    params = get_generator_params(generator)
    forward_trainer = GRPOTrainer(
        params=params,
        lr=config.grpo_lr,
        max_grad_norm=config.grpo_max_grad_norm,
        advantage_eps=config.grpo_advantage_eps,
        clip_eps=config.grpo_clip_eps,
        clip_eps_high=config.grpo_clip_eps_high,
        entropy_coef=config.grpo_entropy_coef,
        num_iterations=1,
        reward_c=cfg.ENV.REWARD.C,
        reward_scale=cfg.ENV.REWARD.SCALE,
        policy_loss_mode="ppo",
    )
    reverse_optimizer = torch.optim.Adam(policy.parameters(), lr=config.reverse_lr)
    normalizer = (
        RunningLogWeightNormalizer(
            decay=config.running_scale_decay,
            advantage_clip=config.running_advantage_clip,
            log_ratio_clip=config.running_log_ratio_clip,
        )
        if config.advantage_normalization == "running"
        else None
    )

    if resume is not None:
        load_generator_checkpoint(generator, resume.checkpoint_path)
        reverse_state_path = _paired_learned_reverse_state(Path(resume.checkpoint_path))
        _load_learned_reverse_state(
            reverse_state_path,
            policy=policy,
            reverse_optimizer=reverse_optimizer,
            forward_trainer=forward_trainer,
            normalizer=normalizer,
            device=device,
        )
        print(f"restored learned-reverse state from {reverse_state_path.name}")

    print(f"run_dir={output_dir}")
    structural_outcomes = (
        len(policy.group_ids)
        if isinstance(policy, TabularTerminalReversePolicy)
        else "n/a"
    )
    replay_enabled = config.effective_replay_batch_size > 0
    print(
        f"method={config.method} device={device} "
        f"reverse_policy_type={config.reverse_policy_type} "
        f"trajectories={len(catalog_trajectories) if catalog_trajectories else 'learned-per-step'} "
        f"structural_outcomes={structural_outcomes} "
        f"outcome_level={config.outcome_level} "
        f"reward_target={config.reward_target} "
        f"fixed_edges={bool(cfg.GFN.MODEL.ONLY_TRAIN_TREE_MODEL)} "
        f"G_on={config.on_policy_batch_size} "
        f"G_replay={config.effective_replay_batch_size} "
        f"G_total={config.grpo_group_size} "
        f"replay_buffer={config.replay_buffer_size if replay_enabled else 0}"
    )
    print("weight=R(x)*q_phi(tau|x)/P_F(tau); reverse update happens after PPO")
    if replay_enabled:
        print(
            "best-tree replay enabled: first "
            f"{data_loader.best_state_batch_size} trees per micro-batch are replayed "
            f"from {len(data_loader.best_trees)} buffered trees"
        )

    metrics_path = str(output_dir / "metrics.jsonl")
    metrics_rows = load_metrics_rows(metrics_path) if resume is not None else []
    epoch_summaries: list[dict[str, float | int | str]] = (
        load_epoch_summaries(output_dir) if resume is not None else []
    )
    global_step = resume.global_step if resume is not None else 0
    start_epoch = resume.start_epoch if resume is not None else 0
    start_step = resume.start_step if resume is not None else 0
    generation_state = None
    tracker = (
        restore_tracker(resume, metrics_rows) if resume is not None else OutcomeTracker()
    )
    cumulative_outcomes: Counter[str] = Counter(tracker.outcome_counts)
    rollout_chunk = resolve_rollout_chunk_size(config)
    log_score_shift = float(getattr(cfg.ENV, "LOG_SCORE_SHIFT", 0.0))

    for epoch in range(start_epoch, cfg.GFN.TRAINING_DATA_LOADER.EPOCHS_NUM):
        exploration = generate_exploration_spec(
            cfg.GFN.TRAINING_DATA_LOADER.EXPLORATION, epoch
        )
        epoch_losses: list[float] = []
        epoch_reverse_losses: list[float] = []
        epoch_ess: list[float] = []

        step_begin = start_step if epoch == start_epoch else 0
        for step in range(step_begin, data_loader.steps_per_epoch):
            random_spec = data_loader.generate_random_spec(exploration, step)
            batch, rollout_trajectories = data_loader.generate_batch(
                generator, random_spec
            )
            action_paths = rollout_tree_action_paths(batch)
            batch_outcome_ids, batch_topology_ids = outcome_id_cache.ids_from_rollout_batch(
                batch,
                rollout_trajectories,
                level=config.outcome_level,
            )
            cumulative_outcomes.update(batch_outcome_ids)
            terminal_log_scores = batch["log_scores"].detach().cpu().tolist()

            if isinstance(policy, TabularTerminalReversePolicy):
                trajectory_indices = policy.catalog_indices(action_paths)
                with torch.inference_mode():
                    reverse_log_probabilities = policy.log_prob(trajectory_indices)
            else:
                trajectory_indices = trajectory_indices_from_paths(
                    action_paths, device=device
                )
                with torch.inference_mode():
                    reverse_log_probabilities = mlp_path_log_probabilities(
                        policy,
                        env,
                        action_paths,
                        terminal_ids=batch_topology_ids,
                        terminal_log_scores=terminal_log_scores,
                    )
            forward_log_probabilities = batch["log_paths_pf"].detach().sum(dim=-1)
            advantages, advantage_metrics = learned_reverse_advantages(
                batch["log_scores"],
                forward_log_probabilities,
                reverse_log_probabilities,
                reward_target=config.reward_target,
                reward_c=float(cfg.ENV.REWARD.C),
                reward_scale=float(cfg.ENV.REWARD.SCALE),
                normalizer=normalizer,
                advantage_eps=config.grpo_advantage_eps,
                terminal_ids=batch_outcome_ids,
                trajectory_indices=trajectory_indices,
            )

            forward_metrics, generation_state = run_on_policy_grpo_step(
                forward_trainer,
                rollout_worker,
                generator,
                batch,
                rollout_trajectories,
                random_spec=random_spec,
                generation_state=generation_state,
                chunk_size=rollout_chunk,
                device=device,
                extra_update_kwargs={
                    "fixed_advantages": advantages,
                    "fixed_advantage_metrics": advantage_metrics,
                },
                group_meta={
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "method": config.method,
                },
            )
            if isinstance(policy, TabularTerminalReversePolicy):
                reverse_metrics = update_reverse_policy(
                    policy,
                    reverse_optimizer,
                    trajectory_indices,
                    train_epochs=config.reverse_train_epochs,
                    grad_clip_norm=config.reverse_grad_clip_norm,
                )
            else:
                reverse_batch = build_reverse_batch(
                    env,
                    action_paths,
                    terminal_ids=batch_topology_ids,
                    terminal_log_scores=terminal_log_scores,
                    device=device,
                )
                reverse_metrics = update_mlp_reverse_policy(
                    policy,
                    reverse_optimizer,
                    reverse_batch,
                    train_epochs=config.reverse_train_epochs,
                    grad_clip_norm=config.reverse_grad_clip_norm,
                )
            replay_tree_count = data_loader.best_state_batch_size
            record = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "method": config.method,
                "training_mode": "on_policy+replay" if replay_enabled else "on_policy",
                "outcome_level": config.outcome_level,
                "reward_target": config.reward_target,
                "on_policy_batch_size": config.on_policy_batch_size,
                "replay_batch_size": config.effective_replay_batch_size,
                "grpo_group_size": config.grpo_group_size,
                "best_trees_buffer_size": len(data_loader.best_trees),
                "batch_unique_outcomes": len(set(batch_outcome_ids)),
                "batch_unique_topologies": len(set(batch_topology_ids)),
                "cumulative_unique_outcomes": len(cumulative_outcomes),
                "global_duplicate_fraction": (
                    1.0
                    - len(cumulative_outcomes)
                    / max(sum(cumulative_outcomes.values()), 1)
                ),
                "mean_log_reward": float(batch["log_rewards"].mean().item()),
                "mean_log_score": float(batch["log_scores"].mean().item()),
                **_replay_batch_metrics(batch, replay_tree_count),
                **forward_metrics,
                **reverse_metrics,
            }
            append_jsonl(metrics_path, record)
            epoch_losses.append(float(record["loss"]))
            epoch_reverse_losses.append(float(record["reverse_loss"]))
            epoch_ess.append(float(record["ips_ess_fraction"]))

            if config.print_every > 0 and global_step % config.print_every == 0:
                replay_msg = ""
                if replay_enabled and "mean_log_score_on_policy" in record:
                    replay_msg = (
                        f" logL_replay={record['mean_log_score_replay'] - log_score_shift:.0f}"
                        f" logL_on={record['mean_log_score_on_policy'] - log_score_shift:.0f}"
                        f" best_buf={record['best_trees_buffer_size']}"
                    )
                print(
                    f"step={global_step:05d} loss={record['loss']:.4f} "
                    f"reverse_nll={record['reverse_loss']:.4f} "
                    f"ESS={record['ips_ess_fraction']:.3f} "
                    f"batch_outcomes={record['batch_unique_outcomes']} "
                    f"batch_topologies={record['batch_unique_topologies']} "
                    f"seen={record['cumulative_unique_outcomes']}"
                    f"{replay_msg}"
                )
            global_step += 1

        summary = {
            "epoch": epoch,
            "method": config.method,
            "mean_loss": float(np.mean(epoch_losses)),
            "mean_reverse_loss": float(np.mean(epoch_reverse_losses)),
            "mean_ips_ess_fraction": float(np.mean(epoch_ess)),
            "cumulative_unique_outcomes": len(cumulative_outcomes),
        }
        epoch_summaries.append(summary)

        if config.checkpoint_every > 0 and (epoch + 1) % config.checkpoint_every == 0:
            checkpoint_name = f"checkpoint_epoch{epoch:04d}.pt"
            checkpoint_path = output_dir / checkpoint_name
            generator.save(str(checkpoint_path))
            _save_learned_reverse_state(
                output_dir / f"learned_reverse_epoch{epoch:04d}.pt",
                policy=policy,
                optimizer=reverse_optimizer,
                forward_trainer=forward_trainer,
                normalizer=normalizer,
                update_step=global_step,
                reverse_policy_type=config.reverse_policy_type,
                reverse_config=reverse_config,
            )
            outcome_tracker = OutcomeTracker()
            outcome_tracker.outcome_counts.update(cumulative_outcomes)
            outcome_tracker.total = sum(cumulative_outcomes.values())
            save_training_state(
                output_dir,
                make_training_state(
                    global_step=global_step,
                    training_mode="on_policy",
                    epoch=epoch,
                    step=step,
                    steps_per_epoch=data_loader.steps_per_epoch,
                    checkpoint_path=str(checkpoint_path),
                    tracker=outcome_tracker,
                ),
                grpo_trainer=forward_trainer,
            )
            _save_best_trees(output_dir, data_loader)

    generator.save(str(output_dir / "final_checkpoint.pt"))
    _save_best_trees(output_dir, data_loader)
    _save_learned_reverse_state(
        output_dir / "learned_reverse_state.pt",
        policy=policy,
        optimizer=reverse_optimizer,
        forward_trainer=forward_trainer,
        normalizer=normalizer,
        update_step=global_step,
        reverse_policy_type=config.reverse_policy_type,
        reverse_config=reverse_config,
    )
    (output_dir / "epoch_summaries.json").write_text(
        json.dumps(epoch_summaries, indent=2), encoding="utf-8"
    )
    outcome_tracker = OutcomeTracker()
    outcome_tracker.outcome_counts.update(cumulative_outcomes)
    outcome_tracker.total = sum(cumulative_outcomes.values())
    save_training_state(
        output_dir,
        make_training_state(
            global_step=global_step,
            training_mode="on_policy",
            epoch=cfg.GFN.TRAINING_DATA_LOADER.EPOCHS_NUM - 1,
            step=data_loader.steps_per_epoch - 1,
            steps_per_epoch=data_loader.steps_per_epoch,
            checkpoint_path=str(output_dir / "final_checkpoint.pt"),
            tracker=outcome_tracker,
        ),
        grpo_trainer=forward_trainer,
    )
    print(f"completed: {output_dir}")
    return str(output_dir)


def parse_config(argv: list[str] | None = None) -> LearnedReverseExperimentConfig:
    parser = build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(
        cfg_path=(
            "src/configs/benchmark_dna_cfgs/discrete_branch_lengths/"
            "cfg_0.001binsize_50bins_temperature_anneal_0.4_tree_only.yaml"
        ),
        dataset_path="dataset/benchmark_datasets/DS1_reduced.pickle",
        output_root="grpo_experiments/learned_reverse_runs",
        run_name="learned_reverse_5taxa",
        epochs=4_000,
        steps_per_epoch=1,
        on_policy_batch_size=128,
        replay_batch_size=0,
        disable_replay=True,
        grpo_lr=1e-4,
        grpo_entropy_coef=0.0,
        grpo_num_iterations=1,
        outcome_level="topology",
        policy_loss_mode="ppo",
        print_every=25,
        checkpoint_every=500,
        rollout_chunk_size=128,
    )
    group = parser.add_argument_group("learned reverse proposal")
    group.add_argument(
        "--reward-target",
        choices=("likelihood", "shifted_linear"),
        default="likelihood",
        help=(
            "likelihood: q*(x) proportional to exp(log L(x)); "
            "shifted_linear: q*(x) proportional to the positive shifted log score."
        ),
    )
    group.add_argument(
        "--learn-edge-lengths",
        action="store_false",
        dest="only_train_tree_model",
        help=(
            "Train the categorical branch-length policy from the selected YAML. "
            "Without this flag, use the fixed-edge tree-only model."
        ),
    )
    group.add_argument(
        "--fixed-edge-lengths",
        action="store_true",
        dest="only_train_tree_model",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(only_train_tree_model=True)
    group.add_argument(
        "--reverse-policy-type",
        choices=("tabular", "mlp"),
        default="tabular",
        help=(
            "tabular: exact normalized q over enumerated merge histories (5 taxa only); "
            "mlp: per-step learned reverse policy that scales beyond small catalogs."
        ),
    )
    group.add_argument("--reverse-lr", type=float, default=1e-3)
    group.add_argument("--reverse-train-epochs", type=int, default=4)
    group.add_argument("--reverse-grad-clip-norm", type=float, default=1.0)
    group.add_argument("--reverse-hidden-size", type=int, default=128)
    group.add_argument("--reverse-num-layers", type=int, default=2)
    group.add_argument(
        "--advantage-normalization",
        choices=("batch", "running"),
        default="running",
    )
    group.add_argument("--running-scale-decay", type=float, default=0.99)
    group.add_argument("--running-advantage-clip", type=float, default=10.0)
    group.add_argument("--running-log-ratio-clip", type=float, default=20.0)
    args = parser.parse_args(argv)
    base = config_from_args(args)
    return LearnedReverseExperimentConfig.from_base(
        base,
        only_train_tree_model=args.only_train_tree_model,
        reward_target=args.reward_target,
        reverse_policy_type=args.reverse_policy_type,
        reverse_lr=args.reverse_lr,
        reverse_train_epochs=args.reverse_train_epochs,
        reverse_grad_clip_norm=args.reverse_grad_clip_norm,
        reverse_hidden_size=args.reverse_hidden_size,
        reverse_num_layers=args.reverse_num_layers,
        advantage_normalization=args.advantage_normalization,
        running_scale_decay=args.running_scale_decay,
        running_advantage_clip=args.running_advantage_clip,
        running_log_ratio_clip=args.running_log_ratio_clip,
    )


def main() -> None:
    config = parse_config()
    run_experiment(config)


if __name__ == "__main__":
    main()
