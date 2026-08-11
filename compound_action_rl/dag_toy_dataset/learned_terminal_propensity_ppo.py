"""PPO with a learned terminal propensity model and a value critic.

This is the non-oracle successor to ``oracle_terminal_propensity_ppo.py``.
Training weights never use exact DAG probabilities.  Instead, a normalized
terminal density model is fitted by maximum likelihood to a recent buffer of
behavior-policy outcomes:

    p_hat_phi(x) = softmax_x f_phi(features(x))
    p_safe(x)    = (1 - delta) p_hat_target(x) + delta / |X|
    W(x)         = R(x) / (sum_x R(x) * p_safe(x))
    A_t          = stop_gradient(W(x) - V_psi(s_t))

The density model is trained on a faster time scale than the policy.  A target
copy supplies stable, stop-gradient probabilities to PPO.  Current PPO
episodes are added to the recent buffer only after their probabilities have
been fixed, so an episode cannot immediately lower its own inverse weight.

For this toy experiment only, exact forward DAG probabilities are computed as
diagnostics.  They measure density calibration and reward-correlated bias but
are never read by the learned provider, the importance targets, the critic, or
the PPO loss.  On a non-enumerable DAG, the diagnostic adapter can simply be
removed while retaining the learned-provider/PPO interface.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from config import TrainConfig  # noqa: E402
from count_ips import Episode  # noqa: E402
from dag_env import State  # noqa: E402
from epsilon_greedy_count_ips import (  # noqa: E402
    EpsilonGreedyCountIPSTrainer,
    ExplorationConfig,
    _plot_exploration,
    _resolve_device,
)
from oracle_terminal_propensity_ppo import (  # noqa: E402
    ExactToyDAGTerminalProbabilityProvider,
    OraclePPOConfig,
    OracleTerminalPropensityPPOTrainer,
    TerminalProbabilityProvider,
    _distribution_metrics,
)
from run_count_ips import (  # noqa: E402
    _plot_final_counts,
    _plot_trajectory_diagnostics,
)


@dataclass(frozen=True)
class LearnedPropensityConfig:
    """Recent-buffer density-estimation and target-network settings."""

    hidden_size: int = 128
    num_layers: int = 2
    lr: float = 3e-3
    grad_clip_norm: float = 1.0
    buffer_size: int = 2_048
    minibatch_size: int = 256
    gradient_steps_per_update: int = 20
    warmup_groups: int = 32
    warmup_gradient_steps: int = 200
    target_tau: float = 0.2
    uniform_mix: float = 0.01

    def validate(self) -> None:
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("density network dimensions must be >= 1")
        if self.lr <= 0.0 or self.grad_clip_norm <= 0.0:
            raise ValueError("density lr and grad clip must be > 0")
        if self.buffer_size < 1 or self.minibatch_size < 1:
            raise ValueError("density buffer and minibatch sizes must be >= 1")
        if self.gradient_steps_per_update < 1:
            raise ValueError("density gradient_steps_per_update must be >= 1")
        if self.warmup_groups < 1 or self.warmup_gradient_steps < 1:
            raise ValueError("density warmup groups and steps must be >= 1")
        if not 0.0 < self.target_tau <= 1.0:
            raise ValueError("density target_tau must be in (0, 1]")
        if not 0.0 <= self.uniform_mix < 1.0:
            raise ValueError("density uniform_mix must be in [0, 1)")


class TerminalDensityNetwork(nn.Module):
    """Shared scorer normalized over a supplied terminal candidate set."""

    def __init__(
        self, feature_dim: int, hidden_size: int, num_layers: int
    ) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        layers: list[nn.Module] = []
        width = feature_dim
        for _ in range(num_layers):
            layers.extend((nn.Linear(width, hidden_size), nn.Tanh()))
            width = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.score_head = nn.Linear(width, 1)
        # The initial normalized distribution is exactly uniform.
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def log_probabilities(self, terminal_features: torch.Tensor) -> torch.Tensor:
        if terminal_features.ndim != 2:
            raise ValueError("terminal_features must have shape (terminals, features)")
        scores = self.score_head(self.trunk(terminal_features)).squeeze(-1)
        return F.log_softmax(scores, dim=0)


def toy_terminal_features(
    terminals: Sequence[State], *, budget: int
) -> torch.Tensor:
    """Toy adapter; the learned PPO core only receives this feature matrix."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    features = [
        (state.x / float(budget), state.y / float(budget))
        for state in terminals
    ]
    return torch.tensor(features, dtype=torch.float32)


class LearnedTerminalProbabilityProvider:
    """Recent-buffer MLE density with a slowly moving target snapshot."""

    def __init__(
        self,
        terminals: Sequence[object],
        terminal_features: torch.Tensor,
        config: LearnedPropensityConfig,
        *,
        device: torch.device,
    ) -> None:
        config.validate()
        if len(terminals) < 2:
            raise ValueError("at least two terminal candidates are required")
        if terminal_features.shape[0] != len(terminals):
            raise ValueError("one feature row is required per terminal")
        self.terminals = list(terminals)
        self.terminal_indices = {
            terminal: index for index, terminal in enumerate(self.terminals)
        }
        self.config = config
        self.device = device
        self.features = terminal_features.to(
            device=device, dtype=torch.float32
        )
        feature_dim = int(self.features.shape[1])
        self.online_model = TerminalDensityNetwork(
            feature_dim, config.hidden_size, config.num_layers
        ).to(device)
        self.target_model = TerminalDensityNetwork(
            feature_dim, config.hidden_size, config.num_layers
        ).to(device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.online_model.parameters(), lr=config.lr
        )
        self.buffer: deque[int] = deque(maxlen=config.buffer_size)
        self._cached_distribution: dict[object, float] = {}

    def add_outcomes(self, outcomes: Sequence[object]) -> None:
        try:
            self.buffer.extend(self.terminal_indices[outcome] for outcome in outcomes)
        except KeyError as error:
            raise ValueError("outcome is outside the terminal candidate set") from error

    def _safe_log_probabilities(
        self, model_log_probabilities: torch.Tensor
    ) -> torch.Tensor:
        uniform_mix = self.config.uniform_mix
        if uniform_mix == 0.0:
            return model_log_probabilities
        uniform_log_probability = -math.log(len(self.terminals))
        return torch.logaddexp(
            model_log_probabilities + math.log1p(-uniform_mix),
            torch.full_like(
                model_log_probabilities,
                math.log(uniform_mix) + uniform_log_probability,
            ),
        )

    def fit(self, gradient_steps: int) -> dict[str, float]:
        if not self.buffer:
            raise RuntimeError("the density buffer is empty")
        if gradient_steps < 1:
            raise ValueError("gradient_steps must be >= 1")
        buffer_indices = torch.tensor(
            list(self.buffer), dtype=torch.long, device=self.device
        )
        minibatch_size = min(
            self.config.minibatch_size, buffer_indices.numel()
        )
        loss_total = 0.0
        grad_norm_total = 0.0
        parameters = list(self.online_model.parameters())
        self.online_model.train()
        for _ in range(gradient_steps):
            sampled_positions = torch.randint(
                buffer_indices.numel(),
                (minibatch_size,),
                device=self.device,
            )
            sampled_outcomes = buffer_indices[sampled_positions]
            self.optimizer.zero_grad(set_to_none=True)
            log_probabilities = self.online_model.log_probabilities(
                self.features
            )
            loss = -log_probabilities[sampled_outcomes].mean()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                parameters, self.config.grad_clip_norm
            )
            self.optimizer.step()
            loss_total += float(loss.item())
            grad_norm_total += float(grad_norm.item())
        self.online_model.eval()

        with torch.inference_mode():
            online_log_probability = self.online_model.log_probabilities(
                self.features
            )
            online_probability = online_log_probability.exp()
            empirical = torch.bincount(
                buffer_indices, minlength=len(self.terminals)
            ).to(dtype=online_probability.dtype)
            empirical /= buffer_indices.numel()
            positive = empirical > 0.0
            empirical_kl = (
                empirical[positive]
                * (
                    empirical[positive].log()
                    - online_log_probability[positive]
                )
            ).sum()
            return {
                "density_online_loss": loss_total / gradient_steps,
                "density_online_grad_norm": grad_norm_total / gradient_steps,
                "density_online_nll_buffer": float(
                    -online_log_probability[buffer_indices].mean().item()
                ),
                "density_online_entropy": float(
                    -(online_probability * online_log_probability).sum().item()
                ),
                "density_online_empirical_tv": float(
                    (0.5 * torch.abs(online_probability - empirical).sum()).item()
                ),
                "density_online_empirical_kl": float(empirical_kl.item()),
                "density_buffer_size": float(buffer_indices.numel()),
                "density_buffer_unique_outcomes": float(
                    torch.unique(buffer_indices).numel()
                ),
            }

    @torch.inference_mode()
    def sync_target(self, *, hard: bool = False) -> None:
        tau = 1.0 if hard else self.config.target_tau
        for target_parameter, online_parameter in zip(
            self.target_model.parameters(), self.online_model.parameters()
        ):
            target_parameter.lerp_(online_parameter, tau)

    @torch.inference_mode()
    def refresh(self) -> None:
        log_probability = self.target_model.log_probabilities(self.features)
        safe_probability = self._safe_log_probabilities(
            log_probability
        ).exp()
        self._cached_distribution = {
            terminal: float(probability)
            for terminal, probability in zip(
                self.terminals, safe_probability.cpu().tolist()
            )
        }

    def _require_refresh(self) -> None:
        if not self._cached_distribution:
            raise RuntimeError("refresh() must be called before probability lookup")

    def log_probabilities(self, outcomes: Sequence[object]) -> np.ndarray:
        self._require_refresh()
        try:
            probability = np.asarray(
                [self._cached_distribution[outcome] for outcome in outcomes],
                dtype=np.float64,
            )
        except KeyError as error:
            raise ValueError("outcome is outside the terminal candidate set") from error
        return np.log(probability)

    @property
    def behavior_distribution(self) -> Mapping[object, float]:
        self._require_refresh()
        return self._cached_distribution

    @property
    def evaluation_distribution(self) -> Mapping[object, float]:
        # A learned outcome marginal does not distinguish behavior/evaluation
        # policies; this alias exists only to satisfy the provider protocol.
        return self.behavior_distribution

    @torch.inference_mode()
    def snapshot_metrics(self) -> dict[str, float]:
        self._require_refresh()
        target_log_probability = self.target_model.log_probabilities(
            self.features
        )
        target_probability = target_log_probability.exp()
        safe_probability = self._safe_log_probabilities(
            target_log_probability
        ).exp()
        return {
            "density_target_entropy": float(
                -(target_probability * target_log_probability).sum().item()
            ),
            "density_target_min_probability": float(
                target_probability.min().item()
            ),
            "density_target_max_probability": float(
                target_probability.max().item()
            ),
            "density_safe_min_probability": float(
                safe_probability.min().item()
            ),
            "density_safe_max_probability": float(
                safe_probability.max().item()
            ),
            "density_safe_probability_mass": float(
                safe_probability.sum().item()
            ),
            "density_uniform_mix": self.config.uniform_mix,
        }


def _weighted_correlation(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> float:
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return 0.0
    normalized = weights / weight_sum
    left_centered = left - float(np.sum(normalized * left))
    right_centered = right - float(np.sum(normalized * right))
    covariance = float(np.sum(normalized * left_centered * right_centered))
    left_variance = float(np.sum(normalized * np.square(left_centered)))
    right_variance = float(np.sum(normalized * np.square(right_centered)))
    denominator = math.sqrt(left_variance * right_variance)
    return covariance / denominator if denominator > 0.0 else 0.0


def density_calibration_metrics(
    exact_probability: np.ndarray,
    learned_probability: np.ndarray,
    rewards: np.ndarray,
) -> dict[str, float]:
    """Compare learned p-hat with exact p for diagnostics only."""
    if (
        exact_probability.shape != learned_probability.shape
        or exact_probability.shape != rewards.shape
    ):
        raise ValueError("calibration arrays must have equal shape")
    tiny = np.finfo(np.float64).tiny
    exact = np.maximum(exact_probability, tiny)
    learned = np.maximum(learned_probability, tiny)
    exact /= exact.sum()
    learned /= learned.sum()
    log_bias = np.log(learned) - np.log(exact)
    exact_to_learned = float(
        np.sum(exact * (np.log(exact) - np.log(learned)))
    )
    learned_to_exact = float(
        np.sum(learned * (np.log(learned) - np.log(exact)))
    )
    unweighted_correlation = (
        float(np.corrcoef(log_bias, rewards)[0, 1])
        if np.std(log_bias) > 0.0 and np.std(rewards) > 0.0
        else 0.0
    )
    slope = (
        float(np.polyfit(rewards, log_bias, 1)[0])
        if np.std(rewards) > 0.0
        else 0.0
    )
    return {
        "density_tv_exact_behavior": float(
            0.5 * np.abs(exact - learned).sum()
        ),
        "density_kl_exact_to_learned": exact_to_learned,
        "density_kl_learned_to_exact": learned_to_exact,
        "density_mean_abs_log_bias": float(
            np.sum(exact * np.abs(log_bias))
        ),
        "density_max_abs_log_bias": float(np.max(np.abs(log_bias))),
        "density_log_bias_reward_correlation": unweighted_correlation,
        "density_weighted_log_bias_reward_correlation": (
            _weighted_correlation(log_bias, rewards, exact)
        ),
        "density_log_bias_reward_slope": slope,
    }


class LearnedTerminalPropensityPPOTrainer(
    OracleTerminalPropensityPPOTrainer
):
    """Learned p-hat provider plugged into the shared PPO/value-critic core."""

    probability_label = "p_hat(x)"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
        exploration: ExplorationConfig | None = None,
        oracle_ppo: OraclePPOConfig | None = None,
        learned_propensity: LearnedPropensityConfig | None = None,
    ) -> None:
        super().__init__(
            config,
            device=device,
            exploration=exploration,
            oracle_ppo=oracle_ppo,
        )
        self.learned_propensity_config = (
            learned_propensity or LearnedPropensityConfig()
        )
        self.learned_propensity_config.validate()

        # The parent constructs this exact provider.  Retain it only under an
        # explicitly diagnostic name, then replace the training provider.
        self.exact_diagnostic_provider = self.probability_provider
        terminal_features = toy_terminal_features(
            self.terminals, budget=self.config.budget
        )
        self.learned_probability_provider = (
            LearnedTerminalProbabilityProvider(
                self.terminals,
                terminal_features,
                self.learned_propensity_config,
                device=self.device,
            )
        )
        self.probability_provider: TerminalProbabilityProvider = (
            self.learned_probability_provider
        )
        self._density_warmup_complete = False
        self._density_fit_metrics: dict[str, float] = {}
        self._warmup_summary: dict[str, Any] = {}
        self._reward_array = np.asarray(
            [self.reward_by_terminal[state] for state in self.terminals],
            dtype=np.float64,
        )

    def _run_density_warmup(self) -> None:
        outcome_counts: dict[State, int] = {}
        total_rollouts = 0
        for _ in range(self.learned_propensity_config.warmup_groups):
            episodes = self.rollout_batch(
                self.config.group_size, explore=True
            )
            outcomes = [episode.terminal for episode in episodes]
            self.learned_probability_provider.add_outcomes(outcomes)
            self._seen_terminals.update(outcomes)
            total_rollouts += len(outcomes)
            for outcome in outcomes:
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        self._density_fit_metrics = self.learned_probability_provider.fit(
            self.learned_propensity_config.warmup_gradient_steps
        )
        self.learned_probability_provider.sync_target(hard=True)
        self.learned_probability_provider.refresh()
        self._density_fit_metrics.update(
            self.learned_probability_provider.snapshot_metrics()
        )
        self._density_warmup_complete = True
        self._warmup_summary = {
            "groups": self.learned_propensity_config.warmup_groups,
            "rollouts": total_rollouts,
            "unique_outcomes": len(outcome_counts),
            "outcome_counts": {
                state.signature: outcome_counts.get(state, 0)
                for state in self.terminals
            },
        }
        print(
            "Density warmup: "
            f"rollouts={total_rollouts}  "
            f"outcomes={len(outcome_counts)}/{len(self.terminals)}  "
            f"buffer={len(self.learned_probability_provider.buffer)}  "
            f"NLL={self._density_fit_metrics['density_online_nll_buffer']:.3f}"
        )

    def _collect_training_groups(
        self,
    ) -> tuple[list[list[Episode]], list[dict[str, float]]]:
        if not self._density_warmup_complete:
            self._run_density_warmup()
        else:
            self._density_fit_metrics = (
                self.learned_probability_provider.fit(
                    self.learned_propensity_config.gradient_steps_per_update
                )
            )
            self.learned_probability_provider.sync_target()
            self.learned_probability_provider.refresh()
            self._density_fit_metrics.update(
                self.learned_probability_provider.snapshot_metrics()
            )

        # Diagnostic only: no value from this provider enters p-hat, W, V, or
        # PPO.  It is refreshed before the rollout so it describes the same q.
        assert isinstance(
            self.exact_diagnostic_provider,
            ExactToyDAGTerminalProbabilityProvider,
        )
        self.exact_diagnostic_provider.refresh()

        # Bypass the parent's oracle refresh; the learned target snapshot above
        # is now frozen for both rollout collection and the subsequent update.
        groups, group_metrics = (
            EpsilonGreedyCountIPSTrainer._collect_training_groups(self)
        )

        current_outcomes = [
            episode.terminal
            for group in groups
            for episode in group
        ]
        # Added only after p-hat and advantages were computed.  The online
        # density will see these outcomes before the *next* policy update.
        self.learned_probability_provider.add_outcomes(current_outcomes)
        for metrics in group_metrics:
            metrics.update(self._density_fit_metrics)
            metrics["density_buffer_size_after_collection"] = float(
                len(self.learned_probability_provider.buffer)
            )
            metrics["density_warmup_rollouts"] = float(
                self._warmup_summary["rollouts"]
            )
            metrics["total_rollouts_per_update"] = float(
                len(current_outcomes)
            )
        return groups, group_metrics

    def _provider_distribution_metrics(self) -> dict[str, float]:
        exact_behavior = np.asarray(
            [
                self.exact_diagnostic_provider.behavior_distribution[state]
                for state in self.terminals
            ],
            dtype=np.float64,
        )
        exact_policy = np.asarray(
            [
                self.exact_diagnostic_provider.evaluation_distribution[state]
                for state in self.terminals
            ],
            dtype=np.float64,
        )
        learned = np.asarray(
            [
                self.learned_probability_provider.behavior_distribution[state]
                for state in self.terminals
            ],
            dtype=np.float64,
        )
        metrics = _distribution_metrics(
            exact_behavior,
            self._target_probability_array,
            prefix="exact_behavior",
        )
        metrics.update(
            _distribution_metrics(
                exact_policy,
                self._target_probability_array,
                prefix="exact_policy",
            )
        )
        metrics.update(
            _distribution_metrics(
                learned,
                self._target_probability_array,
                prefix="learned_propensity",
            )
        )
        metrics.update(
            density_calibration_metrics(
                exact_behavior, learned, self._reward_array
            )
        )
        metrics["exact_behavior_policy_tv"] = float(
            0.5 * np.abs(exact_behavior - exact_policy).sum()
        )
        return metrics

    def _update_training_groups(
        self, groups: list[list[Episode]]
    ) -> dict[str, float]:
        statistics = self.update(
            [episode for group in groups for episode in group]
        )
        if (
            self._current_update_step == 1
            or self._current_update_step % self.config.log_every == 0
        ):
            metrics = self._last_ips_metrics
            print(
                "  learned "
                f"TV(p,p_hat)={metrics['density_tv_exact_behavior']:.3f}  "
                f"KL(p||p_hat)={metrics['density_kl_exact_to_learned']:.3f}  "
                "bias_r="
                f"{metrics['density_weighted_log_bias_reward_correlation']:.3f}  "
                f"TV(q,target)={metrics['exact_behavior_tv_reward_target']:.3f}  "
                f"TV(pi,target)={metrics['exact_policy_tv_reward_target']:.3f}  "
                f"ESS={metrics['ips_ess_fraction']:.3f}  "
                f"clip={metrics['importance_weight_clip_fraction']:.3f}  "
                f"V_loss={statistics['value_loss']:.3f}"
            )
        return statistics

    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "direction_policy": self.direction_policy.state_dict(),
                "step_policy": self.step_policy.state_dict(),
                "value_network": self.value_network.state_dict(),
                "density_online_model": (
                    self.learned_probability_provider.online_model.state_dict()
                ),
                "density_target_model": (
                    self.learned_probability_provider.target_model.state_dict()
                ),
                "density_buffer": list(
                    self.learned_probability_provider.buffer
                ),
                "oracle_ppo_config": asdict(self.oracle_ppo),
                "learned_propensity_config": asdict(
                    self.learned_propensity_config
                ),
                "exploration": asdict(self.exploration),
                "current_epsilon": self.current_epsilon,
                "current_temperature": self.current_temperature,
                "warmup_summary": self._warmup_summary,
                "update_step": update_step,
                "algorithm": {
                    "name": "learned_terminal_propensity_ppo",
                    "propensity": "learned_recent_behavior_terminal_marginal",
                    "exact_oracle_role": "diagnostics_only",
                    "objective": self.oracle_ppo.objective,
                    "value_target": "learned_terminal_importance_target",
                },
            },
            path,
        )
        return path


def _plot_learned_training(
    history: list[dict[str, Any]], *, output: Path
) -> None:
    steps = [row["step"] for row in history]
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    axes[0, 0].plot(
        steps,
        [row["exact_behavior_tv_reward_target"] for row in history],
        label="behavior q",
    )
    axes[0, 0].plot(
        steps,
        [row["exact_policy_tv_reward_target"] for row in history],
        label="base policy pi",
    )
    axes[0, 0].set_title("Exact policy TV to reward target")
    axes[0, 0].legend()

    axes[0, 1].plot(
        steps,
        [row["density_tv_exact_behavior"] for row in history],
        label="TV(p-hat, q)",
    )
    axes[0, 1].plot(
        steps,
        [row["density_kl_exact_to_learned"] for row in history],
        label="KL(q || p-hat)",
    )
    axes[0, 1].set_title("Density calibration")
    axes[0, 1].legend()

    axes[0, 2].plot(
        steps,
        [
            row["density_weighted_log_bias_reward_correlation"]
            for row in history
        ],
        label="q-weighted",
    )
    axes[0, 2].plot(
        steps,
        [row["density_log_bias_reward_correlation"] for row in history],
        alpha=0.65,
        label="all terminals",
    )
    axes[0, 2].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0, 2].set_ylim(-1.05, 1.05)
    axes[0, 2].set_title("Correlation: log(p-hat/q) vs reward")
    axes[0, 2].legend()

    axes[1, 0].plot(
        steps,
        [row["ips_ess_fraction"] for row in history],
        label="weight ESS / batch",
    )
    axes[1, 0].plot(
        steps,
        [row["importance_weight_clip_fraction"] for row in history],
        label="weight clip fraction",
    )
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_title("Importance-weight stability")
    axes[1, 0].legend()

    axes[1, 1].plot(
        steps,
        [row["density_online_nll_buffer"] for row in history],
        label="buffer NLL",
    )
    axes[1, 1].plot(
        steps,
        [row["density_online_empirical_tv"] for row in history],
        label="TV(online, buffer)",
    )
    axes[1, 1].set_title("Fast density-model fit")
    axes[1, 1].legend()

    axes[1, 2].plot(
        steps,
        [row["critic_preupdate_mse"] for row in history],
        label="critic pre-update MSE",
    )
    axes[1, 2].plot(
        steps,
        [row["value_loss"] for row in history],
        label="critic training loss",
    )
    axes[1, 2].set_yscale("symlog", linthresh=1e-4)
    axes[1, 2].set_title("Value-critic fit")
    axes[1, 2].legend()

    for axis in axes.flat:
        axis.set_xlabel("Update")
        axis.grid(alpha=0.22)
    figure.suptitle(
        "Learned terminal-propensity PPO diagnostics "
        "(exact probabilities are evaluation-only)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--num-updates", type=int, default=2_000)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--objective", choices=("raw", "log"), default="raw")
    parser.add_argument("--max-log-weight", type=float, default=8.0)
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--value-hidden-size", type=int, default=128)
    parser.add_argument("--value-num-layers", type=int, default=2)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--value-train-epochs", type=int, default=4)
    parser.add_argument("--value-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--density-hidden-size", type=int, default=128)
    parser.add_argument("--density-num-layers", type=int, default=2)
    parser.add_argument("--density-lr", type=float, default=3e-3)
    parser.add_argument("--density-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--density-buffer-size", type=int, default=2_048)
    parser.add_argument("--density-minibatch-size", type=int, default=256)
    parser.add_argument("--density-steps-per-update", type=int, default=20)
    parser.add_argument("--density-warmup-groups", type=int, default=32)
    parser.add_argument("--density-warmup-steps", type=int, default=200)
    parser.add_argument("--density-target-tau", type=float, default=0.2)
    parser.add_argument("--density-uniform-mix", type=float, default=0.01)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--temperature-start", type=float, default=2.0)
    parser.add_argument("--temperature-end", type=float, default=1.0)
    parser.add_argument("--anneal-updates", type=int, default=1_500)
    parser.add_argument(
        "--schedule", choices=("linear", "cosine"), default="cosine"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--terminal-rewards", type=float, nargs="+", default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

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
        num_updates=args.num_updates,
        train_epochs=args.train_epochs,
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
    oracle_ppo = OraclePPOConfig(
        objective=args.objective,
        max_log_weight=(
            None if args.max_log_weight < 0.0 else args.max_log_weight
        ),
        normalize_advantages=not args.no_normalize_advantages,
        value_hidden_size=args.value_hidden_size,
        value_num_layers=args.value_num_layers,
        value_lr=args.value_lr,
        value_train_epochs=args.value_train_epochs,
        value_grad_clip_norm=args.value_grad_clip_norm,
    )
    learned_propensity = LearnedPropensityConfig(
        hidden_size=args.density_hidden_size,
        num_layers=args.density_num_layers,
        lr=args.density_lr,
        grad_clip_norm=args.density_grad_clip_norm,
        buffer_size=args.density_buffer_size,
        minibatch_size=args.density_minibatch_size,
        gradient_steps_per_update=args.density_steps_per_update,
        warmup_groups=args.density_warmup_groups,
        warmup_gradient_steps=args.density_warmup_steps,
        target_tau=args.density_target_tau,
        uniform_mix=args.density_uniform_mix,
    )
    run_directory = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "learned_terminal_propensity_ppo_runs"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}"
            f"_gs{config.group_size}_seed{config.seed}"
        )
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    trainer = LearnedTerminalPropensityPPOTrainer(
        config,
        device=_resolve_device(args.device),
        exploration=exploration,
        oracle_ppo=oracle_ppo,
        learned_propensity=learned_propensity,
    )
    checkpoint_every = args.checkpoint_every or None

    print(f"Run directory: {run_directory}")
    print(f"Device: {trainer.device}")
    print(f"DAG: {trainer.environment_summary()}")
    print(
        "Algorithm: learned recent-buffer terminal propensity; "
        f"objective={oracle_ppo.objective}; separate state-value critic; "
        f"group_size={config.group_size}"
    )
    print(
        "Exact DAG probabilities: diagnostics only; never used for weights, "
        "advantages, critic targets, or PPO"
    )

    (run_directory / "config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(config),
                "exploration": asdict(exploration),
                "oracle_ppo": asdict(oracle_ppo),
                "learned_propensity": asdict(learned_propensity),
                "device": str(trainer.device),
                "algorithm": {
                    "name": "learned_terminal_propensity_ppo",
                    "propensity": "learned_recent_behavior_terminal_marginal",
                    "exact_oracle_role": "diagnostics_only",
                    "density_update_order": (
                        "fit_previous_recent_buffer_then_freeze_target"
                    ),
                    "current_rollout_usage": (
                        "added_to_buffer_after_current_weights_are_fixed"
                    ),
                    "objective": oracle_ppo.objective,
                },
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
        checkpoint_dir=run_directory / "checkpoints",
    )
    (run_directory / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    trainer.save(
        run_directory / "checkpoint.pt", update_step=config.num_updates
    )

    density_plot = run_directory / "learned_density_diagnostics.png"
    _plot_learned_training(history, output=density_plot)
    exploration_plot = run_directory / "exploration_schedule.png"
    _plot_exploration(history, exploration_plot)
    evaluation = trainer.evaluate(args.final_samples)
    sampling = _plot_final_counts(
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_directory / "sampling_counts.png",
        suptitle="Learned terminal-propensity PPO vs ideal reward sampling",
    )
    trajectories = _plot_trajectory_diagnostics(
        history,
        trainer,
        evaluation,
        samples=args.final_samples,
        output=run_directory / "trajectory_sampling.png",
        subtitle="The learned propensity models outcomes, not trajectories",
    )

    trainer.learned_probability_provider.refresh()
    trainer.exact_diagnostic_provider.refresh()
    final_calibration = trainer._provider_distribution_metrics()
    summary: dict[str, Any] = {
        "environment": trainer.environment_summary(),
        "algorithm": {
            "name": "learned_terminal_propensity_ppo",
            "objective": oracle_ppo.objective,
            "propensity": "learned_recent_behavior_terminal_marginal",
            "exact_oracle_role": "diagnostics_only",
            "value_critic": True,
            "group_frequency_estimator": False,
        },
        "warmup": trainer._warmup_summary,
        "final_calibration": final_calibration,
        "final_learned_probabilities": {
            state.signature: float(
                trainer.learned_probability_provider.behavior_distribution[
                    state
                ]
            )
            for state in trainer.terminals
        },
        "final_exact_behavior_probabilities_diagnostic": {
            state.signature: float(
                trainer.exact_diagnostic_provider.behavior_distribution[state]
            )
            for state in trainer.terminals
        },
        "final_sampling": sampling,
        "trajectory_sampling": trajectories,
        "final_training_diagnostics": {
            key: history[-1][key]
            for key in (
                "density_tv_exact_behavior",
                "density_kl_exact_to_learned",
                "density_weighted_log_bias_reward_correlation",
                "ips_ess_fraction",
                "importance_weight_clip_fraction",
                "critic_preupdate_mse",
                "value_loss",
                "exact_behavior_tv_reward_target",
                "exact_policy_tv_reward_target",
            )
        },
        "plots": {
            "learned_density_diagnostics": density_plot.name,
            "exploration_schedule": exploration_plot.name,
            "sampling_counts": "sampling_counts.png",
            "trajectory_sampling": "trajectory_sampling.png",
        },
    }
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Final sampled TV distance: {sampling['tv_reward_target']:.4f}")
    print(
        "Final density TV to exact behavior: "
        f"{final_calibration['density_tv_exact_behavior']:.4f}"
    )
    print(
        "Final exact base-policy TV distance: "
        f"{final_calibration['exact_policy_tv_reward_target']:.4f}"
    )
    print(f"Final R^2: {sampling['r2_reward_target']:.4f}")
    print(f"Artifacts: {run_directory}")


if __name__ == "__main__":
    main()
