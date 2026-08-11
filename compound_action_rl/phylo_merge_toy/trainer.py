"""
GRPO / IPS-GRPO trainer for the merge (phylo) toy.

Single policy head over merge actions (mirrors the phylo *tree-only* model,
`ONLY_TRAIN_TREE_MODEL: true`). IPS weighting is done entirely in log-space with
a log-sum-exp SNIPS so that astronomically peaked rewards (R = exp(beta*score))
and the resulting inverse propensities do not overflow — the toy analogue of the
"broken log pi(tau) at scale" fix.

Propensity modes:
  none     : plain GRPO, advantage from group-normalised reward
  exact    : trajectory IPS,   log w = -log P_F(tau)              -> pi(x) ∝ m(x) R(x)  (biased)
  marginal : backward-corrected, log w = log P_B(tau|x) - log P_F(tau) -> pi(x) ∝ R(x)   (unbiased)
  count    : legacy within-group signature count weighting
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from catalog import RewardModel, build_catalog
from config import TrainConfig
from merge_env import MergeEnv
from networks import MergePolicyNet


@dataclass
class StepRecord:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float


@dataclass
class Episode:
    steps: list[StepRecord] = field(default_factory=list)
    signature: str = ""
    log_reward: float = 0.0
    reward: float = 0.0
    log_pf: float = 0.0
    log_pb: float = 0.0
    advantage: float = 0.0


def normalize(values: np.ndarray, eps: float) -> np.ndarray:
    mean = values.mean()
    std = values.std()
    if std < eps:
        return values - mean
    return (values - mean) / (std + eps)


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(w * w))
    if denom <= 0.0:
        return 0.0
    return float((w.sum() ** 2) / denom)


class MergeTrainer:
    def __init__(self, config: TrainConfig | None = None, *, device: str = "cpu") -> None:
        self.config = config or TrainConfig()
        self.device = torch.device(device)
        torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)

        self.reward_model = RewardModel(**self.config.reward_model_kwargs())  # type: ignore[arg-type]
        self.env = MergeEnv(n_leaves=self.config.n_leaves, reward_model=self.reward_model)
        self.obs_dim = self.env.obs_dim
        self.num_actions = self.env.num_actions

        self.policy = MergePolicyNet(
            self.obs_dim, self.num_actions, self.config.hidden_size, self.config.num_layers
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)

        self.catalog = build_catalog(self.reward_model)
        self._num_topologies = len(self.catalog.signatures)

    # ---- rollout ------------------------------------------------------------
    def rollout_episode(self) -> Episode:
        env = self.env
        obs = env.reset()
        episode = Episode()
        done = False
        info: dict = {}
        while not done:
            mask = env.action_mask()
            action, log_prob = self.policy.sample(obs, mask)
            next_obs, _, done, info = env.step(action)
            episode.steps.append(
                StepRecord(obs=obs.copy(), mask=mask.copy(), action=action, log_prob=log_prob)
            )
            obs = next_obs
        episode.signature = info["signature"]
        episode.log_reward = float(info["log_reward"])
        episode.reward = float(info["reward"])
        episode.log_pf = float(sum(s.log_prob for s in episode.steps))
        episode.log_pb = float(info["log_pb"])
        return episode

    # ---- IPS log-weights ----------------------------------------------------
    def _log_weights(self, episodes: list[Episode]) -> np.ndarray:
        mode = self.config.propensity_mode
        if mode == "none":
            return np.zeros(len(episodes), dtype=np.float64)
        if mode == "count":
            counts = Counter(ep.signature for ep in episodes)
            g = float(len(episodes))
            return np.array(
                [np.log(1.0 / max(counts[ep.signature] / g, self.config.p_eps)) for ep in episodes],
                dtype=np.float64,
            )
        log_pf = np.array([ep.log_pf for ep in episodes], dtype=np.float64)
        if mode == "exact":
            log_w = -log_pf
        elif mode == "marginal":
            log_pb = np.array([ep.log_pb for ep in episodes], dtype=np.float64)
            log_w = log_pb - log_pf
        else:
            raise ValueError(f"unknown propensity_mode {mode!r}")
        if self.config.max_inverse_weight is not None:
            log_w = np.minimum(log_w, np.log(self.config.max_inverse_weight))
        return log_w

    def _group_advantages(self, episodes: list[Episode]) -> float:
        log_reward = np.array([ep.log_reward for ep in episodes], dtype=np.float64)
        log_w = self._log_weights(episodes)
        log_scaled = log_reward + log_w  # log(R(tau) * weight(tau))

        if self.config.naive_expspace:
            # deliberately unsafe: reward*weight in raw exp space (overflows for
            # large beta). Present so the failure can be reproduced/measured.
            scaled = np.exp(log_reward) * np.exp(log_w)
        else:
            # numerically safe: subtract the group max before exponentiating.
            # advantages are z-scored so the global exp(-max) factor cancels.
            scaled = np.exp(log_scaled - log_scaled.max())

        advs = normalize(scaled, self.config.advantage_eps)
        for ep, adv in zip(episodes, advs):
            ep.advantage = float(adv)

        lin_w = np.exp(log_w - log_w.max())
        return effective_sample_size(lin_w)

    # ---- PPO update ---------------------------------------------------------
    def _ppo_surrogate(
        self, log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantage: torch.Tensor
    ) -> torch.Tensor:
        ratio = torch.exp(log_prob - old_log_prob)
        clip = self.config.clip_ratio
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
        return -torch.min(surr1, surr2).mean()

    def _policy_loss(self, steps: list[StepRecord], advantages: list[float]) -> tuple[torch.Tensor, dict[str, float]]:
        obs = torch.as_tensor(np.stack([s.obs for s in steps]), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.stack([s.mask for s in steps]), dtype=torch.bool, device=self.device)
        actions = torch.as_tensor([s.action for s in steps], dtype=torch.long, device=self.device)
        old_log_prob = torch.as_tensor([s.log_prob for s in steps], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        log_prob, entropy = self.policy.log_prob_and_entropy(obs, masks, actions)
        policy_loss = self._ppo_surrogate(log_prob, old_log_prob, adv)
        ent = entropy.mean()
        loss = policy_loss - self.config.entropy_coef * ent
        ratio = torch.exp(log_prob - old_log_prob)
        return loss, {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(ent.item()),
            "ratio_mean": float(ratio.mean().item()),
        }

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        steps: list[StepRecord] = []
        advantages: list[float] = []
        for ep in episodes:
            for s in ep.steps:
                steps.append(s)
                advantages.append(ep.advantage)
        if not steps:
            return {"loss": 0.0, "policy_loss": 0.0, "entropy": 0.0, "ratio_mean": 1.0}

        totals = {"loss": 0.0, "policy_loss": 0.0, "entropy": 0.0, "ratio_mean": 0.0}
        for _ in range(self.config.train_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss, stats = self._policy_loss(steps, advantages)
            loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            for k in totals:
                totals[k] += stats[k]
        n = float(self.config.train_epochs)
        return {k: v / n for k, v in totals.items()}

    # ---- training loop ------------------------------------------------------
    def train(
        self,
        num_updates: int | None = None,
        log_every: int | None = None,
        *,
        start_step: int = 0,
        eval_every: int | None = None,
        eval_episodes: int = 5000,
        on_eval: Callable[..., None] | None = None,
    ) -> list[dict[str, Any]]:
        num_updates = num_updates if num_updates is not None else self.config.num_updates
        log_every = log_every if log_every is not None else self.config.log_every
        history: list[dict[str, Any]] = []
        self.policy.train()

        for step in range(start_step + 1, start_step + num_updates + 1):
            all_episodes: list[Episode] = []
            mean_ess = 0.0
            for _ in range(self.config.num_groups):
                group = [self.rollout_episode() for _ in range(self.config.group_size)]
                mean_ess += self._group_advantages(group)
                all_episodes.extend(group)
            mean_ess /= self.config.num_groups

            opt_stats = self.update(all_episodes)
            rewards = [ep.reward for ep in all_episodes]
            log_rewards = [ep.log_reward for ep in all_episodes]
            uniq = len({ep.signature for ep in all_episodes})
            row = {
                "step": step,
                "mean_log_reward": float(np.mean(log_rewards)),
                "max_log_reward": float(np.max(log_rewards)),
                "unique_sig_in_batch": uniq,
                "mean_ess": float(mean_ess),
                **opt_stats,
            }
            history.append(row)

            if step == 1 or step % log_every == 0:
                print(
                    f"update {step:4d}  logR={row['mean_log_reward']:.3f}  "
                    f"maxlogR={row['max_log_reward']:.3f}  uniq_sig={uniq}/{self._num_topologies}  "
                    f"ESS={row['mean_ess']:.1f}  loss={row['loss']:.4f}  ent={row['entropy']:.3f}"
                )

            if eval_every and (step == 1 or step % eval_every == 0):
                from eval_sampling import fit_sampling_metrics

                metrics = fit_sampling_metrics(self, episodes=eval_episodes)
                row["eval_sig_hit"] = metrics.signatures_hit
                row["eval_r2_marginal"] = metrics.r2_marginal
                row["eval_r2_ips"] = metrics.r2_ips
                row["eval_logq_slope"] = metrics.logq_slope
                print(
                    f"  eval@{step:4d}  hit={metrics.signatures_hit}/{self._num_topologies}  "
                    f"R²(marginal)={metrics.r2_marginal:.4f}  R²(ips-biased)={metrics.r2_ips:.4f}  "
                    f"logq_slope={metrics.logq_slope:.3f}  ({eval_episodes} eps)"
                )
                if on_eval is not None:
                    on_eval(step, metrics, self)

        return history

    # ---- persistence --------------------------------------------------------
    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent_type": "merge_ips_grpo",
                "policy": self.policy.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "num_actions": self.num_actions,
                "update_step": update_step,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str, *, device: str = "cpu", for_training: bool = False) -> "MergeTrainer":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        trainer = cls(payload["config"], device=device)
        trainer.policy.load_state_dict(payload["policy"])
        trainer._loaded_update_step = int(payload.get("update_step", 0))
        trainer.policy.train() if for_training else trainer.policy.eval()
        return trainer

    @property
    def loaded_update_step(self) -> int:
        return getattr(self, "_loaded_update_step", 0)
