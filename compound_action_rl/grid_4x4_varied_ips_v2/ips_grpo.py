"""IPS-GRPO v2: exact trajectory propensities with SNIPS self-normalization."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from grid_grpo import Episode, GRPOTrainer, effective_sample_size


class IPSGRPOTrainer(GRPOTrainer):
    def _exact_weights(self, episodes: list[Episode]) -> np.ndarray:
        log_p = np.array([ep.log_prob_joint for ep in episodes], dtype=np.float64)
        weights = np.exp(-log_p)
        if self.config.max_inverse_weight is not None:
            weights = np.minimum(weights, self.config.max_inverse_weight)
        return weights

    def _count_weights(self, episodes: list[Episode], group_size: int) -> np.ndarray:
        counts = Counter(ep.outcome for ep in episodes)
        return np.array(
            [
                1.0 / max(counts[ep.outcome] / group_size, self.config.p_eps)
                for ep in episodes
            ],
            dtype=np.float64,
        )

    def _trajectory_weights(self, episodes: list[Episode]) -> np.ndarray:
        if self.config.propensity_mode == "count":
            return self._count_weights(episodes, self.config.group_size)
        return self._exact_weights(episodes)

    def train(
        self,
        num_updates: int | None = None,
        log_every: int | None = None,
        *,
        start_step: int = 0,
        eval_every: int | None = None,
        eval_episodes: int = 2000,
        on_eval: Callable[..., None] | None = None,
    ) -> list[dict[str, Any]]:
        num_updates = num_updates if num_updates is not None else self.config.num_updates
        log_every = log_every if log_every is not None else self.config.log_every
        history: list[dict[str, Any]] = []

        self.path_policy.train()
        self.color_policy.train()

        eval_records = None
        eval_ideal: dict[int, float] | None = None
        eval_rewards: dict[int, float] | None = None
        if eval_every:
            from grid_paths import iter_trajectories

            eval_records = list(iter_trajectories(self.env))
            total = sum(r.reward for r in eval_records)
            eval_ideal = {r.index: r.reward / total for r in eval_records}
            eval_rewards = {r.index: r.reward for r in eval_records}

        for step in range(start_step + 1, start_step + num_updates + 1):
            all_episodes: list[Episode] = []
            mean_ess = 0.0
            mean_inv_weight = 0.0

            for _ in range(self.config.num_groups):
                group = [self.rollout_episode() for _ in range(self.config.group_size)]
                raw_weights = self._trajectory_weights(group)
                mean_inv_weight += float(raw_weights.mean())
                mean_ess += effective_sample_size(raw_weights)
                self._group_advantages(group)
                all_episodes.extend(group)
            mean_ess /= self.config.num_groups
            mean_inv_weight /= self.config.num_groups

            opt_stats = self.update(all_episodes)
            returns = [ep.return_ for ep in all_episodes]
            uniq = len({ep.trajectory_index for ep in all_episodes if ep.trajectory_index >= 0})
            row = {
                "step": step,
                "mean_return": float(np.mean(returns)),
                "max_return": float(np.max(returns)),
                "unique_traj_in_batch": uniq,
                "mean_ess": float(mean_ess),
                "mean_inv_weight": float(mean_inv_weight),
                **opt_stats,
            }
            history.append(row)

            if step == 1 or step % log_every == 0:
                print(
                    f"update {step:4d}  return={row['mean_return']:.3f}  "
                    f"max={row['max_return']:.3f}  uniq_traj={uniq}/{self._num_trajectories}  "
                    f"ESS={row['mean_ess']:.1f}  inv_w={row['mean_inv_weight']:.2f}  "
                    f"loss={row['loss']:.4f}  entropy={row['entropy']:.3f}"
                )

            if eval_every and eval_records is not None and eval_ideal is not None:
                if step == 1 or step % eval_every == 0:
                    from eval_sampling import fit_log_log_metrics, fit_sampling_metrics

                    metrics = fit_sampling_metrics(
                        self,
                        episodes=eval_episodes,
                        reward_by_index=eval_rewards,
                        ideal_density=eval_ideal,
                    )
                    log_metrics = fit_log_log_metrics(self, episodes=eval_episodes)
                    row["eval_traj_hit"] = metrics.trajectories_hit
                    row["eval_r2"] = metrics.r2
                    row["eval_mean_return"] = metrics.mean_return
                    row["eval_log_r2"] = log_metrics.log_r2
                    row["eval_log_slope"] = log_metrics.log_slope
                    print(
                        f"  eval@{step:4d}  hit={metrics.trajectories_hit}/{self._num_trajectories}  "
                        f"R²={metrics.r2:.4f}  logR²={log_metrics.log_r2:.4f}  "
                        f"log_slope={log_metrics.log_slope:.3f}  "
                        f"mean_ret={metrics.mean_return:.3f}  ({eval_episodes} eps)"
                    )
                    if on_eval is not None:
                        on_eval(step, metrics, self)

        return history

    def save(self, path: Path | str, *, update_step: int = 0) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        import torch

        torch.save(
            {
                "agent_type": "ips_grpo_v2",
                "path_policy": self.path_policy.state_dict(),
                "color_policy": self.color_policy.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "update_step": update_step,
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: str = "cpu",
        for_training: bool = False,
    ) -> "IPSGRPOTrainer":
        payload = __import__("torch").load(Path(path), map_location=device, weights_only=False)
        trainer = cls(payload["config"], device=device)
        trainer.path_policy.load_state_dict(payload["path_policy"])
        trainer.color_policy.load_state_dict(payload["color_policy"])
        trainer._loaded_update_step = int(payload.get("update_step", 0))
        if for_training:
            trainer.path_policy.train()
            trainer.color_policy.train()
        else:
            trainer.path_policy.eval()
            trainer.color_policy.eval()
        return trainer
