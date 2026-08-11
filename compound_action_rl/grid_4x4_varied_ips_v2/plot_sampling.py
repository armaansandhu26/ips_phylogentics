"""Plot sampling scatter + enumeration-free log-log diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import COLOR_PROFILES
from eval_sampling import sample_log_prob_reward, sample_trajectory_indices
from grid_paths import iter_trajectories, make_env
from grid_grpo import GRPOTrainer
from ips_grpo import IPSGRPOTrainer
from run_output import RunPaths, new_run_dir


@dataclass
class SamplingPlotResult:
    episodes: int
    trajectories_hit: int
    r2: float
    slope: float
    rmse: float
    mean_return: float
    log_r2: float
    log_slope: float

    def to_dict(self) -> dict:
        return asdict(self)


def ideal_density(records) -> dict[int, float]:
    total = sum(r.reward for r in records)
    return {r.index: r.reward / total for r in records}


def plot_sampling_scatter(
    trainer: GRPOTrainer,
    *,
    out_path: Path,
    episodes: int = 5000,
    title: str | None = None,
) -> SamplingPlotResult:
    n_traj = trainer._num_trajectories
    env = make_env(**trainer.config.profile_kwargs())
    records = list(iter_trajectories(env))
    reward_by_index = {r.index: r.reward for r in records}
    ideal = ideal_density(records)

    rows = sample_trajectory_indices(trainer, episodes)
    counts = Counter(idx for idx, _ in rows if idx >= 0)
    hit = len(counts)
    returns = [r for _, r in rows]

    x_ideal = [reward_by_index[i] for i in sorted(reward_by_index)]
    y_ideal = [ideal[i] for i in sorted(reward_by_index)]
    x_s = np.array([reward_by_index[i] for i in sorted(counts)])
    y_s = np.array([counts[i] / episodes for i in sorted(counts)])

    if x_s.size >= 2:
        slope, intercept = np.polyfit(x_s, y_s, 1)
        pred = slope * x_s + intercept
        ss_res = float(np.sum((y_s - pred) ** 2))
        ss_tot = float(np.sum((y_s - y_s.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
        rmse = float(np.sqrt(ss_res / x_s.size))
    else:
        slope, intercept, r2, rmse = 0.0, 0.0, 0.0, 0.0

    log_rows = sample_log_prob_reward(trainer, episodes)
    log_p = np.array([r[0] for r in log_rows])
    log_r = np.array([r[1] for r in log_rows])
    if log_r.size >= 2:
        log_slope, log_intercept = np.polyfit(log_r, log_p, 1)
        log_pred = log_slope * log_r + log_intercept
        ss_res = float(np.sum((log_p - log_pred) ** 2))
        ss_tot = float(np.sum((log_p - log_p.mean()) ** 2))
        log_r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    else:
        log_slope, log_intercept, log_r2 = 0.0, 0.0, 0.0

    n_traj = trainer._num_trajectories
    profile = trainer.config.color_profile
    if title is None:
        title = f"IPS-GRPO v2 — profile={profile} ({episodes} eps, {hit}/{n_traj} trajectories)"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(x_ideal, y_ideal, "o-", color="#00b894", linewidth=2, label="Ideal ∝ reward", markersize=5)
    if x_s.size:
        ax.scatter(x_s, y_s, s=70, color="#e17055", edgecolors="white", label=f"IPS-GRPO v2 ({hit}/{n_traj} hit)")
        if x_s.size >= 2:
            xl = np.linspace(float(x_s.min()), float(x_s.max()), 100)
            ax.plot(
                xl,
                slope * xl + intercept,
                ":",
                color="#e17055",
                linewidth=2,
                label=f"OLS R²={r2:.3f} slope={slope:.3f}",
            )
    ax.set_xlabel("Trajectory reward (grid coloring)")
    ax.set_ylabel("Sampling density")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return SamplingPlotResult(
        episodes=episodes,
        trajectories_hit=hit,
        r2=float(r2),
        slope=float(slope),
        rmse=float(rmse),
        mean_return=float(np.mean(returns)),
        log_r2=float(log_r2),
        log_slope=float(log_slope),
    )


def plot_log_scatter(
    trainer: GRPOTrainer,
    *,
    out_path: Path,
    episodes: int = 5000,
    title: str | None = None,
) -> tuple[float, float]:
    """log p_θ(τ) vs log R(τ) — slope≈1 means proportional sampling (no enumeration)."""
    rows = sample_log_prob_reward(trainer, episodes)
    log_p = np.array([r[0] for r in rows])
    log_r = np.array([r[1] for r in rows])

    if log_r.size >= 2:
        slope, intercept = np.polyfit(log_r, log_p, 1)
        pred = slope * log_r + intercept
        ss_res = float(np.sum((log_p - pred) ** 2))
        ss_tot = float(np.sum((log_p - log_p.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    else:
        slope, intercept, r2 = 0.0, 0.0, 0.0

    profile = trainer.config.color_profile
    if title is None:
        title = f"log π vs log R — profile={profile} ({episodes} eps)"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(log_r, log_p, s=30, alpha=0.35, color="#e17055", edgecolors="white", label="Rollouts")
    if log_r.size >= 2:
        xl = np.linspace(float(log_r.min()), float(log_r.max()), 100)
        ax.plot(
            xl,
            slope * xl + intercept,
            ":",
            color="#e17055",
            linewidth=2,
            label=f"OLS R²={r2:.3f} slope={slope:.3f}",
        )
        # Reference: slope=1 line through data centroid
        center_r = float(log_r.mean())
        center_p = float(log_p.mean())
        ref_intercept = center_p - center_r
        ax.plot(
            xl,
            xl + ref_intercept,
            "-",
            color="#00b894",
            linewidth=2,
            label="Ideal slope=1",
        )
    ax.set_xlabel("log R(τ)")
    ax.set_ylabel("log π_θ(τ)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return float(r2), float(slope)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ips_grpo.pt"))
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    trainer = IPSGRPOTrainer.load(args.checkpoint)
    out = args.out or new_run_dir(
        color_profile=trainer.config.color_profile,
        group_size=trainer.config.group_size,
    ) / "sampling_scatter.png"

    result = plot_sampling_scatter(trainer, out_path=out, episodes=args.episodes)
    log_out = out.parent / "log_scatter.png"
    plot_log_scatter(trainer, out_path=log_out, episodes=args.episodes)

    profile = trainer.config.color_profile
    print(f"Profile: {profile} — {COLOR_PROFILES[profile]['description']}")
    n_traj = trainer._num_trajectories
    print(f"Unique trajectories hit: {result.trajectories_hit}/{n_traj}  episodes={result.episodes}")
    print(f"Density OLS: R²={result.r2:.4f}  slope={result.slope:.4f}  RMSE={result.rmse:.6f}")
    print(f"Log-log OLS: R²={result.log_r2:.4f}  slope={result.log_slope:.4f}")
    print(f"Saved: {out}")
    print(f"Saved: {log_out}")


if __name__ == "__main__":
    main()
