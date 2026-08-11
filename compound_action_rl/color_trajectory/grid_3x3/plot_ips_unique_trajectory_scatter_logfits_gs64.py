"""IPS-GRPO only: OLS fits on log-transformed density/reward axes (group_size=64)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
GRID_4X4 = ROOT / "grid_4x4"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GRID_4X4))
sys.path.insert(0, str(HERE))

from plot_unique_trajectory_scatter import (  # noqa: E402
    DATA_DIR,
    DEFAULT_IPS_CHECKPOINT,
    match_unique_trajectories,
)
from catalog import iter_trajectories  # noqa: E402
from grpo import agent_label_from_checkpoint  # noqa: E402
from grid_environment import GridEnv  # noqa: E402
from sampling_comparison import (  # noqa: E402
    LinearFitStats,
    _jitter_rewards,
    fit_linear,
    unique_trajectory_density_points,
)

DEFAULT_PLOT_PATH = DATA_DIR / "ips_grpo_unique_trajectory_scatter_logfits_gs64.png"
DEFAULT_FIT_SUMMARY_PATH = DATA_DIR / "ips_grpo_unique_trajectories_gs64_logfits.txt"
DEFAULT_IPS_INDICES = DATA_DIR / "ips_grpo_sample_1000_gs64_trajectory_indices.npy"
SCATTER_COLOR = "#e17055"


@dataclass(frozen=True)
class TransformSpec:
    name: str
    x_label: str
    y_label: str
    x_transform: str
    y_transform: str


TRANSFORMS = (
    TransformSpec(
        name="log(density) vs reward",
        x_label="Reward",
        y_label="log(Sampling density)",
        x_transform="identity",
        y_transform="log",
    ),
    TransformSpec(
        name="density vs log(reward)",
        x_label="log(Reward)",
        y_label="Sampling density",
        x_transform="log",
        y_transform="identity",
    ),
    TransformSpec(
        name="log(density) vs log(reward)",
        x_label="log(Reward)",
        y_label="log(Sampling density)",
        x_transform="log",
        y_transform="log",
    ),
)


def _apply_transform(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return values
    if mode == "log":
        if np.any(values <= 0):
            raise ValueError("log transform requires strictly positive values.")
        return np.log(values)
    raise ValueError(f"Unknown transform: {mode}")


def plot_transform_fits(
    rewards: np.ndarray,
    densities: np.ndarray,
    *,
    title: str,
    save_path: Path,
    jitter: bool = True,
    seed: int = 0,
    show: bool = False,
) -> list[tuple[str, LinearFitStats]]:
    rng = np.random.default_rng(seed)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    fit_results: list[tuple[str, LinearFitStats]] = []

    for ax, spec in zip(axes, TRANSFORMS):
        x = _apply_transform(rewards, spec.x_transform)
        y = _apply_transform(densities, spec.y_transform)
        fit = fit_linear(x, y)
        fit_results.append((spec.name, fit))

        plot_x = _jitter_rewards(x, rng) if jitter else x
        plot_y = _jitter_rewards(y, rng) if jitter else y
        ax.scatter(plot_x, plot_y, s=14, alpha=0.55, color=SCATTER_COLOR, edgecolors="none")

        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        y_line = fit.slope * x_line + fit.intercept
        ax.plot(
            x_line,
            y_line,
            color="#c0392b",
            linewidth=2.0,
            label=f"OLS R²={fit.r2:.3f}, RMSE={fit.rmse:.4f}, slope={fit.slope:.4f}",
        )
        ax.set_xlabel(spec.x_label)
        ax.set_ylabel(spec.y_label)
        ax.set_title(spec.name)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fit_results


def write_transform_fit_summary(
    path: Path,
    fit_results: list[tuple[str, LinearFitStats]],
    *,
    header: str = "OLS fits on transformed axes",
) -> None:
    lines = [header, ""]
    for name, fit in fit_results:
        lines.extend(
            [
                name,
                f"  slope={fit.slope:.6f}",
                f"  intercept={fit.intercept:.6f}",
                f"  R2={fit.r2:.6f}",
                f"  RMSE={fit.rmse:.6f}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IPS-GRPO scatter with log-transform OLS fits (group_size=64)."
    )
    parser.add_argument("--ips-checkpoint", type=Path, default=DEFAULT_IPS_CHECKPOINT)
    parser.add_argument("--ips-indices", type=Path, default=DEFAULT_IPS_INDICES)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--fit-summary-path", type=Path, default=DEFAULT_FIT_SUMMARY_PATH)
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    env = GridEnv()
    records = list(iter_trajectories(env))
    grid_lookup = {r.final_grid: r.index for r in records}
    reward_by_index = {r.index: r.reward for r in records}

    label = f"{agent_label_from_checkpoint(args.ips_checkpoint)} (group_size=64)"
    stats = match_unique_trajectories(
        label=label,
        checkpoint=args.ips_checkpoint,
        env=env,
        grid_lookup=grid_lookup,
        reward_by_index=reward_by_index,
        episodes=args.episodes,
        greedy=args.greedy,
        index_path=args.ips_indices,
    )
    rewards, densities, _indices = unique_trajectory_density_points(stats, reward_by_index)
    print(
        f"{label}: {stats.unique_trajectories_hit}/{stats.enumerated_count} unique trajectories "
        f"({stats.unmatched_episodes} unmatched of {stats.episodes_sampled})"
    )

    title = (
        f"IPS-GRPO transformed OLS fits "
        f"({env.grid_size}x{env.grid_size}, group_size=64, {args.episodes} episodes)"
    )
    fit_results = plot_transform_fits(
        rewards,
        densities,
        title=title,
        save_path=args.plot_path,
        jitter=not args.no_jitter,
        show=args.show,
    )
    write_transform_fit_summary(args.fit_summary_path, fit_results)
    print(f"Saved plot: {args.plot_path}")
    print(f"Saved fit summary: {args.fit_summary_path}")
    for name, fit in fit_results:
        print(f"{name}: OLS R²={fit.r2:.4f}  RMSE={fit.rmse:.6f}  slope={fit.slope:.6f}")


if __name__ == "__main__":
    main()
