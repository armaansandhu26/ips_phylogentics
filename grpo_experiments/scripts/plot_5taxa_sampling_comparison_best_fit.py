#!/usr/bin/env python3
"""Compare three 5-taxa methods with per-method linear least-squares fits."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plot_5taxa_sampling_comparison_grid as base  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_DIR = REPO_ROOT / "grpo_experiments/comparisons/5taxa"
BLUE = "#1976d2"
GRAY = "#555555"


@dataclass(frozen=True)
class Method:
    title: str
    samples: Path
    estimator_label: str
    kind: str = "full_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=COMPARISON_DIR / "sampling_comparison_best_fit_grid.png",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scatter-points", type=int, default=0)
    return parser.parse_args()


def methods() -> list[Method]:
    return [
        Method(
            title="Plain GRPO",
            samples=REPO_ROOT
            / "grpo_experiments/learned_reverse_runs/20260802_174242_grpo_5taxa_full_sig_b4096_grpo/sampled_full_diagnostics_1000000.npz",
            estimator_label=r"Empirical $\hat q(x)=\mathrm{count}(x)/N$",
            kind="empirical_signature",
        ),
        Method(
            title="Original IPS",
            samples=REPO_ROOT
            / "grpo_experiments/learned_reverse_runs/20260802_174242_count_ips_5taxa_full_sig_b4096_ips_grpo/sampled_full_diagnostics_1000000.npz",
            estimator_label=r"Empirical $\hat q(x)=\mathrm{count}(x)/N$",
            kind="empirical_signature",
        ),
        Method(
            title="GFlowNet (trajectory balance)",
            samples=REPO_ROOT
            / "og_code/experiments/full_model/20260703_172421_phylgfn_logreward_g4096_1m_full_replay_op3277_r819_rb4096/plots/forward_trajectory_eval_1000000/og_gflownet_reward_probability_samples.npz",
            estimator_label=r"$P_F(\tau)/P_B(\tau)$",
            kind="gflownet",
        ),
        Method(
            title="Learned reverse IPS (new proposed)",
            samples=REPO_ROOT
            / "grpo_experiments/learned_reverse_runs/20260730_160341_learned_reverse_5taxa_mlp_shifted_linear_b4096_learned_reverse_ips_grpo/sampled_full_diagnostics_1000000.npz",
            estimator_label=r"$P_F(\tau)/q_\phi(\tau|x)$",
        ),
    ]


def load_empirical_signature_frequency(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    with np.load(path) as payload:
        reward = payload["log_score"].astype(np.float64)
        topology_index = payload["topology_index"].astype(np.int16)

    rounded_reward = np.rint(reward * 1000.0).astype(np.int64)
    signatures = np.empty(
        len(reward),
        dtype=[("topology", np.int16), ("reward_milli", np.int64)],
    )
    signatures["topology"] = topology_index
    signatures["reward_milli"] = rounded_reward
    _, first_indices, counts = np.unique(
        signatures,
        return_index=True,
        return_counts=True,
    )
    frequency = counts.astype(np.float64) / len(reward)
    return reward[first_indices], frequency, len(counts), len(reward)


def load_method(method: Method) -> tuple[np.ndarray, np.ndarray, int, int]:
    if method.kind == "empirical_signature":
        return load_empirical_signature_frequency(method.samples)

    spec = base.PanelSpec(
        title=method.title,
        samples=method.samples,
        estimator_label=method.estimator_label,
        kind=method.kind,
    )
    log_probability, log_reward, unique_signatures = base.load_panel(spec)
    return (
        np.exp(log_reward),
        np.exp(log_probability),
        unique_signatures,
        len(log_probability),
    )


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    centered_x = x - x_mean
    centered_y = y - y_mean
    slope = float(np.dot(centered_x, centered_y) / np.dot(centered_x, centered_x))
    intercept = y_mean - slope * x_mean
    fitted = slope * x + intercept
    residual_sum_squares = float(np.sum(np.square(y - fitted)))
    total_sum_squares = float(np.sum(np.square(centered_y)))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else float("nan")
    )
    pearson = (
        float(np.corrcoef(x, y)[0, 1])
        if np.std(x) > 0.0 and np.std(y) > 0.0
        else float("nan")
    )
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "pearson_r": pearson,
    }


def plot_panel(
    ax: plt.Axes,
    *,
    method: Method,
    reward: np.ndarray,
    probability: np.ndarray,
    sample_count: int,
    unique_signatures: int,
    fit: dict[str, float] | None,
    reward_limits: tuple[float, float],
    seed: int,
    scatter_points: int,
) -> None:
    selected = base.select_points(len(reward), scatter_points, seed)
    marker_size, alpha = base.scatter_style(len(selected))
    ax.scatter(
        reward[selected],
        probability[selected],
        s=marker_size,
        alpha=alpha,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
        label=(
            f"{method.estimator_label}\n"
            f"{sample_count:,} sampled trajectories\n"
            f"{unique_signatures:,} unique signatures"
            + ("\nNo fit: fewer than 2 signatures" if fit is None else "")
        ),
    )

    line_y = np.array([], dtype=np.float64)
    if fit is not None:
        # Do not extrapolate a method's fit beyond the reward range it observed.
        line_x = np.linspace(float(reward.min()), float(reward.max()), 200)
        line_y = fit["slope"] * line_x + fit["intercept"]
        ax.plot(
            line_x,
            line_y,
            color=GRAY,
            linestyle="--",
            linewidth=1.5,
            label=(
                r"OLS best fit: $\hat P=aR+b$"
                f"\n$a={fit['slope']:.3e}$, $b={fit['intercept']:.3e}$"
                f"\n$R^2={fit['r_squared']:.4f}$"
            ),
        )
    ax.set_xlim(*reward_limits)
    y_min = float(probability.min())
    y_max = float(probability.max())
    if line_y.size:
        y_min = min(y_min, float(line_y.min()))
        y_max = max(y_max, float(line_y.max()))
    padding = 0.03 * (y_max - y_min if y_max > y_min else max(abs(y_max), 1e-20))
    lower_limit = 0.0 if method.title == "Original IPS" else y_min - padding
    ax.set_ylim(lower_limit, y_max + padding)
    ax.set_title(method.title, fontsize=13)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best", fontsize=8)


def main() -> None:
    args = parse_args()
    method_specs = methods()
    loaded = [load_method(method) for method in method_specs]
    all_rewards = np.concatenate([item[0] for item in loaded])
    reward_min = float(all_rewards.min())
    reward_max = float(all_rewards.max())
    reward_padding = 0.02 * (reward_max - reward_min)
    reward_limits = (reward_min - reward_padding, reward_max + reward_padding)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 16),
        dpi=220,
        sharex=True,
        constrained_layout=True,
    )
    summary: dict = {
        "fit_definition": (
            "Independent ordinary least-squares fit in displayed linear space: "
            "displayed outcome estimate = slope * reward + intercept."
        ),
        "panels": {},
    }
    for ax, method, (reward, probability, unique_signatures, sample_count) in zip(
        axes.ravel(),
        method_specs,
        loaded,
        strict=True,
    ):
        fit = (
            linear_fit(reward, probability)
            if unique_signatures >= 2
            else None
        )
        plot_panel(
            ax,
            method=method,
            reward=reward,
            probability=probability,
            sample_count=sample_count,
            unique_signatures=unique_signatures,
            fit=fit,
            reward_limits=reward_limits,
            seed=args.seed,
            scatter_points=args.scatter_points,
        )
        summary["panels"][method.title] = {
            "samples": sample_count,
            "unique_signatures": unique_signatures,
            "quantity": (
                "empirical_signature_frequency"
                if method.kind == "empirical_signature"
                else "reverse_corrected_pathwise_estimate"
            ),
            "fit": fit,
            "samples_file": str(method.samples),
        }

    fig.supxlabel(r"Terminal reward: $R(x)=3600+\log L(x)$", fontsize=14)
    fig.supylabel("Outcome probability estimate", fontsize=14)
    fig.suptitle(
        "5-taxa outcome probability versus reward",
        fontsize=16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
