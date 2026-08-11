#!/usr/bin/env python3
"""5-taxa pathwise reward comparison using one pooled, fitted normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plot_5taxa_sampling_comparison_grid as base  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_DIR = REPO_ROOT / "grpo_experiments/comparisons/5taxa"
GRAY = "#777777"
BLUE = "#1976d2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=COMPARISON_DIR / "sampling_comparison_shared_z_grid.png",
    )
    parser.add_argument(
        "--grpo-samples",
        type=Path,
        default=REPO_ROOT
        / "grpo_experiments/learned_reverse_runs/20260802_174242_grpo_5taxa_full_sig_b4096_grpo/sampled_full_diagnostics_1000000.npz",
    )
    parser.add_argument(
        "--count-ips-samples",
        type=Path,
        default=REPO_ROOT
        / "grpo_experiments/learned_reverse_runs/20260802_174242_count_ips_5taxa_full_sig_b4096_ips_grpo/sampled_full_diagnostics_1000000.npz",
    )
    parser.add_argument(
        "--gflownet-samples",
        type=Path,
        default=REPO_ROOT
        / "og_code/experiments/full_model/20260703_172421_phylgfn_logreward_g4096_1m_full_replay_op3277_r819_rb4096/plots/reward_probability_eval_1000000/og_gflownet_reward_probability_samples.npz",
    )
    parser.add_argument(
        "--learned-reverse-samples",
        type=Path,
        default=REPO_ROOT
        / "grpo_experiments/learned_reverse_runs/20260730_160341_learned_reverse_5taxa_mlp_shifted_linear_b4096_learned_reverse_ips_grpo/sampled_full_diagnostics_1000000.npz",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scatter-points", type=int, default=0)
    return parser.parse_args()


def panel_specs(args: argparse.Namespace) -> list[base.PanelSpec]:
    return [
        base.PanelSpec(
            title="Plain GRPO",
            samples=args.grpo_samples,
            estimator_label=r"$P_F(\tau)/P_B(\tau)$",
            marker_scale=6.0,
        ),
        base.PanelSpec(
            title="Original IPS",
            samples=args.count_ips_samples,
            estimator_label=r"$P_F(\tau)/P_B(\tau)$",
        ),
        base.PanelSpec(
            title="GFlowNet (trajectory balance)",
            samples=args.gflownet_samples,
            estimator_label=r"$P_F(\tau)/P_B(\tau)$",
            kind="gflownet",
        ),
        base.PanelSpec(
            title="Learned reverse IPS (new proposed)",
            samples=args.learned_reverse_samples,
            estimator_label=r"$P_F(\tau)/q_\phi(\tau|x)$",
        ),
    ]


def fit_shared_log_z(
    loaded: list[tuple[np.ndarray, np.ndarray, int, base.PanelSpec]],
) -> float:
    """Fit one intercept with fixed unit slope in log space across all samples."""
    residual_sum = sum(
        float(np.sum(log_reward - log_probability, dtype=np.float64))
        for log_probability, log_reward, _, _ in loaded
    )
    sample_count = sum(len(log_probability) for log_probability, _, _, _ in loaded)
    return residual_sum / sample_count


def plot_panel(
    ax: plt.Axes,
    *,
    spec: base.PanelSpec,
    log_model_probability: np.ndarray,
    log_target_reward: np.ndarray,
    unique_signatures: int,
    axis_spec: dict[str, float],
    shared_log_z: float,
    probability_limits: tuple[float, float],
    seed: int,
    scatter_points: int,
) -> dict[str, float]:
    selected = base.select_points(len(log_model_probability), scatter_points, seed)
    model_probability = np.exp(log_model_probability[selected])
    target_reward = np.exp(log_target_reward[selected])
    pearson = (
        float("nan")
        if unique_signatures < 2
        else base.pearson_vs_ideal_sampling_linear(
            log_model_probability,
            log_target_reward,
            shared_log_z,
        )
    )
    log_residual = log_model_probability - (log_target_reward - shared_log_z)
    log_rmse = float(np.sqrt(np.mean(np.square(log_residual))))

    marker_size, alpha = base.scatter_style(
        len(selected),
        scale=spec.marker_scale,
    )
    pearson_text = "nan" if not np.isfinite(pearson) else f"{pearson:.4f}"
    ax.scatter(
        target_reward,
        model_probability,
        s=marker_size,
        alpha=alpha,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
        label=(
            f"{spec.estimator_label}\n"
            f"{len(log_model_probability):,} trajectories\n"
            f"{unique_signatures:,} unique signatures\n"
            f"Pearson r(P̂, R)={pearson_text}\n"
            f"log-space RMSE={log_rmse:.3f}"
        ),
    )

    partition = float(np.exp(shared_log_z))
    line_x = np.linspace(axis_spec["reward_min"], axis_spec["reward_max"], 200)
    ax.plot(
        line_x,
        line_x / partition,
        color=GRAY,
        linestyle="--",
        linewidth=1.3,
        label=(
            r"Common fitted reference: $\hat P=R/\hat Z_{\rm shared}$"
            f"\n(log $\\hat Z_{{\\rm shared}}$={shared_log_z:.3f})"
        ),
    )
    ax.set_xlim(axis_spec["reward_min"], axis_spec["reward_max"])
    ax.set_yscale("log")
    ax.set_ylim(*probability_limits)
    ax.set_title(spec.title, fontsize=13)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best", fontsize=8)
    return {"pearson_r_probability_vs_reward": pearson, "log_space_rmse": log_rmse}


def main() -> None:
    args = parse_args()
    specs = panel_specs(args)
    loaded = [(*base.load_panel(spec), spec) for spec in specs]
    shared_log_z = fit_shared_log_z(loaded)

    axis_spec = base.shared_reward_axis_bounds()
    all_reward = np.concatenate([np.exp(item[1]) for item in loaded])
    all_log_reward = np.concatenate([item[1] for item in loaded])
    axis_spec = base.merge_reward_axis_bounds(
        axis_spec,
        reward=all_reward,
        log_reward=all_log_reward,
    )
    partition = float(np.exp(shared_log_z))
    probability_min = min(
        *(float(np.exp(item[0]).min()) for item in loaded),
        axis_spec["reward_min"] / partition,
    )
    probability_max = max(
        *(float(np.exp(item[0]).max()) for item in loaded),
        axis_spec["reward_max"] / partition,
    )
    probability_limits = (probability_min / 2.0, probability_max * 2.0)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 16),
        dpi=220,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    metrics: dict = {
        "fit_definition": (
            "One shared log(Z) minimizing pooled squared error between "
            "log implied probability and log reward - log(Z)."
        ),
        "shared_log_partition": shared_log_z,
        "shared_partition": float(np.exp(shared_log_z)),
        "reward_axis": {
            "min": axis_spec["reward_min"],
            "max": axis_spec["reward_max"],
        },
        "panels": {},
    }

    for ax, (log_probability, log_reward, unique_signatures, spec) in zip(
        axes.ravel(),
        loaded,
        strict=True,
    ):
        panel_metrics = plot_panel(
            ax,
            spec=spec,
            log_model_probability=log_probability,
            log_target_reward=log_reward,
            unique_signatures=unique_signatures,
            axis_spec=axis_spec,
            shared_log_z=shared_log_z,
            probability_limits=probability_limits,
            seed=args.seed,
            scatter_points=args.scatter_points,
        )
        metrics["panels"][spec.title] = {
            "samples": int(len(log_probability)),
            "unique_signatures": unique_signatures,
            **panel_metrics,
            "samples_file": str(spec.samples),
        }

    fig.supxlabel(r"Terminal reward: $R(x)=3600+\log L(x)$", fontsize=14)
    fig.supylabel("Pathwise implied terminal probability (log scale)", fontsize=14)
    fig.suptitle(
        "5-taxa pathwise reward alignment with one pooled fitted normalization",
        fontsize=16,
        y=1.01,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"shared fitted log Z: {shared_log_z:.9f}")
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
