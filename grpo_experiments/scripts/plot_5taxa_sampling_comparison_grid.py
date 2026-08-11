#!/usr/bin/env python3
"""2x2 grid of terminal probability vs reward for 5-taxa full-model baselines."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from reward_probability_plot_reference import (  # noqa: E402
    ideal_line_label,
    linear_probability_limits,
    merge_reward_axis_bounds,
    pearson_vs_ideal_sampling_linear,
    shared_reward_axis_bounds,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAY = "#777777"
BLUE = "#1976d2"


@dataclass(frozen=True)
class PanelSpec:
    title: str
    samples: Path
    estimator_label: str
    show_ideal_log_z: bool = False
    marker_scale: float = 1.0
    kind: str = "full_diagnostics"
    checkpoint_log_partition: float | None = None


def parse_args() -> argparse.Namespace:
    comparisons = REPO_ROOT / "grpo_experiments/comparisons/5taxa"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=comparisons / "sampling_comparison_grid.png",
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
    parser.add_argument(
        "--gflownet-checkpoint-log-partition",
        type=float,
        default=35.66216689774323,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scatter-points", type=int, default=0)
    return parser.parse_args()


def panel_specs(args: argparse.Namespace) -> list[PanelSpec]:
    return [
        PanelSpec(
            title="Plain GRPO",
            samples=args.grpo_samples,
            estimator_label=r"$P_F(\tau)/P_B(\tau)$",
            marker_scale=6.0,
        ),
        PanelSpec(
            title="Original IPS",
            samples=args.count_ips_samples,
            estimator_label=r"$P_F(\tau)/P_B(\tau)$",
        ),
        PanelSpec(
            title="GFlowNet (trajectory balance)",
            samples=args.gflownet_samples,
            estimator_label=r"$P_F(\tau)/P_B(\tau)$",
            show_ideal_log_z=True,
            kind="gflownet",
            checkpoint_log_partition=args.gflownet_checkpoint_log_partition,
        ),
        PanelSpec(
            title="Learned reverse IPS (new proposed)",
            samples=args.learned_reverse_samples,
            estimator_label=r"$P_F(\tau)/q_\phi(\tau|x)$",
        ),
    ]


def load_full_diagnostics(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path) as payload:
        log_score = payload["log_score"].astype(np.float64)
        log_pf = payload["log_pf"].astype(np.float64)
        log_q_reverse = payload["log_q_reverse"].astype(np.float64)
        topology_index = payload["topology_index"].astype(np.int16)

    log_target_reward = np.log(log_score)
    log_model_probability = log_pf - log_q_reverse
    rounded_score = np.rint(log_score * 1000.0).astype(np.int64)
    signature_pairs = np.empty(
        len(log_score),
        dtype=[("topology", np.int16), ("score_milli", np.int64)],
    )
    signature_pairs["topology"] = topology_index
    signature_pairs["score_milli"] = rounded_score
    unique_signatures = int(np.unique(signature_pairs).size)
    return log_model_probability, log_target_reward, unique_signatures


def load_gflownet(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path) as payload:
        log_model_probability = payload["log_probability"].astype(np.float64)
        log_target_reward = payload["log_reward"].astype(np.float64)
        topology_index = payload["topology_index"].astype(np.int16)
        raw_log_likelihood = payload["raw_log_likelihood"].astype(np.float64)

    score_milli = np.rint(raw_log_likelihood * 1000.0).astype(np.int64)
    signature_pairs = np.empty(
        len(log_model_probability),
        dtype=[("topology", np.int16), ("score_milli", np.int64)],
    )
    signature_pairs["topology"] = topology_index
    signature_pairs["score_milli"] = score_milli
    unique_signatures = int(np.unique(signature_pairs).size)
    return log_model_probability, log_target_reward, unique_signatures


def load_panel(spec: PanelSpec) -> tuple[np.ndarray, np.ndarray, int]:
    if spec.kind == "gflownet":
        return load_gflownet(spec.samples)
    return load_full_diagnostics(spec.samples)


def select_points(size: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or size <= maximum:
        return np.arange(size)
    return np.random.default_rng(seed).choice(size, size=maximum, replace=False)


def scatter_style(displayed_points: int, *, scale: float = 1.0) -> tuple[float, float]:
    if displayed_points >= 500_000:
        return 8.0 * scale, 0.10
    if displayed_points >= 100_000:
        return 12.0 * scale, 0.15
    if displayed_points <= 1:
        return 120.0 * scale, 0.95
    return 20.0 * scale, 0.30


def plot_panel(
    ax: plt.Axes,
    *,
    spec: PanelSpec,
    log_model_probability: np.ndarray,
    log_target_reward: np.ndarray,
    unique_signatures: int,
    axis_spec: dict[str, float],
    seed: int,
    scatter_points: int,
) -> float:
    selected = select_points(len(log_model_probability), scatter_points, seed)
    selected_log_probability = log_model_probability[selected]
    selected_log_reward = log_target_reward[selected]
    model_probability = np.exp(selected_log_probability)
    target_reward = np.exp(selected_log_reward)

    estimated_log_partition = float(
        np.mean(log_target_reward - log_model_probability)
    )
    if spec.checkpoint_log_partition is not None:
        line_log_partition = spec.checkpoint_log_partition
    else:
        line_log_partition = estimated_log_partition
    partition = float(np.exp(line_log_partition))
    pearson = pearson_vs_ideal_sampling_linear(
        log_model_probability,
        log_target_reward,
        line_log_partition,
    )

    marker_size, alpha = scatter_style(
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
            f"Pearson r vs ideal={pearson_text}"
        ),
    )

    line_x = np.linspace(axis_spec["reward_min"], axis_spec["reward_max"], 200)
    ax.plot(
        line_x,
        line_x / partition,
        color=GRAY,
        linestyle="--",
        linewidth=1.3,
        label=ideal_line_label(line_log_partition, show_log_z=spec.show_ideal_log_z),
    )
    ax.set_xlim(axis_spec["reward_min"], axis_spec["reward_max"])
    y_min, y_max = linear_probability_limits(model_probability, line_x, partition)
    ax.set_ylim(y_min, y_max)
    ax.set_title(spec.title, fontsize=13)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best", fontsize=9)
    return pearson


def main() -> None:
    args = parse_args()
    panels = panel_specs(args)
    loaded = [(*load_panel(spec), spec) for spec in panels]

    axis_spec = shared_reward_axis_bounds()
    all_reward = np.concatenate([np.exp(item[1]) for item in loaded])
    all_log_reward = np.concatenate([item[1] for item in loaded])
    axis_spec = merge_reward_axis_bounds(
        axis_spec,
        reward=all_reward,
        log_reward=all_log_reward,
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=220, constrained_layout=True)
    metrics: dict[str, float | int | str] = {
        "reward_axis": {
            "min": axis_spec["reward_min"],
            "max": axis_spec["reward_max"],
        },
        "panels": {},
    }

    for ax, (log_model_probability, log_target_reward, unique_signatures, spec) in zip(
        axes.ravel(),
        loaded,
        strict=True,
    ):
        pearson = plot_panel(
            ax,
            spec=spec,
            log_model_probability=log_model_probability,
            log_target_reward=log_target_reward,
            unique_signatures=unique_signatures,
            axis_spec=axis_spec,
            seed=args.seed,
            scatter_points=args.scatter_points,
        )
        metrics["panels"][spec.title] = {
            "samples": int(len(log_model_probability)),
            "unique_signatures": unique_signatures,
            "pearson_r_vs_ideal": pearson,
            "samples_file": str(spec.samples),
        }

    fig.supxlabel(r"Terminal reward: $R(x)=3600+\log L(x)$", fontsize=14)
    fig.supylabel("Pathwise implied terminal probability", fontsize=14)
    fig.suptitle(
        "5-taxa full-model sampling: terminal probability vs reward",
        fontsize=16,
        y=1.01,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
