#!/usr/bin/env python3
"""Union-catalog reward-proportional ideal sampling grid for 5-taxa baselines."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from grpo_experiments.final_eval_experiment.eval_signature_mass_scatter import (  # noqa: E402
    configure_qhat_y_axis,
)

GRAY = "#777777"
BLUE = "#1976d2"
LOG_SCORE_SHIFT = 3600.0
PAIR_DTYPE = np.dtype([("topology", np.int32), ("score_milli", np.int64)])


@dataclass(frozen=True)
class MethodSpec:
    title: str
    samples: Path
    kind: str = "full_diagnostics"


def parse_args() -> argparse.Namespace:
    comparisons = REPO_ROOT / "grpo_experiments/comparisons/5taxa"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=comparisons / "union_ideal_sampling_grid.png",
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
    return parser.parse_args()


def method_specs(args: argparse.Namespace) -> list[MethodSpec]:
    return [
        MethodSpec("Plain GRPO", args.grpo_samples),
        MethodSpec("Original IPS", args.count_ips_samples),
        MethodSpec("GFlowNet (trajectory balance)", args.gflownet_samples, kind="gflownet"),
        MethodSpec("Learned reverse IPS (new proposed)", args.learned_reverse_samples),
    ]


def load_method(spec: MethodSpec) -> dict:
    with np.load(spec.samples) as payload:
        if spec.kind == "gflownet":
            log_reward = payload["log_reward"].astype(np.float64)
            topology_index = payload["topology_index"].astype(np.int32)
            reward = np.exp(log_reward)
        else:
            reward = payload["log_score"].astype(np.float64)
            topology_index = payload["topology_index"].astype(np.int32)
    return {
        "title": spec.title,
        "reward": reward,
        "topology_index": topology_index,
        "n_samples": len(reward),
        "samples_path": str(spec.samples),
    }


def pair_array(topology_index: np.ndarray, reward: np.ndarray) -> np.ndarray:
    pairs = np.empty(len(reward), dtype=PAIR_DTYPE)
    pairs["topology"] = topology_index.astype(np.int32)
    pairs["score_milli"] = np.rint(reward.astype(np.float64) * 1000.0).astype(np.int64)
    return pairs


def build_union_catalog(methods: list[dict]) -> dict:
    all_pairs = np.concatenate(
        [pair_array(m["topology_index"], m["reward"]) for m in methods]
    )
    all_rewards = np.concatenate([m["reward"].astype(np.float64) for m in methods])

    order = np.argsort(all_pairs, kind="mergesort")
    sorted_pairs = all_pairs[order]
    sorted_rewards = all_rewards[order]

    unique_starts = np.flatnonzero(
        np.concatenate([[True], sorted_pairs[1:] != sorted_pairs[:-1]])
    )
    unique_ends = np.concatenate([unique_starts[1:], [len(sorted_pairs)]])
    union_pairs = sorted_pairs[unique_starts]
    union_rewards = sorted_rewards[unique_starts]

    for start, end in zip(unique_starts, unique_ends, strict=True):
        if end - start > 1:
            segment = sorted_rewards[start:end]
            if float(segment.max() - segment.min()) > 1e-3:
                raise ValueError(
                    "conflicting reward for signature "
                    f"{sorted_pairs[start]}: {segment.min()} vs {segment.max()}"
                )

    reward_sum = float(union_rewards.sum())
    ideal_q = union_rewards / reward_sum
    return {
        "union_pairs": union_pairs,
        "signature_reward": union_rewards,
        "ideal_q": ideal_q,
        "reward_sum": reward_sum,
        "ideal_slope": 1.0 / reward_sum,
        "n_signatures": len(union_pairs),
    }


def model_mass_on_union(method: dict, union: dict) -> tuple[np.ndarray, int]:
    pairs = pair_array(method["topology_index"], method["reward"])
    union_pairs = union["union_pairs"]
    indices = np.searchsorted(union_pairs, pairs)
    if not np.all(union_pairs[indices] == pairs):
        raise RuntimeError(f"method {method['title']} contains signatures outside union")
    counts = np.bincount(indices, minlength=len(union_pairs)).astype(np.float64)
    model_q = counts / float(method["n_samples"])
    return model_q, int(np.count_nonzero(counts))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def pearson_on_support(model_q: np.ndarray, ideal_q: np.ndarray) -> float:
    mask = model_q > 0.0
    if mask.sum() < 2:
        return float("nan")
    x = model_q[mask]
    y = ideal_q[mask]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def panel_payload(method: dict, union: dict) -> dict:
    model_q, n_observed = model_mass_on_union(method, union)
    return {
        **method,
        "model_q": model_q,
        "ideal_q": union["ideal_q"],
        "signature_reward": union["signature_reward"],
        "reward_sum": union["reward_sum"],
        "ideal_slope": union["ideal_slope"],
        "n_union_signatures": union["n_signatures"],
        "n_observed_signatures": n_observed,
        "tv_vs_union_ideal": total_variation(model_q, union["ideal_q"]),
        "pearson_vs_union_ideal": pearson_on_support(model_q, union["ideal_q"]),
    }


def shared_axis_limits(union: dict, panels: list[dict]) -> tuple[float, float, float, float]:
    x_min = float(union["signature_reward"].min())
    x_max = float(union["signature_reward"].max())
    y_max = float(union["ideal_q"].max())
    for panel in panels:
        if panel["n_observed_signatures"] < 100:
            continue
        observed = panel["model_q"] > 0.0
        if not np.any(observed):
            continue
        qs = panel["model_q"][observed]
        y_max = max(y_max, float(np.max(qs)), float(np.percentile(qs, 99.9)))
    x_pad = 0.03 * (x_max - x_min if x_max > x_min else max(x_max, 1.0))
    y_pad = 0.03 * y_max if y_max > 0.0 else 1e-12
    return x_min - x_pad, x_max + x_pad, 0.0, y_max + y_pad


def plot_panel(
    ax: plt.Axes,
    panel: dict,
    *,
    x_min: float,
    x_max: float,
    ideal_line_in_legend: bool,
) -> None:
    x = panel["signature_reward"]
    ideal = panel["ideal_q"]
    model = panel["model_q"]
    observed = model > 0.0
    order = np.argsort(x)

    ax.scatter(
        x[order],
        ideal[order],
        marker="o",
        s=2.2,
        linewidths=0,
        color=GRAY,
        alpha=0.18,
        rasterized=True,
        label="union ideal q* ∝ R(x)",
        zorder=2,
    )
    ax.scatter(
        x[observed],
        model[observed],
        marker="x",
        s=3.0,
        linewidths=0.28,
        color=BLUE,
        alpha=0.20,
        rasterized=True,
        label="checkpoint samples",
        zorder=3,
    )
    line_x = np.linspace(x_min, x_max, 200)
    line_y = line_x / panel["reward_sum"]
    line_label = (
        f"ideal: q* = R / Σ_union R (slope={panel['ideal_slope']:.4g})"
        if ideal_line_in_legend
        else "_nolegend_"
    )
    ax.plot(
        line_x,
        line_y,
        linestyle="--",
        color="0.25",
        linewidth=1.3,
        alpha=0.95,
        label=line_label,
        zorder=1,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_title(panel["title"], fontsize=12)
    ax.set_xlabel(rf"Terminal reward: $R(x)={LOG_SCORE_SHIFT:g}+\log L(x)$")
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(frameon=False, loc="upper left", fontsize=7, markerscale=4)
    pearson = panel["pearson_vs_union_ideal"]
    pearson_text = "nan" if not np.isfinite(pearson) else f"{pearson:.4f}"
    ax.text(
        0.98,
        0.02,
        (
            f"{panel['n_samples']:,} samples\n"
            f"{panel['n_observed_signatures']:,} seen / "
            f"{panel['n_union_signatures']:,} union\n"
            f"TV={panel['tv_vs_union_ideal']:.4f}\n"
            f"Pearson r={pearson_text}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
    )


def pairwise_overlap(methods: list[dict]) -> dict[str, int]:
    key_sets = []
    for method in methods:
        pairs = pair_array(method["topology_index"], method["reward"])
        _, unique_idx = np.unique(pairs, return_index=True)
        key_sets.append(set(map(tuple, pairs[unique_idx].tolist())))
    titles = [m["title"] for m in methods]
    overlap: dict[str, int] = {}
    for i in range(len(key_sets)):
        for j in range(i + 1, len(key_sets)):
            overlap[f"{titles[i]} ∩ {titles[j]}"] = len(key_sets[i] & key_sets[j])
    return overlap


def main() -> None:
    args = parse_args()
    specs = method_specs(args)
    methods = [load_method(spec) for spec in specs]
    union = build_union_catalog(methods)
    panels = [panel_payload(method, union) for method in methods]
    x_min, x_max, y_min, y_max = shared_axis_limits(union, panels)

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=220, constrained_layout=True)
    for ax, panel in zip(axes.ravel(), panels, strict=True):
        plot_panel(
            ax,
            panel,
            x_min=x_min,
            x_max=x_max,
            ideal_line_in_legend=(ax is axes.ravel()[0]),
        )

    for ax in axes.ravel():
        ax.set_ylim(y_min, y_max)
    axes[0, 0].set_ylabel("Per-signature empirical mass ($q_{\\hat{}}$)")
    axes[1, 0].set_ylabel("Per-signature empirical mass ($q_{\\hat{}}$)")
    for ax in axes[1, :].ravel():
        configure_qhat_y_axis(ax, samples=panels[0]["n_samples"])

    fig.suptitle(
        "5-taxa union-catalog ideal sampling (1M samples / method)",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "output": str(args.output),
        "plot_kind": "union_catalog_reward_proportional_qhat_vs_reward",
        "ideal_definition": (
            "Union over all methods' observed signatures. "
            "q*(sigma) = R(sigma) / sum_{sigma in union} R(sigma)."
        ),
        "union_catalog": {
            "n_signatures": union["n_signatures"],
            "reward_sum": union["reward_sum"],
            "ideal_slope": union["ideal_slope"],
            "pairwise_signature_overlap": pairwise_overlap(methods),
        },
        "shared_axis": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "panels": {
            panel["title"]: {
                "samples_path": panel["samples_path"],
                "n_samples": panel["n_samples"],
                "n_observed_signatures": panel["n_observed_signatures"],
                "tv_vs_union_ideal": panel["tv_vs_union_ideal"],
                "pearson_vs_union_ideal": panel["pearson_vs_union_ideal"],
            }
            for panel in panels
        },
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
