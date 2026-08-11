#!/usr/bin/env python3
"""Compare 5-taxa checkpoint topology mass against exact 105-topology ground truth."""

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

GRAY = "#666666"
GREEN = "#2e7d32"
BLUE = "#1976d2"
LOG_SCORE_SHIFT = 3600.0


@dataclass(frozen=True)
class MethodSpec:
    title: str
    samples: Path
    kind: str = "full_diagnostics"


def parse_args() -> argparse.Namespace:
    comparisons = REPO_ROOT / "grpo_experiments/comparisons/5taxa"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-reference",
        type=Path,
        default=comparisons
        / "exact_topology_reference"
        / "exact_topology_reference_shifted_linear_1000000_seed0.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=comparisons / "topology_gt_comparison_grid.png",
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


def load_gt_reference(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["topologies"]
    topology_ids = [str(row["topology_id"]) for row in rows]
    topology_id_to_index = {topology_id: idx for idx, topology_id in enumerate(topology_ids)}
    reward = np.asarray(
        [float(np.exp(row["target_log_reward"])) for row in rows],
        dtype=np.float64,
    )
    exact_q = np.asarray([float(row["expected_frequency"]) for row in rows], dtype=np.float64)
    gt_empirical_q = np.asarray(
        [float(row["empirical_frequency"]) for row in rows],
        dtype=np.float64,
    )
    log_likelihood = np.asarray(
        [float(row["terminal_log_likelihood"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "metadata": payload["metadata"],
        "summary": payload["summary"],
        "topology_ids": topology_ids,
        "topology_id_to_index": topology_id_to_index,
        "reward": reward,
        "exact_q": exact_q,
        "gt_empirical_q": gt_empirical_q,
        "log_likelihood": log_likelihood,
        "reward_sum": float(reward.sum()),
    }


def topology_mass_from_npz(path: Path, gt: dict) -> dict:
    with np.load(path) as payload:
        topology_index = payload["topology_index"].astype(np.int64)
        topology_ids = [str(item) for item in payload["topology_ids"]]

    counts = np.zeros(len(gt["topology_ids"]), dtype=np.float64)
    for local_index in topology_index:
        topology_id = topology_ids[int(local_index)]
        if topology_id not in gt["topology_id_to_index"]:
            raise KeyError(f"topology {topology_id} missing from exact GT reference")
        gt_index = gt["topology_id_to_index"][topology_id]
        counts[gt_index] += 1.0
    n_samples = len(topology_index)
    model_q = counts / float(n_samples)
    return {
        "n_samples": n_samples,
        "model_q": model_q,
        "n_observed_topologies": int(np.count_nonzero(counts)),
    }


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def pearson_on_observed(model_q: np.ndarray, target_q: np.ndarray) -> float:
    mask = model_q > 0.0
    if mask.sum() < 2:
        return float("nan")
    x = model_q[mask]
    y = target_q[mask]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def shared_axis_limits(gt: dict, panels: list[dict]) -> tuple[float, float, float, float]:
    x_min = float(gt["log_likelihood"].min())
    x_max = float(gt["log_likelihood"].max())
    y_max = max(
        float(gt["exact_q"].max()),
        float(gt["gt_empirical_q"].max()),
    )
    for panel in panels:
        observed = panel["model_q"] > 0.0
        if panel["n_observed_topologies"] < 100:
            continue
        if np.any(observed):
            y_max = max(y_max, float(np.percentile(panel["model_q"][observed], 99.9)))
    x_pad = 0.02 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.03 * y_max if y_max > 0.0 else 1e-12
    return x_min - x_pad, x_max + x_pad, 0.0, y_max + y_pad


def plot_panel(
    ax: plt.Axes,
    *,
    gt: dict,
    panel: dict,
    x_min: float,
    x_max: float,
    show_exact_legend: bool,
) -> None:
    x = gt["log_likelihood"]
    exact = gt["exact_q"]
    gt_emp = gt["gt_empirical_q"]
    model = panel["model_q"]
    order = np.argsort(x)

    exact_label = "exact q* ∝ R(x)" if show_exact_legend else "_nolegend_"
    gt_label = "GT multinomial sim (1M, seed 0)" if show_exact_legend else "_nolegend_"
    ax.scatter(
        x[order],
        exact[order],
        marker="o",
        s=18,
        color=GRAY,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
        label=exact_label,
        zorder=2,
    )
    ax.scatter(
        x[order],
        gt_emp[order],
        marker=".",
        s=10,
        color=GREEN,
        alpha=0.35,
        edgecolors="none",
        rasterized=True,
        label=gt_label,
        zorder=1,
    )
    observed = model > 0.0
    ax.scatter(
        x[observed],
        model[observed],
        marker="x",
        s=22,
        linewidths=0.5,
        color=BLUE,
        alpha=0.75,
        rasterized=True,
        label="checkpoint samples",
        zorder=3,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_title(panel["title"], fontsize=12)
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(frameon=False, loc="upper left", fontsize=7, markerscale=3)
    pearson = panel["pearson_vs_exact"]
    pearson_text = "nan" if not np.isfinite(pearson) else f"{pearson:.4f}"
    ax.text(
        0.98,
        0.02,
        (
            f"{panel['n_samples']:,} samples\n"
            f"{panel['n_observed_topologies']}/105 topologies\n"
            f"TV vs exact={panel['tv_vs_exact']:.4f}\n"
            f"Pearson r={pearson_text}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
    )


def main() -> None:
    args = parse_args()
    gt = load_gt_reference(args.gt_reference.resolve())
    panels = []
    for spec in method_specs(args):
        mass = topology_mass_from_npz(spec.samples.resolve(), gt)
        model_q = mass["model_q"]
        panels.append(
            {
                "title": spec.title,
                "samples_path": str(spec.samples.resolve()),
                **mass,
                "tv_vs_exact": total_variation(model_q, gt["exact_q"]),
                "tv_vs_gt_empirical": total_variation(model_q, gt["gt_empirical_q"]),
                "pearson_vs_exact": pearson_on_observed(model_q, gt["exact_q"]),
            }
        )

    x_min, x_max, y_min, y_max = shared_axis_limits(gt, panels)
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=220, constrained_layout=True)
    for ax, panel in zip(axes.ravel(), panels, strict=True):
        plot_panel(
            ax,
            gt=gt,
            panel=panel,
            x_min=x_min,
            x_max=x_max,
            show_exact_legend=(ax is axes.ravel()[0]),
        )
        ax.set_ylim(y_min, y_max)

    for ax in axes[1, :]:
        ax.set_xlabel("Terminal-tree log likelihood")
    axes[0, 0].set_ylabel("Per-topology mass ($\\hat q$)")
    axes[1, 0].set_ylabel("Per-topology mass ($\\hat q$)")
    configure_qhat_y_axis(axes[0, 0], samples=panels[0]["n_samples"])
    configure_qhat_y_axis(axes[1, 0], samples=panels[0]["n_samples"])
    fig.suptitle(
        "5-taxa topology-level ground truth (exact 105-topology target, 1M samples / method)",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    rank_fig, rank_ax = plt.subplots(figsize=(12, 6.5), dpi=220, constrained_layout=True)
    ranks = np.arange(1, len(gt["exact_q"]) + 1)
    order = np.argsort(-gt["exact_q"])
    rank_ax.scatter(
        ranks,
        gt["exact_q"][order],
        marker="o",
        s=20,
        color=GRAY,
        alpha=0.7,
        label="exact q*",
    )
    colors = plt.get_cmap("tab10")
    for idx, panel in enumerate(panels):
        ranked = panel["model_q"][order]
        rank_ax.scatter(
            ranks,
            np.maximum(ranked, 0.5 / panel["n_samples"]),
            marker="x",
            s=18,
            linewidths=0.6,
            color=colors(idx),
            alpha=0.8,
            label=panel["title"],
        )
    rank_ax.set_yscale("log")
    rank_ax.set_xlabel("Topology rank under exact target")
    rank_ax.set_ylabel("Per-topology mass (log scale)")
    rank_ax.set_title("All methods vs exact target across 105 topologies")
    rank_ax.grid(True, alpha=0.2, which="both")
    rank_ax.legend(frameon=False, fontsize=8, loc="upper right")
    rank_output = args.output.with_name("topology_gt_comparison_rank.png")
    rank_fig.savefig(rank_output, bbox_inches="tight")
    plt.close(rank_fig)

    summary = {
        "output": str(args.output.resolve()),
        "rank_output": str(rank_output.resolve()),
        "gt_reference": str(args.gt_reference.resolve()),
        "gt_definition": (
            "Exact enumeration of all 105 five-taxon topologies; "
            "q*(T) proportional to shifted-linear reward R(T)."
        ),
        "gt_metadata": gt["metadata"],
        "gt_summary": gt["summary"],
        "panels": {
            panel["title"]: {
                "samples_path": panel["samples_path"],
                "n_samples": panel["n_samples"],
                "n_observed_topologies": panel["n_observed_topologies"],
                "tv_vs_exact": panel["tv_vs_exact"],
                "tv_vs_gt_empirical": panel["tv_vs_gt_empirical"],
                "pearson_vs_exact": panel["pearson_vs_exact"],
            }
            for panel in panels
        },
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {rank_output.resolve()}")
    print(f"wrote {summary_path.resolve()}")


if __name__ == "__main__":
    main()
