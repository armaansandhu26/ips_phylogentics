#!/usr/bin/env python3
"""Union-catalog reward-proportional ideal vs checkpoint sampling (10-taxa early).

Builds one shared (signature, reward) catalog from all methods' samples, then
defines ideal mass q*(sigma) = R(sigma) / sum_{sigma in union} R(sigma) so the
reference is not tied to either training algorithm's visitation distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
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
LOG_SCORE_SHIFT = 5000.0
SignatureKey = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=REPO_ROOT
        / "grpo_experiments/comparisons/10taxa/early_epoch1000_100k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <comparison-dir>/union_ideal_sampling_grid.png",
    )
    parser.add_argument("--checkpoint-epoch", type=int, default=1000)
    return parser.parse_args()


def signature_key(topology_index: int, reward: float) -> SignatureKey:
    return int(topology_index), int(np.rint(reward * 1000.0))


def load_learned_reverse(comparison_dir: Path) -> dict:
    samples_path = comparison_dir / "learned_reverse_samples_100k.npz"
    with np.load(samples_path) as payload:
        log_score = payload["log_score"].astype(np.float64)
        topology_index = payload["topology_index"].astype(np.int32)
    return {
        "title": "Learned reverse IPS",
        "reward": log_score,
        "topology_index": topology_index,
        "n_samples": len(log_score),
    }


def load_gflownet(comparison_dir: Path) -> dict:
    samples_path = (
        comparison_dir / "gflownet" / "og_gflownet_reward_probability_samples.npz"
    )
    with np.load(samples_path) as payload:
        log_reward = payload["log_reward"].astype(np.float64)
        topology_index = payload["topology_index"].astype(np.int32)
    reward = np.exp(log_reward)
    return {
        "title": "GFlowNet",
        "reward": reward,
        "topology_index": topology_index,
        "n_samples": len(reward),
    }


def build_union_catalog(methods: list[dict]) -> dict:
    """Merge unique signatures from all methods; reward is fixed per signature."""
    reward_by_key: dict[SignatureKey, float] = {}
    for method in methods:
        for topology, reward in zip(
            method["topology_index"],
            method["reward"],
            strict=True,
        ):
            key = signature_key(int(topology), float(reward))
            stored = reward_by_key.get(key)
            if stored is None:
                reward_by_key[key] = float(reward)
            elif abs(stored - float(reward)) > 1e-3:
                raise ValueError(
                    f"conflicting reward for signature {key}: {stored} vs {reward}"
                )

    keys = sorted(reward_by_key)
    signature_reward = np.asarray([reward_by_key[key] for key in keys], dtype=np.float64)
    reward_sum = float(signature_reward.sum())
    ideal_q = signature_reward / reward_sum
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    return {
        "keys": keys,
        "key_to_index": key_to_index,
        "signature_reward": signature_reward,
        "ideal_q": ideal_q,
        "reward_sum": reward_sum,
        "ideal_slope": 1.0 / reward_sum,
        "n_signatures": len(keys),
    }


def model_mass_on_union(method: dict, union: dict) -> tuple[np.ndarray, int]:
    counts = np.zeros(union["n_signatures"], dtype=np.float64)
    for topology, reward in zip(
        method["topology_index"],
        method["reward"],
        strict=True,
    ):
        idx = union["key_to_index"][signature_key(int(topology), float(reward))]
        counts[idx] += 1.0
    model_q = counts / float(method["n_samples"])
    n_observed = int(np.count_nonzero(counts))
    return model_q, n_observed


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
    overlap = int(np.count_nonzero(model_q))
    return {
        **method,
        "model_q": model_q,
        "ideal_q": union["ideal_q"],
        "signature_reward": union["signature_reward"],
        "reward_sum": union["reward_sum"],
        "ideal_slope": union["ideal_slope"],
        "n_union_signatures": union["n_signatures"],
        "n_observed_signatures": n_observed,
        "n_overlap_with_other": overlap,
        "tv_vs_union_ideal": total_variation(model_q, union["ideal_q"]),
        "pearson_vs_union_ideal": pearson_on_support(model_q, union["ideal_q"]),
    }


def shared_axis_limits(union: dict, panels: list[dict]) -> tuple[float, float, float, float]:
    x_min = float(union["signature_reward"].min())
    x_max = float(union["signature_reward"].max())
    y_max = float(union["ideal_q"].max())
    for panel in panels:
        observed = panel["model_q"] > 0.0
        if np.any(observed):
            y_max = max(y_max, float(panel["model_q"][observed].max()))
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
    x_sorted = x[order]
    ideal_sorted = ideal[order]

    ax.scatter(
        x_sorted,
        ideal_sorted,
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
        f"ideal line: q* = R / Σ_union R (slope={panel['ideal_slope']:.4g})"
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
    ax.set_title(panel["title"])
    ax.set_xlabel(rf"Terminal reward: $R(x)={LOG_SCORE_SHIFT:g}+\log L(x)$")
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(frameon=False, loc="upper left", fontsize=8, markerscale=4)
    pearson = panel["pearson_vs_union_ideal"]
    pearson_text = "nan" if not np.isfinite(pearson) else f"{pearson:.4f}"
    stats = (
        f"{panel['n_samples']:,} samples\n"
        f"{panel['n_observed_signatures']:,} seen / "
        f"{panel['n_union_signatures']:,} union\n"
        f"TV vs union ideal={panel['tv_vs_union_ideal']:.4f}\n"
        f"Pearson r={pearson_text}"
    )
    ax.text(
        0.98,
        0.02,
        stats,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
    )


def main() -> None:
    args = parse_args()
    comparison_dir = args.comparison_dir.resolve()
    output_path = args.output or (comparison_dir / "union_ideal_sampling_grid.png")

    methods = [load_learned_reverse(comparison_dir), load_gflownet(comparison_dir)]
    union = build_union_catalog(methods)
    panels = [panel_payload(method, union) for method in methods]

    lr_keys = {
        signature_key(int(t), float(r))
        for t, r in zip(methods[0]["topology_index"], methods[0]["reward"], strict=True)
    }
    gfn_keys = {
        signature_key(int(t), float(r))
        for t, r in zip(methods[1]["topology_index"], methods[1]["reward"], strict=True)
    }
    overlap = len(lr_keys & gfn_keys)

    x_min, x_max, y_min, y_max = shared_axis_limits(union, panels)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        dpi=220,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for ax, panel in zip(axes, panels, strict=True):
        plot_panel(
            ax,
            panel,
            x_min=x_min,
            x_max=x_max,
            ideal_line_in_legend=(ax is axes[0]),
        )

    axes[0].set_ylim(y_min, y_max)
    axes[0].set_ylabel("Per-signature empirical mass ($q_{\\hat{}}$)")
    configure_qhat_y_axis(axes[0], samples=panels[0]["n_samples"])
    configure_qhat_y_axis(axes[1], samples=panels[1]["n_samples"])
    fig.suptitle(
        f"10-taxa union-catalog ideal sampling (epoch {args.checkpoint_epoch}, "
        "100k samples / method)",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "comparison_dir": str(comparison_dir),
        "output": str(output_path),
        "plot_kind": "union_catalog_reward_proportional_qhat_vs_reward",
        "ideal_definition": (
            "Union over all methods' observed signatures. "
            "q*(sigma) = R(sigma) / sum_{sigma in union} R(sigma). "
            "Same ideal on every panel; not IPS-weighted and not per-method support."
        ),
        "union_catalog": {
            "n_signatures": union["n_signatures"],
            "reward_sum": union["reward_sum"],
            "ideal_slope": union["ideal_slope"],
            "signature_overlap_between_methods": overlap,
        },
        "shared_axis": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "panels": {
            panel["title"]: {
                "n_samples": panel["n_samples"],
                "n_observed_signatures": panel["n_observed_signatures"],
                "tv_vs_union_ideal": panel["tv_vs_union_ideal"],
                "pearson_vs_union_ideal": panel["pearson_vs_union_ideal"],
            }
            for panel in panels
        },
    }
    summary_path = comparison_dir / "union_ideal_sampling_grid.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
