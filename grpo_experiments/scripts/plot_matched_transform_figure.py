#!/usr/bin/env python3
"""Figure 6 fix: 2x2 matched-transform probability vs reward (both methods, both scales)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from build_paper_table2 import (  # noqa: E402
    load_metrics,
    pearson_linear,
    pearson_loglog,
    resolve_repo_path,
)
from reward_probability_plot_reference import (  # noqa: E402
    ideal_line_label,
    linear_probability_limits,
    log_probability_limits,
    merge_reward_axis_bounds,
    pearson_vs_ideal_sampling,
    pearson_vs_ideal_sampling_linear,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "final/paper/manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final/paper"
GRAY = "#777777"
BLUE = "#1976d2"


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    method_label: str
    metrics: dict[str, Any]
    samples_npz: Path | None
    samples_kind: str
    prerendered_linear: Path | None
    prerendered_log: Path | None
    estimator_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--taxa", type=int, default=27)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: comparisons/paper/figure6_<taxa>taxa_matched_transform.png",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scatter-points",
        type=int,
        default=80_000,
        help="Max points per regenerated panel (0 = all).",
    )
    return parser.parse_args()


def manifest_entries(manifest: dict[str, Any], taxa: int) -> list[dict[str, Any]]:
    entries = [entry for entry in manifest["comparisons"] if int(entry["taxa"]) == taxa]
    if len(entries) != 2:
        raise ValueError(f"expected exactly 2 methods for {taxa} taxa, found {len(entries)}")
    return sorted(entries, key=lambda entry: (0 if entry["method_id"] == "mips_grpo" else 1, entry["method_id"]))


def build_method_spec(entry: dict[str, Any]) -> MethodSpec:
    metrics = load_metrics(entry)
    if entry["method_id"] == "mips_grpo":
        estimator = r"$P_F(\tau)/q_\phi(\tau|x)$"
    else:
        estimator = r"$P_F(\tau)/P_B(\tau|x)$"
    return MethodSpec(
        method_id=entry["method_id"],
        method_label=entry["method_label"],
        metrics=metrics,
        samples_npz=resolve_repo_path(entry.get("samples_npz")),
        samples_kind=str(entry.get("samples_kind", "learned_reverse")),
        prerendered_linear=resolve_repo_path(entry.get("prerendered_linear")),
        prerendered_log=resolve_repo_path(entry.get("prerendered_log")),
        estimator_label=estimator,
    )


def load_learned_reverse(npz_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(npz_path) as payload:
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


def load_gflownet(npz_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(npz_path) as payload:
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


def load_samples(spec: MethodSpec) -> tuple[np.ndarray, np.ndarray, int]:
    if spec.samples_npz is None or not spec.samples_npz.exists():
        raise FileNotFoundError(f"missing samples npz for {spec.method_label}")
    if spec.samples_kind == "gflownet":
        return load_gflownet(spec.samples_npz)
    return load_learned_reverse(spec.samples_npz)


def panel_log_partition(metrics: dict[str, Any]) -> float:
    return float(
        metrics.get(
            "estimated_log_partition",
            metrics.get(
                "importance_estimated_log_partition",
                metrics.get("checkpoint_log_partition", 0.0),
            ),
        )
    )


def select_points(size: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or size <= maximum:
        return np.arange(size)
    return np.random.default_rng(seed).choice(size, size=maximum, replace=False)


def scatter_style(displayed_points: int) -> tuple[float, float]:
    if displayed_points >= 500_000:
        return 8.0, 0.10
    if displayed_points >= 100_000:
        return 12.0, 0.15
    return 20.0, 0.30


def plot_linear_panel(
    ax: plt.Axes,
    *,
    spec: MethodSpec,
    log_model_probability: np.ndarray,
    log_target_reward: np.ndarray,
    unique_signatures: int,
    axis_spec: dict[str, float],
    line_log_partition: float,
    seed: int,
    scatter_points: int,
) -> float:
    selected = select_points(len(log_model_probability), scatter_points, seed)
    model_probability = np.exp(log_model_probability[selected])
    target_reward = np.exp(log_target_reward[selected])
    partition = float(np.exp(line_log_partition))
    pearson = pearson_vs_ideal_sampling_linear(
        log_model_probability,
        log_target_reward,
        line_log_partition,
    )
    marker_size, alpha = scatter_style(len(selected))
    pearson_text = "nan" if not np.isfinite(pearson) else f"{pearson:.3f}"
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
            f"{len(log_model_probability):,} traj.; "
            f"{unique_signatures:,} sig.\n"
            f"linear $r$={pearson_text}"
        ),
    )
    line_x = np.linspace(axis_spec["reward_min"], axis_spec["reward_max"], 200)
    ax.plot(
        line_x,
        line_x / partition,
        color=GRAY,
        linestyle="--",
        linewidth=1.3,
        label=ideal_line_label(line_log_partition, show_log_z=False),
    )
    ax.set_xlim(axis_spec["reward_min"], axis_spec["reward_max"])
    y_min, y_max = linear_probability_limits(model_probability, line_x, partition)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best", fontsize=8)
    return pearson


def plot_log_panel(
    ax: plt.Axes,
    *,
    spec: MethodSpec,
    log_model_probability: np.ndarray,
    log_target_reward: np.ndarray,
    unique_signatures: int,
    axis_spec: dict[str, float],
    line_log_partition: float,
    seed: int,
    scatter_points: int,
) -> float:
    selected = select_points(len(log_model_probability), scatter_points, seed)
    x = log_target_reward[selected]
    y = log_model_probability[selected]
    pearson = pearson_vs_ideal_sampling(
        log_model_probability,
        log_target_reward,
        line_log_partition,
    )
    marker_size, alpha = scatter_style(len(selected))
    pearson_text = "nan" if not np.isfinite(pearson) else f"{pearson:.3f}"
    ax.scatter(
        x,
        y,
        s=marker_size,
        alpha=alpha,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
        label=(
            f"{spec.estimator_label}\n"
            f"{len(log_model_probability):,} traj.; "
            f"{unique_signatures:,} sig.\n"
            f"log-log $r$={pearson_text}"
        ),
    )
    line_x = np.linspace(axis_spec["log_reward_min"], axis_spec["log_reward_max"], 200)
    ax.plot(
        line_x,
        line_x - line_log_partition,
        color=GRAY,
        linestyle="--",
        linewidth=1.3,
        label=ideal_line_label(line_log_partition, show_log_z=False),
    )
    ax.set_xlim(axis_spec["log_reward_min"], axis_spec["log_reward_max"])
    y_min, y_max = log_probability_limits(y, line_x, line_log_partition)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best", fontsize=8)
    return pearson


def show_prerendered(
    ax: plt.Axes,
    image_path: Path,
    *,
    title: str,
    pearson: float | None,
    scale_label: str,
) -> None:
    image = mpimg.imread(image_path)
    ax.imshow(image)
    ax.axis("off")
    pearson_text = "—" if pearson is None or not np.isfinite(pearson) else f"{pearson:.3f}"
    ax.set_title(f"{title}\n({scale_label}; $r$={pearson_text})", fontsize=11)


def shared_axis_spec(
    loaded: list[tuple[np.ndarray, np.ndarray, MethodSpec]],
) -> dict[str, float]:
    all_reward = np.concatenate([np.exp(item[1]) for item in loaded])
    all_log_reward = np.concatenate([item[1] for item in loaded])
    reward_min = float(all_reward.min())
    reward_max = float(all_reward.max())
    log_reward_min = float(all_log_reward.min())
    log_reward_max = float(all_log_reward.max())
    reward_padding = 0.02 * (reward_max - reward_min)
    log_reward_padding = 0.02 * (log_reward_max - log_reward_min)
    return {
        "reward_min": reward_min - reward_padding,
        "reward_max": reward_max + reward_padding,
        "log_reward_min": log_reward_min - log_reward_padding,
        "log_reward_max": log_reward_max + log_reward_padding,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reward_shift = manifest.get("reward_shifts", {}).get(str(args.taxa), "?")
    entries = manifest_entries(manifest, args.taxa)
    specs = [build_method_spec(entry) for entry in entries]

    output = args.output or (
        DEFAULT_OUTPUT_DIR / f"figure6_{args.taxa}taxa_matched_transform.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    loaded_from_npz: list[tuple[np.ndarray, np.ndarray, int, MethodSpec]] = []
    for spec in specs:
        if spec.samples_npz is not None and spec.samples_npz.exists():
            log_p, log_r, unique = load_samples(spec)
            loaded_from_npz.append((log_p, log_r, unique, spec))

    axis_spec = shared_axis_spec(
        [(item[0], item[1], item[3]) for item in loaded_from_npz]
    ) if loaded_from_npz else {}

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=220, constrained_layout=True)
    panel_metrics: dict[str, Any] = {"taxa": args.taxa, "panels": {}}

    for row_idx, spec in enumerate(specs):
        line_log_partition = panel_log_partition(spec.metrics)
        linear_r = pearson_linear(spec.metrics)
        log_r = pearson_loglog(spec.metrics)

        npz_row = next((item for item in loaded_from_npz if item[3].method_id == spec.method_id), None)
        linear_ax = axes[row_idx, 0]
        log_ax = axes[row_idx, 1]

        if npz_row is not None:
            log_model_probability, log_target_reward, unique_signatures, _ = npz_row
            method_axis = merge_reward_axis_bounds(
                axis_spec,
                reward=np.exp(log_target_reward),
                log_reward=log_target_reward,
            )
            regen_linear = plot_linear_panel(
                linear_ax,
                spec=spec,
                log_model_probability=log_model_probability,
                log_target_reward=log_target_reward,
                unique_signatures=unique_signatures,
                axis_spec=method_axis,
                line_log_partition=line_log_partition,
                seed=args.seed,
                scatter_points=args.scatter_points,
            )
            regen_log = plot_log_panel(
                log_ax,
                spec=spec,
                log_model_probability=log_model_probability,
                log_target_reward=log_target_reward,
                unique_signatures=unique_signatures,
                axis_spec=method_axis,
                line_log_partition=line_log_partition,
                seed=args.seed,
                scatter_points=args.scatter_points,
            )
            panel_metrics["panels"][spec.method_id] = {
                "source": "regenerated_npz",
                "samples_npz": str(spec.samples_npz),
                "pearson_linear": regen_linear,
                "pearson_loglog": regen_log,
            }
            linear_ax.set_ylabel(spec.method_label, fontsize=12)
        else:
            if spec.prerendered_linear is None or not spec.prerendered_linear.exists():
                raise FileNotFoundError(
                    f"no NPZ or prerendered linear plot for {spec.method_label}"
                )
            show_prerendered(
                linear_ax,
                spec.prerendered_linear,
                title=spec.method_label,
                pearson=linear_r,
                scale_label="linear scale",
            )
            if spec.prerendered_log is not None and spec.prerendered_log.exists():
                show_prerendered(
                    log_ax,
                    spec.prerendered_log,
                    title=spec.method_label,
                    pearson=log_r,
                    scale_label="log scale",
                )
            else:
                log_ax.axis("off")
                log_ax.text(
                    0.5,
                    0.5,
                    f"{spec.method_label}\nlog-log panel unavailable\n(table r={format_float(log_r)})",
                    ha="center",
                    va="center",
                    transform=log_ax.transAxes,
                )
            panel_metrics["panels"][spec.method_id] = {
                "source": "prerendered_png",
                "pearson_linear": linear_r,
                "pearson_loglog": log_r,
                "prerendered_linear": str(spec.prerendered_linear),
                "prerendered_log": str(spec.prerendered_log) if spec.prerendered_log else None,
            }

    axes[0, 0].set_title("Linear: $P(x)$ vs $R(x)$", fontsize=12)
    axes[0, 1].set_title("Log-log: $\\log P(x)$ vs $\\log R(x)$", fontsize=12)
    if loaded_from_npz:
        fig.supxlabel(
            rf"Terminal reward target $R(x) = {reward_shift} + \log L(x)$ (shared axes per column)",
            fontsize=13,
        )
    fig.suptitle(
        f"{args.taxa}-taxa matched-transform comparison: MIPS-GRPO vs GFlowNet",
        fontsize=15,
        y=1.01,
    )

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(panel_metrics, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output.resolve()}")
    print(f"wrote {metrics_path.resolve()}")


def format_float(value: float | None) -> str:
    if value is None or value != value:
        return "—"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
