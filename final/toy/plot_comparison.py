"""Paper-style Hyper-Grid comparison plots: target reward, sampled distributions, L1 curves."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from final.paths import FINAL_ROOT, REPO_ROOT
from final.toy.eval_sampling import (
    distribution_l1,
    empirical_terminal_grid,
    evaluate_terminal_distribution,
    modes_found_vs_samples,
)
from final.toy.hypergrid import count_modes
from final.toy.hypergrid_env import HyperGridDataset
from final.toy.hypergrid_policy import HyperGridPolicy
from final.toy.hypergrid_rollout import sample_terminals

DEFAULT_DATASET = FINAL_ROOT / "datasets" / "hypergrid_4096"
DEFAULT_GRPO_RUN = (
    FINAL_ROOT / "runs" / "hypergrid_4096" / "grpo" / "20260824_182056_hypergrid_4096_grpo"
)
DEFAULT_IPS_RUN = (
    FINAL_ROOT
    / "runs"
    / "hypergrid_4096"
    / "count_ips"
    / "20260824_182056_hypergrid_4096_count_ips"
)
DEFAULT_OUT = FINAL_ROOT / "runs" / "hypergrid_4096" / "plots"
EVAL_L1_RE = re.compile(r"epoch=(\d+) eval L1=([\d.]+)")

# Publication styling (4-method comparisons).
PAPER_DPI = 300
METHOD_IDS = ("grpo", "count_ips", "learned_reverse_ips", "trajectory_balance")
METHOD_LABELS: dict[str, str] = {
    "grpo": "GRPO",
    "count_ips": "IPS-GRPO",
    "learned_reverse_ips": "MIPS-GRPO",
    "trajectory_balance": "GFlowNet TB",
}
METHOD_ORDER = [METHOD_LABELS[mid] for mid in METHOD_IDS]
METHOD_STYLE: dict[str, dict[str, str]] = {
    METHOD_LABELS["grpo"]: {"color": "#e45756", "marker": "o"},
    METHOD_LABELS["count_ips"]: {"color": "#4c78a8", "marker": "s"},
    METHOD_LABELS["learned_reverse_ips"]: {"color": "#54a24b", "marker": "^"},
    METHOD_LABELS["trajectory_balance"]: {"color": "#f58518", "marker": "D"},
}
LABEL_TO_METHOD_ID = {label: mid for mid, label in METHOD_LABELS.items()}
CHECKPOINT_EPOCH_RE = re.compile(r"checkpoint_epoch(\d+)\.pt$")

# Visual separation for methods that collapse (stay at 1 mode) vs those that recover all modes.
METHOD_LINESTYLE: dict[str, str] = {
    METHOD_LABELS["grpo"]: (0, (4, 2)),            # dashed
    METHOD_LABELS["count_ips"]: (0, (2, 2)),        # dotted
    METHOD_LABELS["learned_reverse_ips"]: "-",
    METHOD_LABELS["trajectory_balance"]: (0, (3, 1, 1, 1)),  # dash-dot
}


def method_style(method_id: str) -> dict[str, str]:
    return METHOD_STYLE[METHOD_LABELS[method_id]]


def find_last_common_checkpoint_epoch(run_dirs: dict[str, Path]) -> int:
    """Latest checkpoint epoch present in every method run directory."""
    if not run_dirs:
        raise ValueError("no run dirs provided")
    common: set[int] | None = None
    for label, run_dir in run_dirs.items():
        epochs = {
            int(m.group(1))
            for path in run_dir.glob("checkpoint_epoch*.pt")
            if (m := CHECKPOINT_EPOCH_RE.search(path.name))
        }
        if not epochs:
            raise FileNotFoundError(f"no checkpoint_epoch*.pt in {run_dir} ({label})")
        common = epochs if common is None else common & epochs
    if not common:
        raise ValueError("no common checkpoint epoch across all methods")
    return max(common)


def _modes_series_sort_key(item: tuple[str, np.ndarray, np.ndarray, int]) -> tuple[int, int]:
    """Sort so collapsed low-index methods (GRPO) and recovered high-index methods (TB) draw on top."""
    label, _, _, max_modes = item
    idx = METHOD_ORDER.index(label) if label in METHOD_ORDER else 0
    tie = idx if max_modes > 1 else -idx
    return (max_modes, tie)


def _linear_sample_counts(n: int, *, num_points: int = 50) -> np.ndarray:
    counts = np.unique(np.linspace(1, n, num=num_points).astype(np.int64))
    return counts[(counts >= 1) & (counts <= n)]


def _log_sample_counts(n: int, *, num_points: int = 40) -> np.ndarray:
    counts = np.unique(np.round(np.logspace(1, np.log10(n), num=num_points)).astype(np.int64))
    return counts[(counts >= 1) & (counts <= n)]


def _plot_modes_found_on_ax(
    ax,
    *,
    label: str,
    sample_counts: np.ndarray,
    modes_found: np.ndarray,
    zorder: int,
    marker_phase: int = 0,
) -> None:
    style = METHOD_STYLE.get(label, {"color": "black", "marker": "o"})
    max_modes = int(modes_found.max()) if len(modes_found) else 0
    collapsed = max_modes <= 1
    markevery = max(1, len(sample_counts) // 8)
    ax.plot(
        sample_counts,
        modes_found,
        linestyle=METHOD_LINESTYLE.get(label, "-"),
        drawstyle="steps-post",
        marker=style["marker"],
        markevery=(slice(marker_phase % markevery, None, markevery) if len(sample_counts) > markevery else markevery),
        markersize=7 if collapsed else 6,
        markerfacecolor=style["color"],
        markeredgecolor="white",
        markeredgewidth=1.0,
        linewidth=2.6 if collapsed else 2.2,
        color=style["color"],
        label=label,
        zorder=zorder,
        alpha=0.95,
    )


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": PAPER_DPI,
            "savefig.dpi": PAPER_DPI,
            "savefig.bbox": "tight",
        }
    )


def latest_run_dir(method_root: Path) -> Path:
    candidates = [p for p in method_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run dirs under {method_root}")

    def _score(path: Path) -> tuple[int, int, float]:
        complete = int((path / "final_checkpoint.pt").exists())
        metrics = path / "metrics.jsonl"
        steps = sum(1 for _ in metrics.open()) if metrics.exists() else 0
        return (complete, steps, path.stat().st_mtime)

    return max(candidates, key=_score)


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = list(run_dir.glob("checkpoint_epoch*.pt"))
    final = run_dir / "final_checkpoint.pt"
    if final.exists():
        candidates.append(final)
    if not candidates:
        return None

    def _global_step(path: Path) -> int:
        meta = torch.load(path, map_location="cpu", weights_only=False)
        return int(meta.get("global_step", -1))

    return max(candidates, key=_global_step)


def load_eval_l1_history(run_dir: Path) -> tuple[list[int], list[float]]:
    by_epoch: dict[int, float] = {}
    summaries = run_dir / "epoch_summaries.json"
    if summaries.exists():
        for row in json.loads(summaries.read_text(encoding="utf-8")):
            if "l1_distance" in row:
                by_epoch[int(row["epoch"])] = float(row["l1_distance"])
    for log in sorted(run_dir.glob("*.log")):
        for line in log.read_text(encoding="utf-8").splitlines():
            match = EVAL_L1_RE.search(line)
            if match:
                by_epoch[int(match.group(1))] = float(match.group(2))
    epochs = sorted(by_epoch)
    return epochs, [by_epoch[e] for e in epochs]


def resolve_suite_paths(suite_id: str) -> tuple[Path, Path, Path, Path]:
    base = FINAL_ROOT / "runs" / suite_id
    dataset = FINAL_ROOT / "datasets" / suite_id
    return (
        dataset,
        latest_run_dir(base / "grpo"),
        latest_run_dir(base / "count_ips"),
        base / "plots",
    )


def plot_gt_vs_method(
    dataset: HyperGridDataset,
    *,
    coords: np.ndarray,
    method_label: str,
    out_path: Path,
) -> dict:
    rewards = dataset.rewards
    target_probs = dataset.load_target_probs()
    spec = dataset.spec
    empirical = empirical_terminal_grid(coords, H=spec.H, D=spec.D)
    eval_metrics = evaluate_terminal_distribution(coords, dataset)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=150)
    im0 = _panel_imshow(
        axes[0],
        rewards,
        title="(a) Target Reward",
        metrics_line="ground-truth R(x)",
        cmap="viridis",
        vmax=float(rewards.max()),
        show_ylabel=True,
    )
    im1 = _panel_imshow(
        axes[1],
        empirical,
        title=f"(b) {method_label}",
        metrics_line=(
            f"L1={eval_metrics['l1_distance']:.3f}, "
            f"modes={int(eval_metrics['num_modes_with_mass'])}/"
            f"{int(eval_metrics['expected_num_modes'])}"
        ),
        cmap="viridis",
        vmax=max(float(empirical.max()), float(target_probs.max()) * 0.5, 1e-6),
        show_ylabel=True,
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.subplots_adjust(wspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {"plot": str(out_path), **{k: v for k, v in eval_metrics.items() if k not in {"sampled_coords", "sampled_rewards"}}}


def plot_modes_found_curve(
    dataset: HyperGridDataset,
    *,
    grpo_coords: np.ndarray,
    ips_coords: np.ndarray,
    out_path: Path,
) -> dict:
    expected = int(count_modes(dataset.rewards))
    grpo_curve = modes_found_vs_samples(
        grpo_coords, dataset, sample_counts=_linear_sample_counts(len(grpo_coords))
    )
    ips_curve = modes_found_vs_samples(
        ips_coords, dataset, sample_counts=_linear_sample_counts(len(ips_coords))
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    series = [
        (METHOD_LABELS["grpo"], grpo_curve["sample_counts"], grpo_curve["modes_found"], int(grpo_curve["modes_found"].max())),
        (METHOD_LABELS["count_ips"], ips_curve["sample_counts"], ips_curve["modes_found"], int(ips_curve["modes_found"].max())),
    ]
    for zorder, (label, xs, ys, _max_modes) in enumerate(
        sorted(series, key=_modes_series_sort_key), start=2
    ):
        idx = METHOD_ORDER.index(label) if label in METHOD_ORDER else 0
        _plot_modes_found_on_ax(
            ax, label=label, sample_counts=xs, modes_found=ys, zorder=zorder, marker_phase=idx
        )

    ax.set_xlabel("Samples Drawn")
    ax.set_ylabel("Modes Found")
    ax.set_title("Peak modes discovered vs samples drawn")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.2, expected + 0.5)
    ax.set_yticks(range(expected + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "plot": str(out_path),
        "expected_modes": expected,
        "grpo_recovery_pct": float(grpo_curve["recovery_rate_pct"]),
        "ips_recovery_pct": float(ips_curve["recovery_rate_pct"]),
        "grpo_modes_at_max_samples": int(grpo_curve["modes_found"][-1]),
        "ips_modes_at_max_samples": int(ips_curve["modes_found"][-1]),
    }


def write_recovery_summary(
    *,
    dataset: HyperGridDataset,
    grpo_coords: np.ndarray,
    ips_coords: np.ndarray,
    grpo_run: Path,
    ips_run: Path,
    out_path: Path,
) -> dict:
    grpo_eval = evaluate_terminal_distribution(grpo_coords, dataset)
    ips_eval = evaluate_terminal_distribution(ips_coords, dataset)
    expected = int(grpo_eval["expected_num_modes"])
    grpo_label = METHOD_LABELS["grpo"]
    ips_label = METHOD_LABELS["count_ips"]
    summary = {
        "task": dataset.root.name,
        "expected_peak_modes": expected,
        "methods": {
            grpo_label: {
                "run_dir": str(grpo_run),
                "recovery_rate_pct": 100.0 * grpo_eval["num_modes_with_mass"] / expected,
                "modes_found": int(grpo_eval["num_modes_with_mass"]),
                "l1_distance": grpo_eval["l1_distance"],
                "peak_mode_mass": grpo_eval["peak_mode_mass"],
                "sampled_unique_terminals": grpo_eval["sampled_unique_terminals"],
            },
            ips_label: {
                "run_dir": str(ips_run),
                "recovery_rate_pct": 100.0 * ips_eval["num_modes_with_mass"] / expected,
                "modes_found": int(ips_eval["num_modes_with_mass"]),
                "l1_distance": ips_eval["l1_distance"],
                "peak_mode_mass": ips_eval["peak_mode_mass"],
                "sampled_unique_terminals": ips_eval["sampled_unique_terminals"],
            },
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.5, 2.2), dpi=150)
    ax.axis("off")
    rows = [
        ["Method", "Recovery rate (%)", "Modes found", "L1", "Unique terminals"],
        [
            grpo_label,
            f"{summary['methods'][grpo_label]['recovery_rate_pct']:.1f}",
            str(summary["methods"][grpo_label]["modes_found"]),
            f"{summary['methods'][grpo_label]['l1_distance']:.3f}",
            str(summary["methods"][grpo_label]["sampled_unique_terminals"]),
        ],
        [
            ips_label,
            f"{summary['methods'][ips_label]['recovery_rate_pct']:.1f}",
            str(summary["methods"][ips_label]["modes_found"]),
            f"{summary['methods'][ips_label]['l1_distance']:.3f}",
            str(summary["methods"][ips_label]["sampled_unique_terminals"]),
        ],
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    fig.tight_layout()
    table_png = out_path.with_suffix(".png")
    fig.savefig(table_png, bbox_inches="tight")
    plt.close(fig)
    summary["table_plot"] = str(table_png)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Centered rolling mean and std with edge padding."""
    y = np.asarray(values, dtype=np.float64)
    n = len(y)
    if n == 0:
        return y, y
    w = max(1, min(int(window), n))
    pad = w // 2
    kernel = np.ones(w, dtype=np.float64) / w
    ypad = np.pad(y, (pad, pad), mode="edge")
    mean = np.convolve(ypad, kernel, mode="valid")[:n]
    y2pad = np.pad(y * y, (pad, pad), mode="edge")
    second_moment = np.convolve(y2pad, kernel, mode="valid")[:n]
    var = np.maximum(second_moment - mean * mean, 0.0)
    return mean, np.sqrt(var)


def _smooth_window(num_points: int) -> int:
    if num_points <= 200:
        return max(10, num_points // 20)
    if num_points <= 5000:
        return max(50, num_points // 100)
    return max(100, min(1000, num_points // 200))


def _plot_smoothed_series(
    ax,
    steps: list[int],
    values: list[float],
    *,
    label: str,
    color: str,
    window: int | None = None,
    legend_label: str | None = None,
) -> None:
    x = np.asarray(steps, dtype=np.int64)
    y = np.asarray(values, dtype=np.float64)
    if len(y) == 0:
        return
    window = window or _smooth_window(len(y))
    mean, std = _rolling_mean_std(y, window)
    ax.plot(x, y, linewidth=0.5, color=color, alpha=0.10)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, linewidth=1.8, color=color, label=legend_label or label)


def plot_training_reward_curve(
    *,
    grpo_run: Path,
    ips_run: Path,
    out_path: Path,
) -> Path:
    def _load_steps(run_dir: Path) -> tuple[list[int], list[float]]:
        rows = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [int(r["global_step"]) for r in rows], [float(r.get("mean_reward", np.nan)) for r in rows]

    g_steps, g_reward = _load_steps(grpo_run)
    i_steps, i_reward = _load_steps(ips_run)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    grpo_style = method_style("grpo")
    ips_style = method_style("count_ips")
    _plot_smoothed_series(
        ax, g_steps, g_reward,
        label=METHOD_LABELS["grpo"], color=grpo_style["color"],
        legend_label=f"{METHOD_LABELS['grpo']} (rolling avg ±1σ)",
    )
    _plot_smoothed_series(
        ax, i_steps, i_reward,
        label=METHOD_LABELS["count_ips"], color=ips_style["color"],
        legend_label=f"{METHOD_LABELS['count_ips']} (rolling avg ±1σ)",
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean terminal reward (batch)")
    ax.set_title("Training reward vs step (rolling average ±1σ)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _method_cache_slug(method_id: str) -> str:
    return method_id


def _coords_cache_path(
    out_dir: Path,
    *,
    method_id: str,
    checkpoint_epoch: int | None,
    num_samples: int,
) -> Path:
    epoch_tag = f"epoch{checkpoint_epoch:04d}_" if checkpoint_epoch is not None else ""
    return out_dir / "cache" / f"{_method_cache_slug(method_id)}_{epoch_tag}n{num_samples}.npz"


def load_or_sample_coords(
    run_dir: Path,
    dataset: HyperGridDataset,
    *,
    method_id: str,
    num_samples: int,
    device: str,
    checkpoint: Path | None = None,
    checkpoint_epoch: int | None = None,
    cache_dir: Path | None = None,
    force_resample: bool = False,
) -> np.ndarray:
    cache_path = (
        _coords_cache_path(cache_dir, method_id=method_id, checkpoint_epoch=checkpoint_epoch, num_samples=num_samples)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and cache_path.exists() and not force_resample:
        payload = np.load(cache_path)
        return np.asarray(payload["coords"])

    coords = load_coords_from_checkpoint(
        run_dir,
        dataset,
        num_samples=num_samples,
        device=device,
        checkpoint=checkpoint,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, coords=coords)
    return coords


def load_coords_from_checkpoint(
    run_dir: Path,
    dataset: HyperGridDataset,
    *,
    num_samples: int,
    batch_size: int = 256,
    device: str = "cpu",
    checkpoint: Path | None = None,
) -> np.ndarray:
    ckpt_path = checkpoint or find_latest_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"no checkpoint found in {run_dir}")
    spec = dataset.spec
    policy = HyperGridPolicy(
        dim=spec.D,
        num_actions=dataset.num_actions,
        H=spec.H,
        hidden_size=256,
        num_layers=2,
    )
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["policy"])
    policy.eval()
    coords, _ = sample_terminals(
        policy,
        dataset,
        num_samples=num_samples,
        batch_size=min(batch_size, num_samples),
        device=device,
    )
    return coords.numpy()


def load_coords_for_run(
    run_dir: Path,
    dataset: HyperGridDataset,
    *,
    num_samples: int = 50_000,
    device: str = "cuda:0",
) -> np.ndarray:
    ckpt = find_latest_checkpoint(run_dir)
    if ckpt is not None:
        return load_coords_from_checkpoint(
            run_dir,
            dataset,
            num_samples=num_samples,
            batch_size=256,
            device=device,
            checkpoint=ckpt,
        )
    return load_coords_from_run(run_dir)
def load_coords_from_run(run_dir: Path) -> np.ndarray:
    npz_paths = sorted(run_dir.glob("sampled_terminals_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"no sampled_terminals_*.npz in {run_dir}")
    payload = np.load(npz_paths[-1])
    return np.asarray(payload["coords"])


def _panel_imshow(
    ax,
    grid: np.ndarray,
    *,
    title: str,
    metrics_line: str | None = None,
    cmap: str,
    vmax: float | None,
    show_ylabel: bool = False,
):
    vmax = float(grid.max()) if vmax is None else vmax
    if vmax <= 0:
        vmax = 1.0
    im = ax.imshow(
        grid.T,
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
    )
    if metrics_line:
        ax.set_title(f"{title}\n{metrics_line}", fontsize=10, pad=6, linespacing=1.25)
    else:
        ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("x", labelpad=2)
    if show_ylabel:
        ax.set_ylabel("y", labelpad=2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=2, pad=1)
    return im


def plot_all_methods_distribution_grid(
    dataset: HyperGridDataset,
    *,
    methods: list[tuple[str, np.ndarray]],
    out_path: Path,
) -> dict:
    """Target reward + sampled distributions for all methods in one row."""
    _apply_paper_style()
    rewards = dataset.rewards
    target_probs = dataset.load_target_probs()
    spec = dataset.spec
    n_panels = 1 + len(methods)

    fig = plt.figure(figsize=(2.55 * n_panels, 3.6))
    gs = fig.add_gridspec(
        1,
        n_panels,
        wspace=0.28,
        top=0.88,
        bottom=0.12,
        left=0.04,
        right=0.92,
    )

    ax_target = fig.add_subplot(gs[0, 0])
    im_target = _panel_imshow(
        ax_target,
        rewards,
        title="(a) Target Reward",
        metrics_line="ground-truth R(x)",
        cmap="viridis",
        vmax=float(rewards.max()),
        show_ylabel=True,
    )
    cbar_target = fig.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04)
    cbar_target.ax.tick_params(labelsize=8)

    method_metrics: dict[str, dict] = {}
    vmax_sample = max(float(target_probs.max()) * 0.5, 1e-6)
    sample_axes = []
    sample_images = []
    panel_letter = ord("b")

    for col, (label, coords) in enumerate(methods, start=1):
        ax = fig.add_subplot(gs[0, col])
        grid = empirical_terminal_grid(coords, H=spec.H, D=spec.D)
        eval_metrics = evaluate_terminal_distribution(coords, dataset)
        vmax = max(float(grid.max()), vmax_sample)
        im = _panel_imshow(
            ax,
            grid,
            title=f"({chr(panel_letter)}) {label}",
            metrics_line=(
                f"L1={eval_metrics['l1_distance']:.3f}, "
                f"modes={int(eval_metrics['num_modes_with_mass'])}/"
                f"{int(eval_metrics['expected_num_modes'])}"
            ),
            cmap="viridis",
            vmax=vmax,
            show_ylabel=False,
        )
        sample_axes.append(ax)
        sample_images.append(im)
        method_metrics[label] = {
            k: v for k, v in eval_metrics.items() if k not in {"sampled_coords", "sampled_rewards"}
        }
        panel_letter += 1

    if sample_images:
        cbar_sample = fig.colorbar(
            sample_images[0],
            ax=sample_axes,
            fraction=0.025,
            pad=0.02,
            location="right",
        )
        cbar_sample.ax.tick_params(labelsize=8)
        cbar_sample.set_label("sampled mass", fontsize=9, labelpad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {"plot": str(out_path), "methods": method_metrics}


def plot_all_methods_training_l1(
    run_dirs: dict[str, Path],
    *,
    out_path: Path,
    max_epoch: int | None = None,
) -> Path:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    all_l1: list[float] = []
    for label in METHOD_ORDER:
        if label not in run_dirs:
            continue
        epochs, l1_vals = load_eval_l1_history(run_dirs[label])
        if max_epoch is not None:
            pairs = [(e, v) for e, v in zip(epochs, l1_vals) if e <= max_epoch]
            if not pairs:
                continue
            epochs, l1_vals = zip(*pairs)
            epochs, l1_vals = list(epochs), list(l1_vals)
        style = METHOD_STYLE[label]
        ax.plot(
            epochs,
            l1_vals,
            marker=style["marker"],
            markevery=max(1, len(epochs) // 12),
            markersize=4,
            linewidth=1.8,
            color=style["color"],
            label=label,
        )
        all_l1.extend(l1_vals)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 distance to target")
    ax.set_title("Eval L1 vs training epoch")
    ax.set_ylim(0.0, max(2.2, max(all_l1, default=[2.0]) * 1.05))
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, borderpad=0.6)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_all_methods_modes_found(
    dataset: HyperGridDataset,
    *,
    method_coords: list[tuple[str, np.ndarray]],
    out_path: Path,
    x_scale: str = "linear",
) -> dict:
    _apply_paper_style()
    expected = int(count_modes(dataset.rewards))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    result: dict[str, float | int | str] = {
        "expected_modes": expected,
        "plot": str(out_path),
        "x_scale": x_scale,
    }

    sample_counts_fn = _log_sample_counts if x_scale == "log" else _linear_sample_counts
    series: list[tuple[str, np.ndarray, np.ndarray, int]] = []
    for label, coords in method_coords:
        n = len(coords)
        curve = modes_found_vs_samples(
            coords,
            dataset,
            sample_counts=sample_counts_fn(n),
        )
        max_modes = int(curve["modes_found"].max()) if len(curve["modes_found"]) else 0
        series.append((label, curve["sample_counts"], curve["modes_found"], max_modes))
        key = LABEL_TO_METHOD_ID.get(label, label.lower().replace(" ", "_").replace("-", "_"))
        result[f"{key}_recovery_pct"] = float(curve["recovery_rate_pct"])
        result[f"{key}_modes_at_max_samples"] = int(curve["modes_found"][-1])

    # Draw low-recovery curves first; collapsed/recovered leaders on top within each group.
    for zorder, (label, xs, ys, _max_modes) in enumerate(
        sorted(series, key=_modes_series_sort_key),
        start=2,
    ):
        idx = METHOD_ORDER.index(label) if label in METHOD_ORDER else 0
        _plot_modes_found_on_ax(
            ax,
            label=label,
            sample_counts=xs,
            modes_found=ys,
            zorder=zorder,
            marker_phase=idx,
        )

    ax.set_xlabel("Samples drawn")
    ax.set_ylabel("Peak modes found")
    title_suffix = " (log x)" if x_scale == "log" else ""
    ax.set_title(f"Mode discovery vs samples drawn{title_suffix}")
    if x_scale == "log":
        ax.set_xscale("log")
        ax.set_xlim(left=1)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    else:
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_ylim(-0.15, expected + 0.35)
    ax.set_yticks(range(expected + 1))
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, borderpad=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return result


def plot_all_methods_training_reward(
    run_dirs: dict[str, Path],
    *,
    out_path: Path,
    max_epoch: int | None = None,
) -> Path:
    _apply_paper_style()

    def _load_steps(run_dir: Path) -> tuple[list[int], list[float]]:
        rows = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if max_epoch is not None:
            rows = [r for r in rows if int(r.get("epoch", 0)) <= max_epoch]
        return [int(r["global_step"]) for r in rows], [float(r.get("mean_reward", np.nan)) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label in METHOD_ORDER:
        if label not in run_dirs:
            continue
        steps, reward = _load_steps(run_dirs[label])
        style = METHOD_STYLE[label]
        _plot_smoothed_series(ax, steps, reward, label=label, color=style["color"])

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean terminal reward (batch)")
    ax.set_title("Training reward (rolling average ±1σ)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, borderpad=0.6, fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def write_all_methods_recovery_summary(
    *,
    dataset: HyperGridDataset,
    method_coords: list[tuple[str, np.ndarray]],
    run_dirs: dict[str, Path],
    out_path: Path,
) -> dict:
    _apply_paper_style()
    expected = int(count_modes(dataset.rewards))
    summary: dict = {
        "task": dataset.root.name,
        "expected_peak_modes": expected,
        "methods": {},
    }

    for label, coords in method_coords:
        ev = evaluate_terminal_distribution(coords, dataset)
        summary["methods"][label] = {
            "run_dir": str(run_dirs[label]),
            "recovery_rate_pct": 100.0 * ev["num_modes_with_mass"] / expected,
            "modes_found": int(ev["num_modes_with_mass"]),
            "l1_distance": ev["l1_distance"],
            "peak_mode_mass": ev["peak_mode_mass"],
            "sampled_unique_terminals": ev["sampled_unique_terminals"],
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    rows = [["Method", "Recovery (%)", "Modes", "L1", "Unique terminals"]]
    for label in METHOD_ORDER:
        if label not in summary["methods"]:
            continue
        m = summary["methods"][label]
        rows.append(
            [
                label,
                f"{m['recovery_rate_pct']:.1f}",
                str(m["modes_found"]),
                f"{m['l1_distance']:.3f}",
                str(m["sampled_unique_terminals"]),
            ]
        )

    fig, ax = plt.subplots(figsize=(8.8, 0.45 * len(rows) + 0.8))
    ax.axis("off")
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.6)
    fig.tight_layout()
    table_png = out_path.with_suffix(".png")
    fig.savefig(table_png)
    plt.close(fig)
    summary["table_plot"] = str(table_png)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def generate_checkpoint_comparison(
    dataset: HyperGridDataset,
    *,
    run_dirs: dict[str, Path],
    checkpoint_epoch: int,
    out_dir: Path,
    num_samples: int = 50_000,
    device: str = "cuda:0",
    force_resample: bool = False,
    plots: set[str] | None = None,
) -> dict:
    """Full 4-method paper plot set at a common checkpoint epoch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"checkpoint_epoch{checkpoint_epoch:04d}.pt"
    all_plots = {"terminal", "l1", "reward", "modes", "recovery"}
    want = all_plots if plots is None else plots

    method_coords: list[tuple[str, np.ndarray]] = []
    for label in METHOD_ORDER:
        if label not in run_dirs:
            continue
        method_id = LABEL_TO_METHOD_ID[label]
        run_dir = run_dirs[label]
        ckpt = run_dir / ckpt_name
        if not ckpt.exists():
            raise FileNotFoundError(f"missing {ckpt} for {label}")
        coords = load_or_sample_coords(
            run_dir,
            dataset,
            method_id=method_id,
            num_samples=num_samples,
            device=device,
            checkpoint=ckpt,
            checkpoint_epoch=checkpoint_epoch,
            cache_dir=out_dir,
            force_resample=force_resample,
        )
        method_coords.append((label, coords))

    result: dict = {
        "checkpoint_epoch": checkpoint_epoch,
        "num_samples": num_samples,
        "out_dir": str(out_dir.resolve()),
    }

    if "terminal" in want:
        result["terminal_distribution_comparison"] = plot_all_methods_distribution_grid(
            dataset,
            methods=method_coords,
            out_path=out_dir / "terminal_distribution_comparison.png",
        )
    if "l1" in want:
        result["training_l1_curve"] = str(
            plot_all_methods_training_l1(
                run_dirs,
                out_path=out_dir / "training_l1_curve.png",
                max_epoch=checkpoint_epoch,
            )
        )
    if "reward" in want:
        result["training_reward_curve"] = str(
            plot_all_methods_training_reward(
                run_dirs,
                out_path=out_dir / "training_reward_curve.png",
                max_epoch=checkpoint_epoch,
            )
        )
    if "modes" in want:
        result["modes_found_vs_samples"] = plot_all_methods_modes_found(
            dataset,
            method_coords=method_coords,
            out_path=out_dir / "modes_found_vs_samples.png",
            x_scale="linear",
        )
        result["modes_found_vs_samples_logx"] = plot_all_methods_modes_found(
            dataset,
            method_coords=method_coords,
            out_path=out_dir / "modes_found_vs_samples_logx.png",
            x_scale="log",
        )
    if "recovery" in want:
        result["recovery_summary"] = write_all_methods_recovery_summary(
            dataset=dataset,
            method_coords=method_coords,
            run_dirs=run_dirs,
            out_path=out_dir / "recovery_summary.json",
        )

    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def resolve_all_method_runs(suite_id: str) -> dict[str, Path]:
    base = FINAL_ROOT / "runs" / suite_id
    subdirs = {
        "grpo": base / "grpo",
        "count_ips": base / "count_ips",
        "learned_reverse_ips": base / "learned_reverse_ips",
        "trajectory_balance": base / "trajectory_balance",
    }
    runs: dict[str, Path] = {}
    for mid in METHOD_IDS:
        subdir = subdirs[mid]
        if subdir.is_dir():
            runs[METHOD_LABELS[mid]] = latest_run_dir(subdir)
    return runs


def plot_terminal_distribution_grid(
    dataset: HyperGridDataset,
    *,
    grpo_coords: np.ndarray,
    ips_coords: np.ndarray,
    grpo_run: Path,
    ips_run: Path,
    out_path: Path,
) -> dict:
    rewards = dataset.rewards
    target_probs = dataset.load_target_probs()
    spec = dataset.spec
    grpo_grid = empirical_terminal_grid(grpo_coords, H=spec.H, D=spec.D)
    ips_grid = empirical_terminal_grid(ips_coords, H=spec.H, D=spec.D)

    grpo_l1 = distribution_l1(grpo_grid, target_probs)
    ips_l1 = distribution_l1(ips_grid, target_probs)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=150)

    im0 = _panel_imshow(
        axes[0],
        rewards,
        title="(a) Target Reward",
        metrics_line="ground-truth R(x)",
        cmap="viridis",
        vmax=float(rewards.max()),
        show_ylabel=True,
    )
    im1 = _panel_imshow(
        axes[1],
        grpo_grid,
        title=f"(b) {METHOD_LABELS['grpo']}",
        metrics_line=f"L1={grpo_l1:.3f}",
        cmap="viridis",
        vmax=max(float(grpo_grid.max()), float(target_probs.max()) * 0.5, 1e-6),
        show_ylabel=True,
    )
    im2 = _panel_imshow(
        axes[2],
        ips_grid,
        title=f"(c) {METHOD_LABELS['count_ips']}",
        metrics_line=f"L1={ips_l1:.3f}",
        cmap="viridis",
        vmax=max(float(ips_grid.max()), float(target_probs.max()) * 0.5, 1e-6),
        show_ylabel=False,
    )

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.subplots_adjust(bottom=0.22, wspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "plot": str(out_path),
        "grpo_l1": grpo_l1,
        "ips_l1": ips_l1,
        "grpo_run": str(grpo_run),
        "ips_run": str(ips_run),
    }


def _epochs_with_l1(summaries: list[dict]) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    l1_vals: list[float] = []
    for row in summaries:
        if "l1_distance" in row:
            epochs.append(int(row["epoch"]))
            l1_vals.append(float(row["l1_distance"]))
    return epochs, l1_vals


def plot_training_l1_curve(
    *,
    grpo_run: Path,
    ips_run: Path,
    out_path: Path,
) -> Path:
    g_epochs, g_l1 = load_eval_l1_history(grpo_run)
    i_epochs, i_l1 = load_eval_l1_history(ips_run)

    grpo_style = method_style("grpo")
    ips_style = method_style("count_ips")
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(
        g_epochs, g_l1,
        marker=grpo_style["marker"], color=grpo_style["color"],
        label=METHOD_LABELS["grpo"], linewidth=2,
    )
    ax.plot(
        i_epochs, i_l1,
        marker=ips_style["marker"], color=ips_style["color"],
        label=METHOD_LABELS["count_ips"], linewidth=2,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 distance to target")
    ax.set_title("Eval L1 vs training epoch")
    ax.set_ylim(0.0, max(2.2, max(g_l1 + i_l1, default=[2.0]) * 1.05))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_training_diagnostics(
    run_dir: Path,
    *,
    method_label: str,
    out_path: Path,
) -> Path:
    summaries = json.loads((run_dir / "epoch_summaries.json").read_text(encoding="utf-8"))
    metrics_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    epochs, l1_vals = _epochs_with_l1(summaries)
    steps = [int(r["global_step"]) for r in metrics_rows]
    mean_reward = [float(r.get("mean_reward", np.nan)) for r in metrics_rows]
    unique = [float(r.get("batch_unique_outcomes", np.nan)) for r in metrics_rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)

    axes[0, 0].plot(steps, mean_reward, linewidth=1)
    axes[0, 0].set_title("Mean terminal reward (batch)")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, unique, linewidth=1, color="tab:orange")
    axes[0, 1].set_title("Unique terminals per batch")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].grid(True, alpha=0.3)

    if epochs:
        axes[1, 0].plot(epochs, l1_vals, marker="o")
    axes[1, 0].set_title("Eval L1 to target")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)

    cum_unique = [float(r.get("cumulative_unique_outcomes", np.nan)) for r in metrics_rows]
    axes[1, 1].plot(steps, cum_unique, linewidth=1, color="tab:green")
    axes[1, 1].set_title("Cumulative unique terminals")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{method_label} training diagnostics", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot Hyper-Grid comparison figures.")
    parser.add_argument("--suite", default=None, help="e.g. hypergrid_64 — uses latest run dirs")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--grpo-run", type=Path, default=None)
    parser.add_argument("--ips-run", type=Path, default=None)
    parser.add_argument("--learned-reverse-run", type=Path, default=None, help="Override MIPS-GRPO run dir.")
    parser.add_argument("--mips-grpo-run", type=Path, default=None, help="Alias for --learned-reverse-run.")
    parser.add_argument("--tb-run", type=Path, default=None)
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Plot target vs all four methods (GRPO, IPS-GRPO, MIPS-GRPO, GFlowNet TB).",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-epoch", type=int, default=None)
    parser.add_argument(
        "--last-common-checkpoint",
        action="store_true",
        help="Use the latest checkpoint epoch shared by all four method runs.",
    )
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--sample-device", default="cuda:0")
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Ignore cached terminal samples and re-roll from checkpoints.",
    )
    parser.add_argument(
        "--plots",
        default=None,
        help="Comma-separated subset: terminal,l1,reward,modes,recovery (default: all).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Upload comparison plots to a wandb summary run.",
    )
    parser.add_argument("--wandb-project", default="phylogfn-final")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    args = parser.parse_args(argv)

    if args.all_methods and args.suite is not None:
        dataset_path = FINAL_ROOT / "datasets" / args.suite
        run_dirs = resolve_all_method_runs(args.suite)
        if args.grpo_run is not None:
            run_dirs[METHOD_LABELS["grpo"]] = args.grpo_run.resolve()
        if args.ips_run is not None:
            run_dirs[METHOD_LABELS["count_ips"]] = args.ips_run.resolve()
        mips_run = args.mips_grpo_run or args.learned_reverse_run
        if mips_run is not None:
            run_dirs[METHOD_LABELS["learned_reverse_ips"]] = mips_run.resolve()
        if args.tb_run is not None:
            run_dirs[METHOD_LABELS["trajectory_balance"]] = args.tb_run.resolve()
        if not run_dirs:
            raise FileNotFoundError(
                f"no method run dirs found for suite {args.suite!r}; "
                "pass --grpo-run/--ips-run/--mips-grpo-run explicitly"
            )

        checkpoint_epoch = args.checkpoint_epoch
        if args.last_common_checkpoint:
            checkpoint_epoch = find_last_common_checkpoint_epoch(run_dirs)
            print(f"Using last common checkpoint epoch: {checkpoint_epoch}")

        if checkpoint_epoch is not None:
            out_dir = (
                args.out_dir
                or FINAL_ROOT / "runs" / args.suite / "plots" / f"epoch_{checkpoint_epoch:04d}"
            ).resolve()
        else:
            out_dir = (args.out_dir or FINAL_ROOT / "runs" / args.suite / "plots").resolve()

        dataset = HyperGridDataset.load(dataset_path)

        if checkpoint_epoch is not None:
            plot_subset = None
            if args.plots is not None:
                plot_subset = {p.strip() for p in args.plots.split(",") if p.strip()}
            result = generate_checkpoint_comparison(
                dataset,
                run_dirs=run_dirs,
                checkpoint_epoch=checkpoint_epoch,
                out_dir=out_dir,
                num_samples=args.num_samples,
                device=args.sample_device,
                force_resample=args.force_resample,
                plots=plot_subset,
            )
            print(json.dumps(result, indent=2, default=str))
            return

        method_coords: list[tuple[str, np.ndarray]] = []
        for label in METHOD_ORDER:
            if label not in run_dirs:
                continue
            coords = load_coords_for_run(
                run_dirs[label].resolve(), dataset, num_samples=args.num_samples, device=args.sample_device
            )
            method_coords.append((label, coords))

        all_methods = plot_all_methods_distribution_grid(
            dataset,
            methods=method_coords,
            out_path=out_dir / "terminal_distribution_comparison.png",
        )
        l1_curve = plot_all_methods_training_l1(run_dirs, out_path=out_dir / "training_l1_curve.png")
        reward_curve = plot_all_methods_training_reward(run_dirs, out_path=out_dir / "training_reward_curve.png")
        modes_curve = plot_all_methods_modes_found(
            dataset, method_coords=method_coords, out_path=out_dir / "modes_found_vs_samples.png", x_scale="linear"
        )
        modes_curve_logx = plot_all_methods_modes_found(
            dataset, method_coords=method_coords, out_path=out_dir / "modes_found_vs_samples_logx.png", x_scale="log"
        )
        recovery = write_all_methods_recovery_summary(
            dataset=dataset,
            method_coords=method_coords,
            run_dirs=run_dirs,
            out_path=out_dir / "recovery_summary.json",
        )
        payload = {
            "out_dir": str(out_dir),
            "terminal_distribution_comparison": all_methods,
            "training_l1_curve": str(l1_curve),
            "training_reward_curve": str(reward_curve),
            "modes_found_vs_samples": modes_curve,
            "modes_found_vs_samples_logx": modes_curve_logx,
            "recovery_summary": recovery,
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, default=str))
        return

    if args.suite is not None:
        dataset_path, grpo_run, ips_run, out_dir = resolve_suite_paths(args.suite)
    else:
        dataset_path = args.dataset or DEFAULT_DATASET
        grpo_run = args.grpo_run or DEFAULT_GRPO_RUN
        ips_run = args.ips_run or DEFAULT_IPS_RUN
        out_dir = args.out_dir or DEFAULT_OUT

    dataset = HyperGridDataset.load(dataset_path)
    out_dir = out_dir.resolve()
    grpo_run = grpo_run.resolve()
    ips_run = ips_run.resolve()

    grpo_coords = load_coords_for_run(
        grpo_run, dataset, num_samples=50_000, device=args.sample_device
    )
    ips_coords = load_coords_for_run(
        ips_run, dataset, num_samples=50_000, device=args.sample_device
    )

    comparison = plot_terminal_distribution_grid(
        dataset,
        grpo_coords=grpo_coords,
        ips_coords=ips_coords,
        grpo_run=grpo_run,
        ips_run=ips_run,
        out_path=out_dir / "terminal_distribution_comparison.png",
    )
    l1_curve = plot_training_l1_curve(
        grpo_run=grpo_run,
        ips_run=ips_run,
        out_path=out_dir / "training_l1_curve.png",
    )
    reward_curve = plot_training_reward_curve(
        grpo_run=grpo_run,
        ips_run=ips_run,
        out_path=out_dir / "training_reward_curve.png",
    )
    modes_curve = plot_modes_found_curve(
        dataset,
        grpo_coords=grpo_coords,
        ips_coords=ips_coords,
        out_path=out_dir / "modes_found_vs_samples.png",
    )
    recovery = write_recovery_summary(
        dataset=dataset,
        grpo_coords=grpo_coords,
        ips_coords=ips_coords,
        grpo_run=grpo_run,
        ips_run=ips_run,
        out_path=out_dir / "recovery_summary.json",
    )
    grpo_gt = plot_gt_vs_method(
        dataset,
        coords=grpo_coords,
        method_label="GRPO",
        out_path=out_dir / "gt_vs_grpo.png",
    )
    ips_gt = plot_gt_vs_method(
        dataset,
        coords=ips_coords,
        method_label=METHOD_LABELS["count_ips"],
        out_path=out_dir / "gt_vs_count_ips.png",
    )
    grpo_diag = plot_training_diagnostics(
        grpo_run,
        method_label="GRPO",
        out_path=grpo_run / "plots" / "training_diagnostics.png",
    )
    ips_diag = plot_training_diagnostics(
        ips_run,
        method_label=METHOD_LABELS["count_ips"],
        out_path=ips_run / "plots" / "training_diagnostics.png",
    )

    # Mirror key plots into each run dir for convenience.
    for run_dir, name in ((grpo_run, "terminal_distribution_comparison.png"), (ips_run, "terminal_distribution_comparison.png")):
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        src = out_dir / name
        dst = plots_dir / name
        dst.write_bytes(src.read_bytes())

    shared_l1 = out_dir / "training_l1_curve.png"
    for run_dir in (grpo_run, ips_run):
        (run_dir / "plots" / "training_l1_curve.png").write_bytes(shared_l1.read_bytes())

    print(
        json.dumps(
            {
                **comparison,
                "training_l1_curve": str(l1_curve),
                "training_reward_curve": str(reward_curve),
                "modes_found_vs_samples": modes_curve,
                "recovery_summary": recovery,
                "gt_vs_grpo": grpo_gt,
                "gt_vs_count_ips": ips_gt,
                "grpo_diagnostics": str(grpo_diag),
                "ips_diagnostics": str(ips_diag),
                "out_dir": str(out_dir),
                "grpo_run": str(grpo_run),
                "ips_run": str(ips_run),
            },
            indent=2,
        )
    )

    if args.wandb:
        from final.logging.wandb_logger import FinalWandbLogger, WandbSettings

        suite_id = args.suite or "hypergrid"
        settings = WandbSettings.from_cli(
            enabled=True,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=f"{suite_id}_comparison",
            group=args.wandb_group or suite_id,
            tags=("hypergrid", "comparison"),
        )
        logger = FinalWandbLogger.configure(settings)
        if logger is not None:
            settings.apply_to_env()
            logger.init({"suite": suite_id, "kind": "comparison"})
            for png in sorted(out_dir.glob("*.png")):
                logger.log_plot(png)
            logger.finish()


if __name__ == "__main__":
    main()
