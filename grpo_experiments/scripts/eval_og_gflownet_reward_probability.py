#!/usr/bin/env python3
"""Plot reward versus implied terminal probability for an og_code GFlowNet.

This script deliberately imports ``src`` from the selected run's ``backup``
directory. Some og_code experiments changed the reward implementation in-place,
so evaluating with the repository's current og_code source can silently use a
different target from the one used for training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from reward_probability_plot_reference import (  # noqa: E402
    ideal_line_label,
    linear_probability_limits,
    log_probability_limits,
    merge_reward_axis_bounds,
    pearson_vs_ideal_sampling,
    pearson_vs_ideal_sampling_linear,
    shared_reward_axis_bounds,
)


GRAY = "#777777"
BLUE = "#1976d2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/benchmark_datasets/DS1_reduced.pickle"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Default: latest checkpoint_*.pt under <run-dir>/checkpoints.",
    )
    parser.add_argument("-n", "--num-trees", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--scatter-points",
        type=int,
        default=0,
        help="Maximum plotted trajectories; 0 plots all sampled trajectories.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/plots/reward_probability_eval_<n>.",
    )
    parser.add_argument(
        "--reference-run-dir",
        type=Path,
        default=None,
        help="Reference run for shared reward-axis limits.",
    )
    parser.add_argument(
        "--shared-reference",
        action="store_true",
        help=(
            "Use a shared reward-axis range. GFlowNet keeps its checkpoint log Z "
            "for the ideal line unless --reference-log-partition is set."
        ),
    )
    parser.add_argument(
        "--reference-log-partition",
        type=float,
        default=None,
        help="Optional fixed log partition for the ideal line.",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Skip sampling and replot from existing og_gflownet_reward_probability_samples.npz.",
    )
    return parser.parse_args()


def resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        candidate = checkpoint
        if not candidate.is_absolute() and not candidate.exists():
            candidate = run_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"missing checkpoint: {candidate}")
        return candidate
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint_*.pt files under {run_dir / 'checkpoints'}")
    return checkpoints[-1]


def configure_run_imports(run_dir: Path):
    backup_root = (run_dir / "backup").resolve()
    if not (backup_root / "src").is_dir():
        raise FileNotFoundError(f"missing backed-up source tree: {backup_root / 'src'}")
    sys.path.insert(0, str(backup_root))

    from src.configs.defaults import get_cfg_defaults
    from src.env import build_env
    from src.gfn.build import build_gfn
    from src.gfn.rollout_worker_phylo import RolloutWorker
    from src.utils.utils import correct_cfg_data, load_sequences

    return (
        get_cfg_defaults,
        build_env,
        build_gfn,
        RolloutWorker,
        correct_cfg_data,
        load_sequences,
    )


def select_points(size: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or size <= maximum:
        return np.arange(size)
    return np.random.default_rng(seed).choice(size, size=maximum, replace=False)


def scatter_style(num_points: int) -> dict[str, object]:
    if num_points >= 500_000:
        return {"s": 1.0, "alpha": 0.035}
    if num_points >= 100_000:
        return {"s": 2.0, "alpha": 0.06}
    return {"s": 8.0, "alpha": 0.18}


def save_plots(
    output_dir: Path,
    *,
    log_probability: np.ndarray,
    log_reward: np.ndarray,
    checkpoint_log_partition: float,
    unique_signatures: int,
    scatter_points: int,
    seed: int,
    reference_log_partition: float | None = None,
    axis_spec: dict[str, float] | None = None,
    log_score_shift: float = 3600.0,
) -> dict[str, float]:
    selected = select_points(len(log_probability), scatter_points, seed)
    style = scatter_style(len(selected))
    x_log_reward = log_reward[selected]
    y_log_probability = log_probability[selected]
    all_reward = np.exp(log_reward)
    all_probability = np.exp(log_probability)
    reward = all_reward[selected]
    probability = all_probability[selected]
    estimated_log_partition = float(np.mean(log_reward - log_probability))
    line_log_partition = (
        reference_log_partition
        if reference_log_partition is not None
        else checkpoint_log_partition
    )

    log_pearson = pearson_vs_ideal_sampling(
        log_probability,
        log_reward,
        line_log_partition,
    )
    linear_pearson = pearson_vs_ideal_sampling_linear(
        log_probability,
        log_reward,
        line_log_partition,
    )
    slope, intercept = np.polyfit(log_reward, log_probability, 1)
    common_stats = (
        f"{len(log_probability):,} trajectories\n"
        f"{unique_signatures:,} unique signatures"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=220, constrained_layout=True)
    ax.scatter(
        x_log_reward,
        y_log_probability,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
        label=(
            "Original GFlowNet: $P_F(\\tau)/P_B(\\tau)$\n"
            f"{common_stats}\nPearson r vs ideal={log_pearson:.4f}"
        ),
        **style,
    )
    if axis_spec is not None:
        line_x = np.linspace(
            axis_spec["log_reward_min"],
            axis_spec["log_reward_max"],
            200,
        )
    elif x_log_reward.size == 1:
        center = float(x_log_reward[0])
        line_x = np.linspace(center - 1.0, center + 1.0, 200)
    else:
        line_x = np.linspace(float(x_log_reward.min()), float(x_log_reward.max()), 200)
    ax.plot(
        line_x,
        line_x - line_log_partition,
        color=GRAY,
        linestyle="--",
        linewidth=1.3,
        label=ideal_line_label(line_log_partition, show_log_z=True),
    )
    if axis_spec is not None:
        ax.set_xlim(axis_spec["log_reward_min"], axis_spec["log_reward_max"])
    y_min, y_max = log_probability_limits(
        y_log_probability,
        line_x,
        line_log_partition,
    )
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(
        rf"Log terminal reward: $\log R(x)=\log({log_score_shift:g}+\log L(x))$"
    )
    ax.set_ylabel("Pathwise implied log terminal probability")
    ax.set_title("Original GFlowNet: model probability versus terminal reward (log scale)")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "og_gflownet_log_probability_vs_log_reward.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=220, constrained_layout=True)
    ax.scatter(
        reward,
        probability,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
        label=(
            "Original GFlowNet: $P_F(\\tau)/P_B(\\tau)$\n"
            f"{common_stats}\nPearson r vs ideal={linear_pearson:.4f}"
        ),
        **style,
    )
    if axis_spec is not None:
        line_x = np.linspace(axis_spec["reward_min"], axis_spec["reward_max"], 200)
    elif reward.size == 1:
        center = float(reward[0])
        line_x = np.linspace(center - 1.0, center + 1.0, 200)
    else:
        line_x = np.linspace(float(reward.min()), float(reward.max()), 200)
    ax.plot(
        line_x,
        line_x / math.exp(line_log_partition),
        color=GRAY,
        linestyle="--",
        linewidth=1.3,
        label=ideal_line_label(line_log_partition, show_log_z=True),
    )
    if axis_spec is not None:
        ax.set_xlim(axis_spec["reward_min"], axis_spec["reward_max"])
    y_min, y_max = linear_probability_limits(
        probability,
        line_x,
        math.exp(line_log_partition),
    )
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(rf"Terminal reward: $R(x)={log_score_shift:g}+\log L(x)$")
    ax.set_ylabel("Pathwise implied terminal probability")
    ax.set_title("Original GFlowNet: model probability versus terminal reward")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "og_gflownet_model_probability_vs_reward.png", bbox_inches="tight")
    plt.close(fig)

    calibrated_probability = probability * math.exp(estimated_log_partition)
    lower = min(float(reward.min()), float(calibrated_probability.min()))
    upper = max(float(reward.max()), float(calibrated_probability.max()))
    padding = 0.03 * (upper - lower)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=220, constrained_layout=True)
    ax.scatter(
        reward,
        calibrated_probability,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
        label=(
            "Original GFlowNet: $P_F(\\tau)/P_B(\\tau)$\n"
            f"{common_stats}\nPearson r vs ideal={linear_pearson:.4f}"
        ),
        **style,
    )
    ax.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        color=GRAY,
        linestyle="--",
        linewidth=1.3,
        label="Ideal calibration",
    )
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(rf"Terminal reward: $R(x)={log_score_shift:g}+\log L(x)$")
    ax.set_ylabel(r"Partition-calibrated terminal probability: $ZP(x)$")
    ax.set_title("og GFlowNet: linear probability–reward calibration")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(
        output_dir / "og_gflownet_partition_calibrated_probability_vs_reward.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    log_balance_residual = log_reward - log_probability - checkpoint_log_partition
    fitted_residual = log_reward - log_probability - estimated_log_partition
    shifted = np.exp((log_reward - log_probability) - np.max(log_reward - log_probability))
    ess = float(shifted.sum() ** 2 / np.square(shifted).sum())
    metrics = {
        "samples": int(len(log_probability)),
        "plotted_samples": int(len(selected)),
        "unique_observed_signatures": unique_signatures,
        "log_probability_vs_log_reward_pearson_vs_ideal": log_pearson,
        "probability_vs_reward_pearson_vs_ideal": linear_pearson,
        "log_probability_on_log_reward_slope": float(slope),
        "log_probability_on_log_reward_intercept": float(intercept),
        "checkpoint_log_partition": checkpoint_log_partition,
        "importance_estimated_log_partition": estimated_log_partition,
        "checkpoint_log_partition_error": (
            checkpoint_log_partition - estimated_log_partition
        ),
        "checkpoint_balance_residual_mean": float(log_balance_residual.mean()),
        "checkpoint_balance_residual_std": float(log_balance_residual.std()),
        "fitted_balance_residual_std": float(fitted_residual.std()),
        "importance_ess": ess,
        "importance_ess_fraction": ess / len(log_probability),
    }
    if reference_log_partition is not None:
        metrics["reference_log_partition"] = reference_log_partition
        metrics["shared_reward_axes"] = axis_spec is not None
    return metrics


def load_log_score_shift(run_dir: Path) -> float:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return 3600.0
    run_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return float(run_cfg.get("ENV", {}).get("LOG_SCORE_SHIFT", 3600.0))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing run directory: {run_dir}")
    log_score_shift = load_log_score_shift(run_dir)
    reward_target_label = (
        f"R(x) = {log_score_shift:g} + terminal_log_likelihood"
    )
    if args.num_trees <= 0 or args.batch_size <= 0:
        raise ValueError("--num-trees and --batch-size must be positive")

    reference_log_partition = args.reference_log_partition
    axis_spec = None
    reference_run_dir = None
    if args.shared_reference or args.reference_run_dir is not None:
        axis_spec = shared_reward_axis_bounds(args.reference_run_dir)

    output_dir = args.output_dir or (
        run_dir / "plots" / f"reward_probability_eval_{args.num_trees}"
    )
    samples_path = output_dir / "og_gflownet_reward_probability_samples.npz"

    if args.replot_only:
        if not samples_path.exists():
            raise FileNotFoundError(f"missing samples for replot: {samples_path}")
        with np.load(samples_path) as payload:
            log_probability = payload["log_probability"].astype(np.float64)
            log_reward = payload["log_reward"].astype(np.float64)
            raw_log_likelihood = payload["raw_log_likelihood"].astype(np.float64)
            topology_index = payload["topology_index"]
            topology_ids = payload["topology_ids"]
        score_milli = np.rint(raw_log_likelihood * 1000.0).astype(np.int64)
        signature_pairs = np.empty(
            len(log_probability),
            dtype=[("topology", np.int32), ("score_milli", np.int64)],
        )
        signature_pairs["topology"] = topology_index
        signature_pairs["score_milli"] = score_milli
        unique_signatures = int(np.unique(signature_pairs).size)
        if axis_spec is not None:
            axis_spec = merge_reward_axis_bounds(
                axis_spec,
                reward=np.exp(log_reward),
                log_reward=log_reward,
            )
            reference_run_dir = axis_spec["reference_run_dir"]
        checkpoint_log_partition = float(
            np.mean(log_reward - log_probability)
        )
        metrics = save_plots(
            output_dir,
            log_probability=log_probability,
            log_reward=log_reward,
            checkpoint_log_partition=checkpoint_log_partition,
            unique_signatures=unique_signatures,
            scatter_points=args.scatter_points,
            seed=args.seed,
            reference_log_partition=reference_log_partition,
            axis_spec=axis_spec,
            log_score_shift=log_score_shift,
        )
        if reference_run_dir is not None:
            metrics["reference_run_dir"] = reference_run_dir
        metrics.update(
            {
                "run_dir": str(run_dir),
                "samples_file": str(samples_path),
                "reward_target": reward_target_label,
                "observed_topologies": int(len(topology_ids)),
                "replot_only": True,
            }
        )
        metrics_path = output_dir / "comparison_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        for path in sorted(output_dir.iterdir()):
            print(f"wrote {path.resolve()}")
        return

    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config: {config_path}")

    (
        get_cfg_defaults,
        build_env,
        build_gfn,
        RolloutWorker,
        correct_cfg_data,
        load_sequences,
    ) = configure_run_imports(run_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    all_sequences = load_sequences(str(args.dataset))
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(config_path))
    cfg.AMP = False
    cfg.LOGGING.ENABLE_TENSORBOARD = False
    cfg = correct_cfg_data(all_sequences, 1, cfg)
    log_score_shift = float(getattr(cfg.ENV, "LOG_SCORE_SHIFT", log_score_shift))
    reward_target_label = (
        f"R(x) = {log_score_shift:g} + terminal_log_likelihood"
    )

    env = build_env(cfg, all_sequences)
    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    generator.load(str(checkpoint))
    generator.eval()
    rollout_worker = RolloutWorker(env)
    checkpoint_log_partition = float(generator.compute_log_Z().detach().cpu().item())

    print(f"run: {run_dir}")
    print(f"checkpoint: {checkpoint}")
    print(f"backed-up source: {run_dir / 'backup'}")
    print(f"device: {device}")
    print(f"checkpoint log Z: {checkpoint_log_partition:.6f}")
    print(f"sampling {args.num_trees:,} trajectories")

    log_probability = np.empty(args.num_trees, dtype=np.float32)
    log_pf = np.empty(args.num_trees, dtype=np.float32)
    log_pb = np.empty(args.num_trees, dtype=np.float32)
    log_reward = np.empty(args.num_trees, dtype=np.float32)
    raw_log_likelihood = np.empty(args.num_trees, dtype=np.float32)
    topology_index = np.empty(args.num_trees, dtype=np.int32)
    topology_ids: list[str] = []
    topology_lookup: dict[str, int] = {}
    generated = 0
    next_print = 100_000
    with torch.inference_mode():
        while generated < args.num_trees:
            batch_size = min(args.batch_size, args.num_trees - generated)
            batch, trajectories = rollout_worker.rollout(
                generator,
                batch_size,
                generate_full_trajectories=True,
            )
            batch_log_pf = batch["log_paths_pf"].sum(dim=-1)
            batch_log_pb = batch["log_paths_pb"].sum(dim=-1)
            batch_log_probability = batch_log_pf - batch_log_pb
            stop = generated + batch_size
            log_probability[generated:stop] = batch_log_probability.detach().cpu().numpy()
            log_pf[generated:stop] = batch_log_pf.detach().cpu().numpy()
            log_pb[generated:stop] = batch_log_pb.detach().cpu().numpy()
            log_reward[generated:stop] = batch["log_rewards"].detach().cpu().numpy()
            raw_log_likelihood[generated:stop] = batch["log_scores"].detach().cpu().numpy()
            for offset, trajectory in enumerate(trajectories):
                terminal_tree = trajectory.current_state.subtrees[0]
                topology_id = str(terminal_tree.tree_topology_id)
                index = topology_lookup.get(topology_id)
                if index is None:
                    index = len(topology_ids)
                    topology_lookup[topology_id] = index
                    topology_ids.append(topology_id)
                topology_index[generated + offset] = index
            generated = stop
            if generated >= next_print or generated == args.num_trees:
                print(f"sampled {generated:,}/{args.num_trees:,}")
                while next_print <= generated:
                    next_print += 100_000

    if not (
        np.isfinite(log_probability).all()
        and np.isfinite(log_pf).all()
        and np.isfinite(log_pb).all()
        and np.isfinite(log_reward).all()
        and np.isfinite(raw_log_likelihood).all()
    ):
        raise ValueError("evaluation produced NaN/Inf")
    expected_log_reward = np.log(
        log_score_shift + raw_log_likelihood.astype(np.float64)
    )
    max_reward_error = float(
        np.max(np.abs(expected_log_reward - log_reward.astype(np.float64)))
    )
    if max_reward_error > 1e-4:
        raise ValueError(
            f"backed-up run does not match {reward_target_label}: "
            f"maximum log-reward error {max_reward_error}"
        )

    score_milli = np.rint(
        raw_log_likelihood.astype(np.float64) * 1000.0
    ).astype(np.int64)
    signature_pairs = np.empty(
        args.num_trees,
        dtype=[("topology", np.int32), ("score_milli", np.int64)],
    )
    signature_pairs["topology"] = topology_index
    signature_pairs["score_milli"] = score_milli
    unique_signatures = int(np.unique(signature_pairs).size)
    if axis_spec is not None:
        axis_spec = merge_reward_axis_bounds(
            axis_spec,
            reward=np.exp(log_reward.astype(np.float64)),
            log_reward=log_reward.astype(np.float64),
        )
        reference_run_dir = axis_spec["reference_run_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        samples_path,
        log_probability=log_probability,
        log_pf=log_pf,
        log_pb=log_pb,
        log_reward=log_reward,
        raw_log_likelihood=raw_log_likelihood,
        topology_index=topology_index,
        topology_ids=np.asarray(topology_ids),
    )
    metrics = save_plots(
        output_dir,
        log_probability=log_probability.astype(np.float64),
        log_reward=log_reward.astype(np.float64),
        checkpoint_log_partition=checkpoint_log_partition,
        unique_signatures=unique_signatures,
        scatter_points=args.scatter_points,
        seed=args.seed,
        reference_log_partition=reference_log_partition,
        axis_spec=axis_spec,
        log_score_shift=log_score_shift,
    )
    if reference_run_dir is not None:
        metrics["reference_run_dir"] = reference_run_dir
    metrics.update(
        {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "dataset": str(args.dataset),
            "device": str(device),
            "reward_target": reward_target_label,
            "maximum_reward_definition_error": max_reward_error,
            "observed_topologies": len(topology_ids),
        }
    )
    metrics_path = output_dir / "comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for path in sorted(output_dir.iterdir()):
        print(f"wrote {path.resolve()}")


if __name__ == "__main__":
    main()
