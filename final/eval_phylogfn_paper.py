#!/usr/bin/env python3
"""Evaluate a pristine upstream PhyloGFN run with the paper's estimator."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("-n", "--num-trees", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mll-repeats", type=int, default=10)
    parser.add_argument("--mll-samples", type=int, default=1024)
    return parser.parse_args()


def latest_checkpoint(run_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        path = requested if requested.is_absolute() else run_dir / requested
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint_*.pt in {run_dir / 'checkpoints'}")
    return checkpoints[-1]


def configure_run_imports(run_dir: Path):
    backup_root = (run_dir / "backup").resolve()
    if not (backup_root / "src").is_dir():
        raise FileNotFoundError(f"missing backed-up upstream source: {backup_root / 'src'}")
    sys.path.insert(0, str(backup_root))
    from src.configs.defaults import get_cfg_defaults
    from src.env import build_env
    from src.gfn.build import build_gfn
    from src.gfn.gfn_evaluator import GFNEvaluator
    from src.gfn.rollout_worker_phylo import RolloutWorker
    from src.utils.utils import correct_cfg_data, load_sequences

    return (
        get_cfg_defaults,
        build_env,
        build_gfn,
        GFNEvaluator,
        RolloutWorker,
        correct_cfg_data,
        load_sequences,
    )


def save_fit_plots(
    output_dir: Path,
    log_probability: np.ndarray,
    log_reward: np.ndarray,
    log_z: float,
) -> None:
    order = np.argsort(log_reward)
    line_x = log_reward[order]
    ideal_y = line_x - log_z

    fig, ax = plt.subplots(figsize=(8, 7), dpi=220, constrained_layout=True)
    ax.scatter(log_reward, log_probability, s=15, alpha=0.6, edgecolors="none")
    ax.plot(line_x, ideal_y, "--", color="#777777", label=rf"Ideal: $\log P=\log R-\log Z$, $\log Z={log_z:.2f}$")
    ax.set_xlabel("Unnormalized posterior log density (log reward)")
    ax.set_ylabel("Estimated terminal log probability")
    ax.set_title("Paper PhyloGFN: terminal probability versus posterior density")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "paper_gflownet_log_probability_vs_log_reward.png", bbox_inches="tight")
    plt.close(fig)

    relative_reward = np.exp(log_reward - np.max(log_reward))
    relative_probability = np.exp(log_probability - np.max(log_probability))
    lower = min(float(relative_reward.min()), float(relative_probability.min()))
    fig, ax = plt.subplots(figsize=(8, 7), dpi=220, constrained_layout=True)
    ax.scatter(relative_reward, relative_probability, s=15, alpha=0.6, edgecolors="none")
    ax.plot([lower, 1.0], [lower, 1.0], "--", color="#777777", label="Ideal relative calibration")
    ax.set_xlabel("Relative posterior density")
    ax.set_ylabel("Relative model probability")
    ax.set_title("Paper PhyloGFN: relative sampling calibration")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "paper_gflownet_model_probability_vs_reward.png", bbox_inches="tight")
    plt.close(fig)


def save_pathwise_plots(
    output_dir: Path,
    log_probability: np.ndarray,
    log_reward: np.ndarray,
    log_z: float,
) -> dict[str, float]:
    order = np.argsort(log_reward)
    line_x = log_reward[order]
    ideal_log_probability = line_x - log_z
    log_pearson = float(np.corrcoef(log_reward, log_probability)[0, 1])
    slope, intercept = np.polyfit(log_reward, log_probability, 1)

    count = log_probability.size
    if count >= 100_000:
        scatter_style = {"s": 2.0, "alpha": 0.06}
    elif count >= 10_000:
        scatter_style = {"s": 4.0, "alpha": 0.1}
    else:
        scatter_style = {"s": 10.0, "alpha": 0.2}

    fig, ax = plt.subplots(figsize=(8, 7), dpi=220, constrained_layout=True)
    ax.scatter(
        log_reward,
        log_probability,
        edgecolors="none",
        rasterized=True,
        label=f"{count:,} forward trajectories\nPearson r={log_pearson:.4f}",
        **scatter_style,
    )
    ax.plot(
        line_x,
        ideal_log_probability,
        "--",
        color="#777777",
        label=rf"Ideal: $\log P_F(\tau)/P_B(\tau)=\log R-\log Z$",
    )
    ax.set_xlabel("Unnormalized posterior log density (log reward)")
    ax.set_ylabel(r"Pathwise implied log density: $\log P_F(\tau)-\log P_B(\tau)$")
    ax.set_title("Paper PhyloGFN: pathwise density versus posterior density")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(
        output_dir / "pathwise_log_probability_vs_log_reward.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    reward_center = float(np.max(log_reward))
    relative_reward = np.exp(np.clip(log_reward - reward_center, -745.0, 50.0))
    relative_probability = np.exp(
        np.clip(log_probability - (reward_center - log_z), -745.0, 50.0)
    )
    linear_pearson = float(np.corrcoef(relative_reward, relative_probability)[0, 1])
    upper = max(1.0, float(np.quantile(relative_probability, 0.999)))
    fig, ax = plt.subplots(figsize=(8, 7), dpi=220, constrained_layout=True)
    ax.scatter(
        relative_reward,
        relative_probability,
        edgecolors="none",
        rasterized=True,
        label=f"{count:,} forward trajectories\nPearson r={linear_pearson:.4f}",
        **scatter_style,
    )
    ax.plot([0.0, upper], [0.0, upper], "--", color="#777777", label=r"Ideal: $P(x)\propto R(x)$")
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_xlabel("Relative unnormalized posterior density")
    ax.set_ylabel("Relative pathwise implied density")
    ax.set_title("Paper PhyloGFN: pathwise probability versus reward")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "model_probability_vs_reward.png", bbox_inches="tight")
    plt.close(fig)

    residual = log_probability - (log_reward - log_z)
    return {
        "pathwise_log_probability_vs_log_reward_pearson": log_pearson,
        "pathwise_probability_vs_reward_pearson": linear_pearson,
        "pathwise_log_probability_on_log_reward_slope": float(slope),
        "pathwise_log_probability_on_log_reward_intercept": float(intercept),
        "pathwise_balance_residual_mean": float(residual.mean()),
        "pathwise_balance_residual_std": float(residual.std()),
    }


def main() -> None:
    args = parse_args()
    if args.num_trees <= 0 or args.batch_size <= 0:
        raise ValueError("--num-trees and --batch-size must be positive")
    run_dir = args.run_dir.resolve()
    dataset = args.dataset.resolve()
    checkpoint = latest_checkpoint(run_dir, args.checkpoint)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    (
        get_cfg_defaults,
        build_env,
        build_gfn,
        GFNEvaluator,
        RolloutWorker,
        correct_cfg_data,
        load_sequences,
    ) = configure_run_imports(run_dir)

    all_sequences = load_sequences(str(dataset))
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(run_dir / "config.yaml"))
    cfg.AMP = False
    cfg.LOGGING.ENABLE_TENSORBOARD = False
    cfg = correct_cfg_data(all_sequences, 1, cfg)

    device = torch.device(args.device)
    env = build_env(cfg, all_sequences)
    env.to(device)
    generator = build_gfn(cfg, env, device, ddp=False)
    generator.load(str(checkpoint))
    generator.eval()
    rollout_worker = RolloutWorker(env)
    evaluator = GFNEvaluator(
        cfg.GFN.MODEL.EVALUATION,
        rollout_worker,
        generator,
        verbose=True,
    )

    log_probability, log_reward, pearson = evaluator.evaluate_gfn_quality_pearsonr()
    log_probability = np.asarray(log_probability, dtype=np.float64)
    log_reward = np.asarray(log_reward, dtype=np.float64)
    log_z = float(generator.compute_log_Z().detach().cpu().reshape(-1)[0])

    mll = np.asarray(
        [evaluator.evaluate_marginal_likelihood(args.mll_samples) for _ in range(args.mll_repeats)],
        dtype=np.float64,
    )
    sampled_log_reward = np.empty(args.num_trees, dtype=np.float32)
    sampled_log_score = np.empty(args.num_trees, dtype=np.float32)
    sampled_log_pf = np.empty(args.num_trees, dtype=np.float32)
    sampled_log_pb = np.empty(args.num_trees, dtype=np.float32)
    sampled_pathwise_log_probability = np.empty(args.num_trees, dtype=np.float32)
    generated = 0
    with torch.inference_mode():
        while generated < args.num_trees:
            size = min(args.batch_size, args.num_trees - generated)
            data, _ = rollout_worker.rollout(generator, size, generate_full_trajectories=False)
            stop = generated + size
            batch_log_pf = data["log_paths_pf"].sum(dim=-1)
            batch_log_pb = data["log_paths_pb"].sum(dim=-1)
            sampled_log_reward[generated:stop] = data["log_rewards"].detach().cpu().numpy()
            sampled_log_score[generated:stop] = data["log_scores"].detach().cpu().numpy()
            sampled_log_pf[generated:stop] = batch_log_pf.detach().cpu().numpy()
            sampled_log_pb[generated:stop] = batch_log_pb.detach().cpu().numpy()
            sampled_pathwise_log_probability[generated:stop] = (
                batch_log_pf - batch_log_pb
            ).detach().cpu().numpy()
            generated = stop
            print(f"sampled {generated:,}/{args.num_trees:,}", flush=True)

    output_dir = run_dir / "plots" / f"reward_probability_eval_{args.num_trees}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_fit_plots(output_dir, log_probability, log_reward, log_z)
    pathwise_metrics = save_pathwise_plots(
        output_dir,
        sampled_pathwise_log_probability.astype(np.float64),
        sampled_log_reward.astype(np.float64),
        log_z,
    )
    samples_path = output_dir / "paper_gflownet_evaluation.npz"
    np.savez_compressed(
        samples_path,
        evaluation_log_probability=log_probability,
        evaluation_log_reward=log_reward,
        sampled_log_reward=sampled_log_reward,
        sampled_log_score=sampled_log_score,
        sampled_log_pf=sampled_log_pf,
        sampled_log_pb=sampled_log_pb,
        sampled_pathwise_log_probability=sampled_pathwise_log_probability,
        marginal_log_likelihood=mll,
    )
    residual = log_probability - (log_reward - log_z)
    metrics = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "posterior_samples": args.num_trees,
        "probability_fit_states": int(log_probability.size),
        "backward_trajectories_per_state": int(cfg.GFN.MODEL.EVALUATION.TRAJECTORIES_PER_STATES),
        "log_probability_vs_log_reward_pearson": float(pearson),
        "checkpoint_log_partition": log_z,
        "balance_residual_mean": float(residual.mean()),
        "balance_residual_std": float(residual.std()),
        "marginal_log_likelihood_mean": float(mll.mean()),
        "marginal_log_likelihood_std": float(mll.std()),
        "marginal_log_likelihood_repeats": args.mll_repeats,
        "marginal_log_likelihood_samples_per_repeat": args.mll_samples,
        "samples_file": str(samples_path),
    }
    metrics.update(pathwise_metrics)
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
