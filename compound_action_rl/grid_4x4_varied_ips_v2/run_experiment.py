"""
End-to-end IPS-GRPO v2 experiment: train → logs/config/history → training + sampling plots.

Improvements over grid_3x3_varied — extended to 4×4 (1280 trajectories):
  1. Exact trajectory propensities (SNIPS self-normalization)
  2. train_epochs=1, looser clip_ratio
  3. Split per-model PPO losses + per-step color counterfactual credit
  4. Detached state_rep + position auxiliary head on model 1
  5. Enumeration-free log π vs log R diagnostic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import COLOR_PROFILES
from ips_grpo import IPSGRPOTrainer
from plot_sampling import plot_log_scatter, plot_sampling_scatter
from plot_training import plot_training_from_history
from run_output import (
    RunPaths,
    TeeStdout,
    config_payload,
    new_run_dir,
    save_run_config,
)
from train_common import add_train_args, config_from_args, print_train_header


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    add_train_args(parser)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint to resume (can be a prior run's checkpoint.pt)",
    )
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5000, help="Episodes per in-training eval")
    parser.add_argument(
        "--plot-episodes",
        type=int,
        default=5000,
        help="Episodes for final sampling scatter plot",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional suffix for run folder name",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run directory (default: auto timestamp under data/runs/)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training; replot from an existing --run-dir",
    )
    parser.add_argument("--smooth-window", type=int, default=50, help="Rolling mean window for return plot")


def _plot_title(config, *, total_updates: int) -> str:
    return (
        f"IPS-GRPO v2 {config.color_profile} — gs={config.group_size} "
        f"× ng={config.num_groups} — {total_updates} updates"
    )


def _save_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> RunPaths:
    if args.plot_only:
        if args.run_dir is None:
            raise SystemExit("--plot-only requires --run-dir")
        paths = RunPaths(args.run_dir.resolve())
        if not paths.history.exists() or not paths.checkpoint.exists():
            raise SystemExit(f"Missing history or checkpoint in {paths.run_dir}")
        _generate_plots(paths, args, total_updates=_read_total_updates(paths))
        print(f"Plots updated in {paths.run_dir}")
        return paths

    config = config_from_args(args)
    run_dir = args.run_dir or new_run_dir(
        color_profile=config.color_profile,
        group_size=config.group_size,
        run_name=args.run_name,
    )
    paths = RunPaths(run_dir.resolve())

    save_run_config(
        paths.config,
        config_payload(
            config,
            color_profile_spec=COLOR_PROFILES[config.color_profile],
            extra={
                "eval_every": args.eval_every,
                "eval_episodes": args.eval_episodes,
                "plot_episodes": args.plot_episodes,
                "resume_from": str(args.resume_from) if args.resume_from else None,
                "run_name": args.run_name,
            },
        ),
    )

    with TeeStdout(paths.train_log):
        print(f"Run directory: {paths.run_dir}")
        print_train_header(config, agent="IPS-GRPO v2")

        start_step = 0
        if args.resume_from is not None:
            trainer = IPSGRPOTrainer.load(args.resume_from, for_training=True)
            start_step = trainer.loaded_update_step
            print(f"Resumed from {args.resume_from} at update {start_step}")
        else:
            trainer = IPSGRPOTrainer(config)

        def on_eval(step: int, _metrics, tr: IPSGRPOTrainer) -> None:
            tr.save(paths.checkpoint, update_step=step)

        history = trainer.train(
            num_updates=config.num_updates,
            log_every=config.log_every,
            start_step=start_step,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            on_eval=on_eval,
        )
        final_step = start_step + config.num_updates
        ckpt = trainer.save(paths.checkpoint, update_step=final_step)
        paths.history.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Saved checkpoint: {ckpt}")

    total_updates = final_step
    summary = _generate_plots(paths, args, total_updates=total_updates, config=config)
    _save_summary(paths.summary, summary)
    print(f"Done. Artifacts in {paths.run_dir}")
    return paths


def _read_total_updates(paths: RunPaths) -> int:
    if paths.history.exists():
        history = json.loads(paths.history.read_text(encoding="utf-8"))
        if history:
            return int(history[-1]["step"])
    if paths.config.exists():
        payload = json.loads(paths.config.read_text(encoding="utf-8"))
        tc = payload.get("train_config", {})
        return int(tc.get("num_updates", 0))
    return 0


def _generate_plots(
    paths: RunPaths,
    args: argparse.Namespace,
    *,
    total_updates: int,
    config=None,
) -> dict[str, Any]:
    if config is None and paths.config.exists():
        payload = json.loads(paths.config.read_text(encoding="utf-8"))
        tc = payload["train_config"]
        config = type("Cfg", (), tc)()

    title = _plot_title(config, total_updates=total_updates)
    _, evals = plot_training_from_history(
        paths.history,
        paths.training_plot,
        title=title,
        smooth_window=args.smooth_window,
    )

    trainer = IPSGRPOTrainer.load(paths.checkpoint)
    n_traj = trainer._num_trajectories
    sampling = plot_sampling_scatter(
        trainer,
        out_path=paths.sampling_plot,
        episodes=args.plot_episodes,
    )
    log_r2, log_slope = plot_log_scatter(
        trainer,
        out_path=paths.log_scatter,
        episodes=args.plot_episodes,
    )

    best_eval = max(evals, key=lambda e: e.r2) if evals else None
    summary: dict[str, Any] = {
        "total_updates": total_updates,
        "final_sampling": sampling.to_dict(),
        "final_log_scatter": {"log_r2": log_r2, "log_slope": log_slope},
    }
    if best_eval is not None:
        summary["best_eval"] = {
            "step": best_eval.step,
            "r2": best_eval.r2,
            "trajectories_hit": best_eval.trajectories_hit,
            "mean_return": best_eval.mean_return,
            "log_r2": best_eval.log_r2,
            "log_slope": best_eval.log_slope,
        }
    print(f"Training plot: {paths.training_plot}")
    print(f"Sampling plot: {paths.sampling_plot}")
    print(f"Log scatter:   {paths.log_scatter}")
    print(
        f"Final sampling: {sampling.trajectories_hit}/{n_traj} hit, "
        f"R²={sampling.r2:.4f}, mean_ret={sampling.mean_return:.3f}"
    )
    print(f"Final log-log:  R²={log_r2:.4f}, slope={log_slope:.3f} (ideal≈1)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="IPS-GRPO v2 end-to-end experiment pipeline")
    add_experiment_args(parser)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
