"""
End-to-end merge-toy experiment: train -> logs/config/history -> plots.

Examples
--------
Trajectory-IPS (should converge to the *biased* target pi ∝ m(x) R(x)):
    python run_experiment.py --reward-profile phylo_peaked --propensity-mode exact

Marginal (backward-corrected) IPS (should recover pi ∝ R(x)):
    python run_experiment.py --reward-profile phylo_peaked --propensity-mode marginal

Plain GRPO baseline:
    python run_experiment.py --reward-profile gentle --propensity-mode none
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from catalog import build_catalog, validate_catalog
from config import REWARD_PROFILES
from plots import plot_qhat_vs_logreward, plot_signature_sampling, plot_training_from_history
from run_output import RunPaths, TeeStdout, config_payload, new_run_dir, save_run_config
from trainer import MergeTrainer
from train_common import add_train_args, config_from_args, print_train_header


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    add_train_args(parser)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5000)
    parser.add_argument("--plot-episodes", type=int, default=20000)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=50)


def _title(config, total_updates: int) -> str:
    return (
        f"merge toy n={config.n_leaves} — {config.reward_profile}/{config.propensity_mode} "
        f"— gs={config.group_size}×ng={config.num_groups} — {total_updates} updates"
    )


def run_experiment(args: argparse.Namespace) -> RunPaths:
    if args.plot_only:
        if args.run_dir is None:
            raise SystemExit("--plot-only requires --run-dir")
        paths = RunPaths(args.run_dir.resolve())
        _generate_plots(paths, args, total_updates=_read_total_updates(paths))
        print(f"Plots updated in {paths.run_dir}")
        return paths

    config = config_from_args(args)
    trainer = MergeTrainer(config)
    validate_catalog(trainer.catalog)
    catalog_summary = trainer.catalog.summary()

    run_dir = args.run_dir or new_run_dir(
        reward_profile=config.reward_profile,
        propensity_mode=config.propensity_mode,
        group_size=config.group_size,
        run_name=args.run_name,
    )
    paths = RunPaths(run_dir.resolve())

    save_run_config(
        paths.config,
        config_payload(
            config,
            reward_profile_spec=REWARD_PROFILES[config.reward_profile],
            catalog_summary=catalog_summary,
            extra={
                "eval_every": args.eval_every,
                "eval_episodes": args.eval_episodes,
                "plot_episodes": args.plot_episodes,
                "run_name": args.run_name,
            },
        ),
    )

    with TeeStdout(paths.train_log):
        print(f"Run directory: {paths.run_dir}")
        print_train_header(config, catalog_summary)

        def on_eval(step: int, _metrics, tr: MergeTrainer) -> None:
            tr.save(paths.checkpoint, update_step=step)

        history = trainer.train(
            num_updates=config.num_updates,
            log_every=config.log_every,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            on_eval=on_eval,
        )
        ckpt = trainer.save(paths.checkpoint, update_step=config.num_updates)
        paths.history.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Saved checkpoint: {ckpt}")

    summary = _generate_plots(paths, args, total_updates=config.num_updates, config=config)
    paths.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Artifacts in {paths.run_dir}")
    return paths


def _read_total_updates(paths: RunPaths) -> int:
    if paths.history.exists():
        history = json.loads(paths.history.read_text(encoding="utf-8"))
        if history:
            return int(history[-1]["step"])
    return 0


def _generate_plots(paths: RunPaths, args: argparse.Namespace, *, total_updates: int, config=None) -> dict[str, Any]:
    if config is None:
        payload = json.loads(paths.config.read_text(encoding="utf-8"))
        from config import TrainConfig

        config = TrainConfig(**payload["train_config"])

    title = _title(config, total_updates)
    _, evals = plot_training_from_history(paths.history, paths.training_plot, title=title, smooth_window=args.smooth_window)

    trainer = MergeTrainer.load(paths.checkpoint)
    sampling = plot_signature_sampling(trainer, out_path=paths.sampling_plot, episodes=args.plot_episodes, title=title)
    logq_r2, logq_slope = plot_qhat_vs_logreward(trainer, out_path=paths.logq_plot, episodes=args.plot_episodes)

    summary = {
        "total_updates": total_updates,
        "catalog": trainer.catalog.summary(),
        "final_sampling": sampling.to_dict(),
        "final_logq": {"logq_r2": logq_r2, "logq_slope": logq_slope},
    }
    if evals:
        best = max(evals, key=lambda e: e["eval_r2_marginal"])
        summary["best_eval_marginal_r2"] = {"step": best["step"], "r2_marginal": best["eval_r2_marginal"], "r2_ips": best["eval_r2_ips"]}

    print(f"Training plot: {paths.training_plot}")
    print(f"Sampling plot: {paths.sampling_plot}")
    print(f"log q̂ plot:    {paths.logq_plot}")
    print(
        f"Final: {sampling.signatures_hit}/{sampling.num_topologies} topologies hit  "
        f"R²(marginal ∝R)={sampling.r2_marginal:.4f}  R²(biased ∝mR)={sampling.r2_ips:.4f}  "
        f"log q̂-vs-log R slope={logq_slope:.3f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge (phylo) toy end-to-end experiment")
    add_experiment_args(parser)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
