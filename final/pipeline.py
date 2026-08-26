"""Train -> sample -> plot pipeline for one method in a suite."""

from __future__ import annotations

import json
import os
from pathlib import Path

from final.configs import SuiteConfig, load_suite
from final.logging.wandb_logger import WandbSettings
from final.methods import get_runner
from final.methods.base import write_json
from final.run_utils import popen_command, run_command, wait_for_run_dir


def resolve_run_dir(
    runner,
    output_root: Path,
    run_name: str,
    *,
    resume_from: Path | None,
    excluded: set[Path] | None = None,
) -> Path:
    if resume_from is not None:
        return resume_from.resolve()
    marker = runner.run_ready_marker(output_root / "placeholder")
    marker_name = marker.name if marker is not None else "resolved_config.yaml"
    if runner.name == "phylgfn":
        return wait_for_run_dir(
            output_root,
            run_name,
            marker_name=marker_name,
            glob_pattern=f"*_{run_name}*",
            excluded=excluded,
        )
    return wait_for_run_dir(
        output_root,
        run_name,
        marker_name=marker_name,
        excluded=excluded,
    )


def _finalize_wandb(
    wandb_logger,
    runner,
    run_dir: Path,
    num_trees: int,
) -> None:
    if wandb_logger is None:
        return
    metrics_path = runner.comparison_metrics_path(run_dir, num_trees)
    if metrics_path.exists():
        comparison = json.loads(metrics_path.read_text(encoding="utf-8"))
        wandb_logger.log_metrics(0, {f"eval/{k}": v for k, v in comparison.items()})
    plots_dir = run_dir / "plots"
    if plots_dir.exists():
        wandb_logger.watch_plot_dirs([plots_dir]).scan_once()
    wandb_logger.finish()


def run_pipeline(
    suite: SuiteConfig,
    method: str,
    *,
    device: str = "cuda:0",
    cuda_device: int = 0,
    skip_train: bool = False,
    skip_sample: bool = False,
    skip_plots: bool = False,
    resume_from: Path | None = None,
    resume_checkpoint: str | None = None,
    run_dir: Path | None = None,
    log_file: Path | None = None,
    wandb_settings: WandbSettings | None = None,
) -> dict:
    from final.logging.wandb_logger import FinalWandbLogger

    runner = get_runner(method)
    output_root = runner.output_root(suite)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = suite.run_name(method)

    wandb_settings = wandb_settings or WandbSettings()
    if wandb_settings.enabled and not wandb_settings.run_name:
        wandb_settings = WandbSettings(
            enabled=wandb_settings.enabled,
            project=wandb_settings.project,
            entity=wandb_settings.entity,
            run_name=f"{suite.id}_{method}",
            group=wandb_settings.group or suite.id,
            tags=wandb_settings.tags,
        )
    wandb_logger = FinalWandbLogger.configure(wandb_settings)
    if wandb_logger is not None:
        wandb_settings.apply_to_env()
        wandb_logger.init(
            {
                "suite_id": suite.id,
                "method": method,
                "taxa": suite.taxa,
                "log_score_shift": suite.log_score_shift,
                "run_name": run_name,
            }
        )

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env["PHYLOGFN_SEED"] = str(suite.training.seed)
    if wandb_settings.enabled:
        env["FINAL_WANDB"] = "1"
        env["WANDB_PROJECT"] = wandb_settings.project
        if wandb_settings.entity:
            env["WANDB_ENTITY"] = wandb_settings.entity
        if wandb_settings.run_name:
            env["WANDB_RUN_NAME"] = wandb_settings.run_name
        env["WANDB_GROUP"] = wandb_settings.group or suite.id
        if wandb_settings.tags:
            env["FINAL_WANDB_TAGS"] = ",".join(wandb_settings.tags)
        if "WANDB_RUN_ID" in os.environ:
            env["WANDB_RUN_ID"] = os.environ["WANDB_RUN_ID"]

    if run_dir is None and not skip_train:
        train_spec = runner.build_train_command(
            suite,
            output_root=output_root,
            run_name=run_name,
            resume_from=resume_from,
            resume_checkpoint=resume_checkpoint,
        )
        train_spec = type(train_spec)(argv=train_spec.argv, cwd=train_spec.cwd, env=env)
        if resume_from is not None:
            if method == "phylgfn":
                excluded = {p.resolve() for p in output_root.iterdir() if p.is_dir()}
                run_command(train_spec, log_file=log_file)
                run_dir = resolve_run_dir(
                    runner,
                    output_root,
                    run_name,
                    resume_from=None,
                    excluded=excluded,
                )
            else:
                run_command(train_spec, log_file=log_file)
                run_dir = resume_from.resolve()
        else:
            excluded = set(output_root.glob("*"))
            proc = popen_command(train_spec, log_file=log_file)
            run_dir = resolve_run_dir(
                runner,
                output_root,
                run_name,
                resume_from=None,
                excluded=excluded,
            )
            exit_code = proc.wait()
            if exit_code != 0:
                raise RuntimeError(f"training failed with exit code {exit_code}: {run_dir}")
    elif run_dir is None:
        raise ValueError("run_dir is required when --skip-train is set")

    run_dir = run_dir.resolve()
    manifest: dict = {
        "suite_id": suite.id,
        "method": method,
        "run_dir": str(run_dir),
        "run_name": run_name,
    }

    checkpoint = runner.expected_checkpoint(run_dir)
    if not skip_train and method != "phylgfn" and not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint after training: {checkpoint}")

    num_trees = suite.sampling.num_trees
    batch_size = suite.sampling.batch_size

    curves_spec = runner.build_training_curves_command(run_dir)
    if curves_spec is not None and not skip_plots:
        curves_spec = type(curves_spec)(argv=curves_spec.argv, cwd=curves_spec.cwd, env=env)
        curves_plot_dir = run_dir / "plots"
        run_command(
            curves_spec,
            log_file=log_file,
            plot_dirs=[curves_plot_dir],
            wandb_logger=wandb_logger,
        )
        manifest["training_curves"] = str(run_dir / "plots" / "training_curves.png")

    if method == "phylgfn":
        phylgfn_plot_dir = run_dir / "plots" / f"reward_probability_eval_{num_trees}"
        if not skip_sample:
            sample_spec = runner.build_sample_command(
                suite,
                run_dir,
                num_trees=num_trees,
                batch_size=batch_size,
                device=device,
            )
            sample_spec = type(sample_spec)(argv=sample_spec.argv, cwd=sample_spec.cwd, env=env)
            run_command(
                sample_spec,
                log_file=log_file,
                plot_dirs=[phylgfn_plot_dir],
                wandb_logger=wandb_logger,
            )
        metrics_path = runner.comparison_metrics_path(run_dir, num_trees)
        if metrics_path.exists():
            manifest["comparison_metrics"] = str(metrics_path)
        manifest["plots_dir"] = str(phylgfn_plot_dir)
        _write_method_manifest(run_dir, manifest)
        _finalize_wandb(wandb_logger, runner, run_dir, num_trees)
        return manifest

    samples_path = run_dir / f"sampled_full_diagnostics_{num_trees}.npz"
    if not skip_sample:
        sample_spec = runner.build_sample_command(
            suite,
            run_dir,
            num_trees=num_trees,
            batch_size=batch_size,
            device=device,
        )
        sample_spec = type(sample_spec)(argv=sample_spec.argv, cwd=sample_spec.cwd, env=env)
        run_command(sample_spec, log_file=log_file)
    manifest["samples"] = str(samples_path)

    if not skip_plots and hasattr(runner, "build_plot_command"):
        plot_spec = runner.build_plot_command(run_dir, samples_path)
        plot_spec = type(plot_spec)(argv=plot_spec.argv, cwd=plot_spec.cwd, env=env)
        metrics_path = runner.comparison_metrics_path(run_dir, num_trees)
        plot_dir = metrics_path.parent
        run_command(
            plot_spec,
            log_file=log_file,
            plot_dirs=[plot_dir],
            wandb_logger=wandb_logger,
        )
        if metrics_path.exists():
            manifest["comparison_metrics"] = str(metrics_path)
        manifest["plots_dir"] = str(metrics_path.parent)

    _write_method_manifest(run_dir, manifest)
    _finalize_wandb(wandb_logger, runner, run_dir, num_trees)
    return manifest


def _write_method_manifest(run_dir: Path, manifest: dict) -> None:
    path = run_dir / "final_manifest.json"
    write_json(path, manifest)


def parse_pipeline_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run train -> sample -> plots for one method in a comparison suite."
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Suite id (e.g. 27taxa_noreplay) or path to suite JSON.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("grpo", "count_ips", "learned_reverse", "phylgfn"),
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run directory (required with --skip-train unless --resume-from).",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training metrics and plots to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="phylogfn-final")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="wandb group (defaults to suite id).",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional wandb tags.",
    )
    return parser.parse_args(argv)


def _wandb_settings_from_args(args) -> WandbSettings:
    return WandbSettings.from_cli(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_pipeline_args(argv)
    suite = load_suite(args.suite)
    manifest = run_pipeline(
        suite,
        args.method,
        device=args.device,
        cuda_device=args.cuda_device,
        skip_train=args.skip_train,
        skip_sample=args.skip_sample,
        skip_plots=args.skip_plots,
        resume_from=args.resume_from,
        resume_checkpoint=args.resume_checkpoint,
        run_dir=args.run_dir,
        log_file=args.log_file,
        wandb_settings=_wandb_settings_from_args(args),
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
