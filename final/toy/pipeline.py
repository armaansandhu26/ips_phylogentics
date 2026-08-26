"""Train + evaluate Hyper-Grid GRPO / count IPS from a suite config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from final.logging.wandb_logger import WandbSettings
from final.paths import CONFIGS_DIR, REPO_ROOT
from grpo_experiments.hypergrid.config import HypergridExperimentConfig
from grpo_experiments.hypergrid.runner import run_experiment


def load_hypergrid_suite(path: Path | str) -> dict:
    path = Path(path)
    if path.suffix != ".json":
        candidate = CONFIGS_DIR / f"{path.name}.json"
        path = candidate if candidate.exists() else path.with_suffix(".json")
    return json.loads(path.read_text(encoding="utf-8"))


def suite_to_config(suite: dict, *, method: str, output: Path | None = None) -> HypergridExperimentConfig:
    dataset = (REPO_ROOT / suite["dataset"]).resolve()
    training = suite["training"]
    eval_cfg = suite.get("eval", {})
    out = output or (REPO_ROOT / "final" / "runs" / suite["id"] / method)
    reverse = training.get("learned_reverse", {})
    return HypergridExperimentConfig(
        method=method,
        dataset=dataset,
        output=out.resolve(),
        run_name=suite["id"],
        device="cuda:0",
        seed=int(training.get("seed", 0)),
        epochs=int(training.get("epochs", 100)),
        steps_per_epoch=int(training.get("steps_per_epoch", 10)),
        batch_size=int(training.get("batch_size", 256)),
        lr=float(training.get("lr", 1e-3)),
        clip_eps=float(training.get("clip_eps", 0.2)),
        entropy_coef=float(training.get("entropy_coef", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        num_iterations=int(training.get("num_iterations", 1)),
        ips_prob_floor=float(training.get("ips_prob_floor", 1e-6)),
        hidden_size=int(training.get("hidden_size", 256)),
        num_layers=int(training.get("num_layers", 2)),
        checkpoint_every=int(training.get("checkpoint_every", 10)),
        print_every=int(training.get("print_every", 1)),
        resume_from=None,
        eval_every=int(eval_cfg.get("every_epochs", 10)),
        eval_samples=int(eval_cfg.get("samples", 100_000)),
        reverse_lr=float(reverse.get("reverse_lr", 1e-3)),
        reverse_train_epochs=int(reverse.get("reverse_train_epochs", 4)),
        reverse_hidden_size=int(reverse.get("reverse_hidden_size", 128)),
        reverse_num_layers=int(reverse.get("reverse_num_layers", 2)),
        reverse_grad_clip_norm=float(reverse.get("reverse_grad_clip_norm", 1.0)),
        reward_target=str(reverse.get("reward_target", "likelihood")),
        advantage_normalization=str(reverse.get("advantage_normalization", "batch")),
    )


def _wandb_settings_from_args(args) -> WandbSettings:
    return WandbSettings.from_cli(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyper-Grid pipeline for one method.")
    parser.add_argument("--suite", default="hypergrid_4096")
    parser.add_argument("--method", choices=("grpo", "count_ips", "learned_reverse_ips", "trajectory_balance"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override suite training.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override suite training.batch_size.")
    parser.add_argument("--eval-every", type=int, default=None, help="Override suite eval.every_epochs.")
    parser.add_argument("--eval-samples", type=int, default=None, help="Override suite eval.samples.")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Stream training metrics and live plots to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="phylogfn-final")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="wandb group (defaults to suite id).",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    suite = load_hypergrid_suite(args.suite)
    cfg = suite_to_config(suite, method=args.method, output=args.output)
    if args.resume_from is not None:
        cfg = HypergridExperimentConfig(**{**cfg.__dict__, "resume_from": args.resume_from.resolve()})
    if args.epochs is not None:
        cfg = HypergridExperimentConfig(**{**cfg.__dict__, "epochs": int(args.epochs)})
    if args.batch_size is not None:
        cfg = HypergridExperimentConfig(**{**cfg.__dict__, "batch_size": int(args.batch_size)})
    if args.eval_every is not None:
        cfg = HypergridExperimentConfig(**{**cfg.__dict__, "eval_every": int(args.eval_every)})
    if args.eval_samples is not None:
        cfg = HypergridExperimentConfig(**{**cfg.__dict__, "eval_samples": int(args.eval_samples)})
    if args.device is not None:
        cfg = HypergridExperimentConfig(**{**cfg.__dict__, "device": args.device})

    wandb_settings = _wandb_settings_from_args(args)
    if wandb_settings.enabled and not wandb_settings.group:
        wandb_settings = WandbSettings(
            enabled=wandb_settings.enabled,
            project=wandb_settings.project,
            entity=wandb_settings.entity,
            run_name=wandb_settings.run_name,
            group=args.suite,
            tags=wandb_settings.tags,
        )

    out = run_experiment(cfg, wandb_settings=wandb_settings)
    print(json.dumps({"run_dir": str(out), "method": args.method}, indent=2))


if __name__ == "__main__":
    main()
