"""Experiment configuration for Hyper-Grid GRPO / IPS runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from final.paths import FINAL_ROOT, REPO_ROOT


DEFAULT_DATASET = FINAL_ROOT / "datasets" / "hypergrid_4096"
DEFAULT_OUTPUT = REPO_ROOT / "final" / "runs" / "hypergrid_4096"


@dataclass
class HypergridExperimentConfig:
    method: str
    dataset: Path
    output: Path
    run_name: str | None
    device: str
    seed: int
    epochs: int
    steps_per_epoch: int
    batch_size: int
    lr: float
    clip_eps: float
    entropy_coef: float
    max_grad_norm: float
    num_iterations: int
    ips_prob_floor: float
    hidden_size: int
    num_layers: int
    checkpoint_every: int
    print_every: int
    resume_from: Path | None
    eval_every: int
    eval_samples: int
    reverse_lr: float = 1e-3
    reverse_train_epochs: int = 4
    reverse_hidden_size: int = 128
    reverse_num_layers: int = 2
    reverse_grad_clip_norm: float = 1.0
    reward_target: str = "likelihood"
    advantage_normalization: str = "batch"

    def output_root(self) -> Path:
        return self.output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GRPO / count IPS on Hyper-Grid.")
    parser.add_argument("--method", choices=("grpo", "count_ips", "learned_reverse_ips", "trajectory_balance"), default="grpo")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--ips-prob-floor", type=float, default=1e-6)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=100_000)
    parser.add_argument("--reverse-lr", type=float, default=1e-3)
    parser.add_argument("--reverse-train-epochs", type=int, default=4)
    parser.add_argument("--reverse-hidden-size", type=int, default=128)
    parser.add_argument("--reverse-num-layers", type=int, default=2)
    parser.add_argument("--reverse-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--reward-target", choices=("likelihood", "shifted_linear"), default="likelihood")
    parser.add_argument("--advantage-normalization", choices=("batch", "running"), default="batch")
    return parser


def config_from_args(args: argparse.Namespace) -> HypergridExperimentConfig:
    device = args.device
    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return HypergridExperimentConfig(
        method=str(args.method),
        dataset=args.dataset.resolve(),
        output=args.output.resolve(),
        run_name=args.run_name,
        device=device,
        seed=int(args.seed),
        epochs=int(args.epochs),
        steps_per_epoch=int(args.steps_per_epoch),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        clip_eps=float(args.clip_eps),
        entropy_coef=float(args.entropy_coef),
        max_grad_norm=float(args.max_grad_norm),
        num_iterations=int(args.num_iterations),
        ips_prob_floor=float(args.ips_prob_floor),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        checkpoint_every=int(args.checkpoint_every),
        print_every=int(args.print_every),
        resume_from=args.resume_from.resolve() if args.resume_from else None,
        eval_every=int(args.eval_every),
        eval_samples=int(args.eval_samples),
        reverse_lr=float(args.reverse_lr),
        reverse_train_epochs=int(args.reverse_train_epochs),
        reverse_hidden_size=int(args.reverse_hidden_size),
        reverse_num_layers=int(args.reverse_num_layers),
        reverse_grad_clip_norm=float(args.reverse_grad_clip_norm),
        reward_target=str(args.reward_target),
        advantage_normalization=str(args.advantage_normalization),
    )


def save_resolved_config(output_dir: Path, cfg: HypergridExperimentConfig) -> None:
    import json

    (output_dir / "resolved_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
