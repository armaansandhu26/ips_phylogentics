"""Train one RGFN, GRPO, or MIPS-GRPO molecule model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import REPO_ROOT
from .methods import METHODS, METHOD_NAMES, normalize_method_name
from .upstream import (
    RGFN_COMMIT,
    configure_runtime_environment,
    get_rgfn_commit,
    resolve_rgfn_root,
    validate_rgfn_root,
)


def _json_string(value: str | Path) -> str:
    return json.dumps(str(value))


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, help=f"One of: {', '.join(METHOD_NAMES)}")
    parser.add_argument("--cfg", required=True, help="Gin config, resolved before entering RGFN")
    parser.add_argument("--rgfn-root", default=None)
    parser.add_argument("--output-root", default=str(REPO_ROOT / "molecule_synthesis" / "runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--forward-trajectories", type=int, default=None)
    parser.add_argument("--replay-trajectories", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-reactions", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--reward-mode", choices=("linear", "log"), default="linear")
    parser.add_argument("--count-probability-floor", type=float, default=1e-6)
    parser.add_argument("--reverse-loss-weight", type=float, default=1.0)
    parser.add_argument("--reverse-learning-rate", type=float, default=1e-3)
    parser.add_argument("--reverse-train-epochs", type=int, default=4)
    parser.add_argument("--reverse-grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--advantage-normalization", choices=("batch", "running"), default="running"
    )
    parser.add_argument("--running-scale-decay", type=float, default=0.99)
    parser.add_argument("--advantage-clip", type=float, default=10.0)
    parser.add_argument("--log-ratio-clip", type=float, default=20.0)
    parser.add_argument("--exploration-rate", type=float, default=0.0)
    parser.add_argument("--reward-beta", type=float, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    return parser


def _common_bindings(args: argparse.Namespace, output_root: Path, run_name: str) -> list[str]:
    bindings = [
        f"run_name={_json_string(run_name)}",
        f"user_root_dir={_json_string(output_root)}",
        f"WandbLogger.mode={_json_string(args.wandb_mode)}",
    ]
    optional = {
        "Trainer.n_iterations": args.iterations,
        "Trainer.train_forward_n_trajectories": args.forward_trajectories,
        "Trainer.train_replay_n_trajectories": args.replay_trajectories,
        "Trainer.train_batch_size": args.batch_size,
        "ReactionEnv.max_num_reactions": args.max_reactions,
        "Trainer.device": args.device,
        "TrajectoryBalanceOptimizer.lr": args.learning_rate,
    }
    for name, value in optional.items():
        if value is not None:
            bindings.append(f"{name}={json.dumps(value)}")
    if args.reward_beta is not None:
        bindings.append(f"beta={json.dumps(args.reward_beta)}")
    return bindings


def _method_bindings(args: argparse.Namespace, method: str) -> list[str]:
    spec = METHODS[method]
    if spec.objective == "trajectory_balance":
        return []

    objective_names = {
        "grpo": "GRPOObjective",
        "count_ips_grpo": "CountIPSGRPOObjective",
        "mips_grpo": "MIPSGRPOObjective",
    }
    objective_name = objective_names[spec.objective]
    common = [
        f"objective/gin.singleton.constructor=@{objective_name}",
        "Trainer.objective=@objective/gin.singleton()",
    ]
    on_policy = common + ["train/forward/RandomSampler.policy=%forward_policy"]
    if spec.objective == "grpo":
        return on_policy + [
            "GRPOObjective.forward_policy=%forward_policy",
            "GRPOObjective.backward_policy=@UniformPolicy()",
            f"GRPOObjective.clip_eps={args.clip_eps}",
            f"GRPOObjective.reward_mode={_json_string(args.reward_mode)}",
        ]
    if spec.objective == "count_ips_grpo":
        return on_policy + [
            "CountIPSGRPOObjective.forward_policy=%forward_policy",
            "CountIPSGRPOObjective.backward_policy=@UniformPolicy()",
            f"CountIPSGRPOObjective.clip_eps={args.clip_eps}",
            f"CountIPSGRPOObjective.reward_mode={_json_string(args.reward_mode)}",
            f"CountIPSGRPOObjective.probability_floor={args.count_probability_floor}",
        ]
    forward_lr = args.learning_rate if args.learning_rate is not None else 1e-4
    mips_sampler = (
        [
            "train/forward/RandomSampler.policy=%train_forward_policy",
            f"ExploratoryPolicy.first_policy_weight={1.0 - args.exploration_rate}",
        ]
        if args.exploration_rate > 0.0
        else ["train/forward/RandomSampler.policy=%forward_policy"]
    )
    return common + mips_sampler + [
        "Trainer.optimizer=@MIPSOptimizer()",
        f"MIPSOptimizer.forward_lr={forward_lr}",
        f"MIPSOptimizer.reverse_lr={args.reverse_learning_rate}",
        f"MIPSOptimizer.reverse_train_epochs={args.reverse_train_epochs}",
        f"MIPSOptimizer.reverse_grad_clip_norm={args.reverse_grad_clip_norm}",
        "MIPSGRPOObjective.forward_policy=%forward_policy",
        "MIPSGRPOObjective.backward_policy=%backward_policy",
        f"MIPSGRPOObjective.clip_eps={args.clip_eps}",
        f"MIPSGRPOObjective.reward_mode={_json_string(args.reward_mode)}",
        f"MIPSGRPOObjective.reverse_loss_weight={args.reverse_loss_weight}",
        f"MIPSGRPOObjective.advantage_normalization={_json_string(args.advantage_normalization)}",
        f"MIPSGRPOObjective.running_scale_decay={args.running_scale_decay}",
        f"MIPSGRPOObjective.advantage_clip={args.advantage_clip}",
        f"MIPSGRPOObjective.log_ratio_clip={args.log_ratio_clip}",
        f"MIPSGRPOObjective.exploration_rate={args.exploration_rate}",
        "MIPSGRPOObjective.separate_reverse_updates=True",
    ]


def run(args: argparse.Namespace) -> Path:
    configure_runtime_environment()
    method = normalize_method_name(args.method)
    rgfn_root = resolve_rgfn_root(args.rgfn_root)
    validate_rgfn_root(rgfn_root)

    cfg = Path(args.cfg).expanduser()
    if not cfg.is_absolute():
        cfg = (Path.cwd() / cfg).resolve()
    if not cfg.is_file():
        raise FileNotFoundError(f"Gin config does not exist: {cfg}")

    output_root = Path(args.output_root).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"standalone/{method}/{timestamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # RGFN is intentionally kept as a pinned external checkout. Add it before
    # importing gin/RGFN, while keeping this repository importable after chdir.
    for path in (str(REPO_ROOT), str(rgfn_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import gin
    import torch

    import rgfn  # noqa: F401  # registers upstream gin configurables
    from rgfn.trainer.trainer import Trainer
    from rgfn.utils.helpers import seed_everything

    from molecule_synthesis import objectives  # noqa: F401  # registers our objectives
    from molecule_synthesis import optimizers  # noqa: F401  # registers MIPS optimizer
    from molecule_synthesis import minichem  # noqa: F401  # registers reduced data factory

    bindings = _common_bindings(args, output_root, run_name) + _method_bindings(args, method)
    checkpoint = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else None
    best_metrics: dict[str, float] = {}

    with _working_directory(rgfn_root):
        gin.clear_config()
        seed_everything(args.seed)
        gin.parse_config_files_and_bindings([str(cfg)], bindings=bindings)
        # RGFN's one-hot action embedding writes a derived ``._cache`` tensor
        # into checkpoints after first use. A fresh model has no corresponding
        # parameter, so upstream strict resume rejects an otherwise valid
        # checkpoint. Restore the real model/optimizer state explicitly while
        # dropping only that derived cache, as the sampling loader does.
        trainer = Trainer(resume_path=None)
        if checkpoint is not None:
            checkpoint_dict = torch.load(checkpoint, map_location=trainer.device)
            model_state = {
                key: value
                for key, value in checkpoint_dict["model"].items()
                if not key.endswith("._cache")
            }
            trainer.objective.load_state_dict(model_state)
            trainer.optimizer.optimizer.load_state_dict(checkpoint_dict["optimizer"])
            for state in trainer.optimizer.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(trainer.device)
            if trainer.lr_scheduler is not None and checkpoint_dict["lr_scheduler"] is not None:
                trainer.lr_scheduler.lr_scheduler.load_state_dict(
                    checkpoint_dict["lr_scheduler"]
                )
            trainer.best_valid_metrics = checkpoint_dict["metrics"]
            trainer.start_iteration = int(trainer.best_valid_metrics["epoch"]) + 1
            if trainer.train_replay_buffer is not None and checkpoint_dict["replay_buffer"]:
                trainer.train_replay_buffer.load_state_dict(checkpoint_dict["replay_buffer"])
            print(f"Loaded checkpoint from {trainer.start_iteration} iteration")
        try:
            trainer.logger.log_code("rgfn")
            trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
            trainer.logger.log_to_file(gin.config_str(), "config")
            best_metrics = trainer.train()
        finally:
            trainer.close()
            gin.clear_config()

    manifest = {
        "schema_version": 1,
        "method": method,
        "method_label": METHODS[method].label,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "seed": args.seed,
        "config": str(cfg),
        "rgfn_root": str(rgfn_root),
        "rgfn_commit": get_rgfn_commit(rgfn_root),
        "expected_rgfn_commit": RGFN_COMMIT,
        "best_metrics": best_metrics,
        "bindings": bindings,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"RUN_DIR={run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
