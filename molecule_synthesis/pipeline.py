"""Run comparable molecule-synthesis methods from one suite config."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .aggregate import aggregate_suite
from .config import REPO_ROOT, list_suites, load_suite
from .methods import METHOD_NAMES, normalize_method_name
from .upstream import resolve_rgfn_root, validate_rgfn_root


def build_train_command(
    *,
    python: str,
    method: str,
    cfg: Path,
    rgfn_root: Path,
    output_root: Path,
    run_name: str,
    training: dict,
) -> list[str]:
    command = [
        python,
        "-m",
        "molecule_synthesis.train",
        "--method",
        method,
        "--cfg",
        str(cfg),
        "--rgfn-root",
        str(rgfn_root),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
    ]
    flags = {
        "seed": "--seed",
        "iterations": "--iterations",
        "forward_trajectories": "--forward-trajectories",
        "replay_trajectories": "--replay-trajectories",
        "batch_size": "--batch-size",
        "max_reactions": "--max-reactions",
        "device": "--device",
        "learning_rate": "--learning-rate",
        "clip_eps": "--clip-eps",
        "reward_mode": "--reward-mode",
        "count_probability_floor": "--count-probability-floor",
        "reverse_loss_weight": "--reverse-loss-weight",
        "reverse_learning_rate": "--reverse-learning-rate",
        "reverse_train_epochs": "--reverse-train-epochs",
        "reverse_grad_clip_norm": "--reverse-grad-clip-norm",
        "advantage_normalization": "--advantage-normalization",
        "running_scale_decay": "--running-scale-decay",
        "advantage_clip": "--advantage-clip",
        "log_ratio_clip": "--log-ratio-clip",
        "exploration_rate": "--exploration-rate",
        "reward_beta": "--reward-beta",
        "wandb_mode": "--wandb-mode",
    }
    for key, flag in flags.items():
        value = training.get(key)
        if value is not None:
            command.extend([flag, str(value)])
    return command


def _select_methods(requested: str, configured: tuple[str, ...]) -> tuple[str, ...]:
    if requested == "all":
        return configured
    method = normalize_method_name(requested)
    if method not in configured:
        raise ValueError(f"Method {method!r} is not enabled by this suite")
    return (method,)


def _build_sample_command(
    *,
    python: str,
    run_dir: Path,
    sampling: dict,
    evaluation: dict,
    rgfn_root: Path,
    device: str | None,
    target_path: Path | None,
) -> list[str]:
    command = [
        python,
        "-m",
        "molecule_synthesis.sample",
        "--run-dir",
        str(run_dir),
        "--n-samples",
        str(sampling["n_samples"]),
        "--batch-size",
        str(sampling["batch_size"]),
        "--rgfn-root",
        str(rgfn_root),
    ]
    if device is not None:
        command.extend(["--device", device])
    if target_path is not None:
        command.extend(["--target-json", str(target_path)])
    flags = {
        "mode_threshold": "--mode-threshold",
        "similarity_threshold": "--similarity-threshold",
        "max_modes": "--max-modes",
        "top_k": "--top-k",
    }
    for key, flag in flags.items():
        if key in evaluation:
            command.extend([flag, str(evaluation[key])])
    if "scaffold_thresholds" in evaluation:
        command.extend(
            [
                "--scaffold-thresholds",
                ",".join(str(value) for value in evaluation["scaffold_thresholds"]),
            ]
        )
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List configured suites")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--method", default="all", help=f"all or one of: {', '.join(METHOD_NAMES)}")
    parser.add_argument("--rgfn-root", default=None)
    parser.add_argument("--output-root", default=str(REPO_ROOT / "molecule_synthesis" / "runs"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default=None, help="Override suite device")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="Override suite W&B mode (use disabled for local/socket-restricted runs)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Run one seed instead of all suite seeds")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for name in list_suites():
            print(name)
        return 0
    if not args.suite:
        raise SystemExit("--suite is required unless --list is used")

    suite = load_suite(args.suite)
    rgfn_root = resolve_rgfn_root(args.rgfn_root)
    validate_rgfn_root(rgfn_root)
    cfg = suite.resolve_config(rgfn_root)
    if not cfg.is_file():
        raise FileNotFoundError(f"Suite config does not exist: {cfg}")

    base_training = dict(suite.training)
    if args.device is not None:
        base_training["device"] = args.device
    if args.wandb_mode is not None:
        base_training["wandb_mode"] = args.wandb_mode
    methods = _select_methods(args.method, suite.methods)
    seeds = (args.seed,) if args.seed is not None else suite.seeds
    output_root = Path(args.output_root).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_dir = output_root / suite.suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    run_dirs: dict[str, dict[str, str]] = {}
    target_path: Path | None = None

    if suite.enumeration is not None:
        target_path = suite_dir / "target_distribution.json"
        enumeration_command = [
            args.python,
            "-m",
            "molecule_synthesis.enumerate_space",
            "--output",
            str(target_path),
            "--rgfn-root",
            str(rgfn_root),
        ]
        enumeration_flags = {
            "beta": "--beta",
            "max_reactions": "--max-reactions",
            "fragments_per_group": "--fragments-per-group",
            "max_partial_trajectories": "--max-partial-trajectories",
        }
        for key, flag in enumeration_flags.items():
            if key in suite.enumeration:
                enumeration_command.extend([flag, str(suite.enumeration[key])])
        print(shlex.join(enumeration_command), flush=True)
        if not args.dry_run:
            subprocess.run(enumeration_command, cwd=REPO_ROOT, check=True)

    for seed in seeds:
        for method in methods:
            training = dict(base_training)
            training.update(suite.method_overrides.get(method, {}))
            training["seed"] = seed
            run_name = f"{suite.suite_id}/{method}/seed_{seed}/{timestamp}"
            command = build_train_command(
                python=args.python,
                method=method,
                cfg=cfg,
                rgfn_root=rgfn_root,
                output_root=output_root,
                run_name=run_name,
                training=training,
            )
            print(shlex.join(command), flush=True)
            if args.dry_run:
                if not args.skip_sample:
                    sample_command = _build_sample_command(
                        python=args.python,
                        run_dir=output_root / run_name,
                        sampling=suite.sampling,
                        evaluation=suite.evaluation,
                        rgfn_root=rgfn_root,
                        device=args.device,
                        target_path=target_path,
                    )
                    print(shlex.join(sample_command), flush=True)
                continue
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            run_dir = output_root / run_name
            manifest = run_dir / "manifest.json"
            checkpoint = run_dir / "train" / "checkpoints" / "last_gfn.pt"
            if not manifest.is_file() or not checkpoint.is_file():
                raise RuntimeError(
                    f"{method} finished without required artifacts: {manifest}, {checkpoint}"
                )
            if not args.skip_sample:
                sample_command = _build_sample_command(
                    python=args.python,
                    run_dir=run_dir,
                    sampling=suite.sampling,
                    evaluation=suite.evaluation,
                    rgfn_root=rgfn_root,
                    device=args.device,
                    target_path=target_path,
                )
                print(shlex.join(sample_command), flush=True)
                subprocess.run(sample_command, cwd=REPO_ROOT, check=True)
                summary = run_dir / "samples" / "summary.json"
                if not summary.is_file():
                    raise RuntimeError(f"Sampling finished without summary: {summary}")
            run_dirs.setdefault(method, {})[str(seed)] = str(run_dir)

    if args.dry_run:
        return 0

    manifest_path = suite_dir / "suite.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        if previous_manifest.get("suite") == suite.suite_id:
            for method, seed_runs in previous_manifest.get("runs", {}).items():
                if isinstance(seed_runs, str):
                    continue
                for seed, run_dir in seed_runs.items():
                    run_dirs.setdefault(method, {}).setdefault(seed, run_dir)

    completed_seeds = sorted(
        {
            int(seed)
            for method_runs in run_dirs.values()
            for seed in method_runs
        }
    )
    suite_manifest = {
        "schema_version": 2,
        "suite": suite.suite_id,
        "suite_config": str(suite.source_path),
        "rgfn_config": str(cfg),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": run_dirs,
        "seeds": completed_seeds,
        "target_distribution": str(target_path) if target_path is not None else None,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(suite_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"SUITE_MANIFEST={suite_dir / 'suite.json'}")
    if not args.skip_sample:
        comparison_json, comparison_csv = aggregate_suite(suite_dir)
        print(f"COMPARISON_JSON={comparison_json}")
        print(f"COMPARISON_CSV={comparison_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
