"""Sweep frozen-policy propensity-estimation sizes at fixed optimizer batch size."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class SweepConfig:
    budget: int
    max_step: int
    estimation_sizes: tuple[int, ...]
    optimization_batch_size: int
    num_updates: int
    lr: float
    hidden_size: int
    num_layers: int
    clip_ratio: float
    entropy_coef: float
    seed: int
    log_every: int
    eval_every: int
    eval_episodes: int
    final_samples: int
    checkpoint_every: int
    device: str

    def validate(self) -> None:
        if self.budget < 1 or self.max_step < 1:
            raise ValueError("budget and max_step must be >= 1")
        if not self.estimation_sizes:
            raise ValueError("at least one estimation size is required")
        if len(set(self.estimation_sizes)) != len(self.estimation_sizes):
            raise ValueError("estimation sizes must be unique")
        if any(size < self.optimization_batch_size for size in self.estimation_sizes):
            raise ValueError(
                "every estimation size must be >= optimization_batch_size"
            )


def result_from_summary(
    summary: dict,
    *,
    estimation_size: int,
    num_updates: int,
) -> dict[str, float | int]:
    final = summary["final_sampling"]
    return {
        "estimation_size": estimation_size,
        "total_estimation_rollouts": estimation_size * num_updates,
        "actual_unique_outcomes": int(final["actual_unique_outcomes"]),
        "r2_reward_target": float(final["r2_reward_target"]),
        "tv_reward_target": float(final["tv_reward_target"]),
        "max_abs_prob_error": float(final["max_abs_prob_error"]),
    }


def write_aggregate(
    config: SweepConfig,
    results: list[dict[str, float | int]],
    *,
    output: Path,
) -> None:
    ordered = sorted(results, key=lambda row: int(row["estimation_size"]))
    payload = {
        "sweep_config": asdict(config),
        "completed_runs": ordered,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_aggregate(
    results: list[dict[str, float | int]],
    *,
    output: Path,
) -> None:
    ordered = sorted(results, key=lambda row: int(row["estimation_size"]))
    sizes = [int(row["estimation_size"]) for row in ordered]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    axes[0, 0].plot(
        sizes,
        [float(row["r2_reward_target"]) for row in ordered],
        "o-",
        color="#0984e3",
    )
    axes[0, 0].set_ylabel("R² to reward target")
    axes[0, 0].set_ylim(top=1.02)
    axes[0, 0].set_title("Target-shape fit (higher is better)")

    axes[0, 1].plot(
        sizes,
        [float(row["tv_reward_target"]) for row in ordered],
        "o-",
        color="#d63031",
    )
    axes[0, 1].set_ylabel("TV distance")
    axes[0, 1].set_ylim(bottom=0.0)
    axes[0, 1].set_title("Distribution error (lower is better)")

    axes[1, 0].plot(
        sizes,
        [int(row["actual_unique_outcomes"]) for row in ordered],
        "o-",
        color="#00b894",
    )
    axes[1, 0].set_ylabel("Outcomes hit in final sampling")
    axes[1, 0].set_ylim(bottom=0.0)
    axes[1, 0].set_title("Final terminal coverage")

    axes[1, 1].plot(
        sizes,
        [float(row["max_abs_prob_error"]) for row in ordered],
        "o-",
        color="#6c5ce7",
    )
    axes[1, 1].set_ylabel("Maximum probability error")
    axes[1, 1].set_ylim(bottom=0.0)
    axes[1, 1].set_title("Worst terminal error")

    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xticks(sizes, [str(size) for size in sizes])
        axis.set_xlabel("Frozen-policy estimation size")
        axis.grid(alpha=0.22)
    fig.suptitle("Frozen-behavior Count-IPS estimation-size sweep")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument(
        "--estimation-sizes",
        type=int,
        nargs="+",
        default=(512, 256, 128, 64, 32, 16),
    )
    parser.add_argument("--optimization-batch-size", type=int, default=16)
    parser.add_argument("--num-updates", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--final-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="reuse sub-runs that already contain summary.json",
    )
    args = parser.parse_args()

    config = SweepConfig(
        budget=args.budget,
        max_step=args.max_step,
        estimation_sizes=tuple(args.estimation_sizes),
        optimization_batch_size=args.optimization_batch_size,
        num_updates=args.num_updates,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        final_samples=args.final_samples,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
    )
    config.validate()
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent
        / "data"
        / "frozen_behavior_count_ips_sweeps"
        / (
            f"{datetime.now():%Y%m%d_%H%M%S}_b{config.budget}_"
            f"opt{config.optimization_batch_size}_seed{config.seed}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=args.skip_existing)
    (run_dir / "sweep_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    runner = Path(__file__).resolve().with_name("frozen_behavior_count_ips.py")
    results: list[dict[str, float | int]] = []
    for estimation_size in config.estimation_sizes:
        child_dir = run_dir / f"estimation_{estimation_size:04d}"
        child_summary = child_dir / "summary.json"
        if args.skip_existing and child_summary.is_file():
            print(f"Reusing completed run: {child_dir}")
        else:
            if child_dir.exists():
                raise RuntimeError(
                    f"Incomplete run directory already exists: {child_dir}. "
                    "Move it aside or choose a new --run-dir before resuming."
                )
            command = [
                sys.executable,
                str(runner),
                "--budget",
                str(config.budget),
                "--max-step",
                str(config.max_step),
                "--num-updates",
                str(config.num_updates),
                "--estimation-size",
                str(estimation_size),
                "--optimization-batch-size",
                str(config.optimization_batch_size),
                "--lr",
                str(config.lr),
                "--hidden-size",
                str(config.hidden_size),
                "--num-layers",
                str(config.num_layers),
                "--clip-ratio",
                str(config.clip_ratio),
                "--entropy-coef",
                str(config.entropy_coef),
                "--seed",
                str(config.seed),
                "--log-every",
                str(config.log_every),
                "--eval-every",
                str(config.eval_every),
                "--eval-episodes",
                str(config.eval_episodes),
                "--final-samples",
                str(config.final_samples),
                "--checkpoint-every",
                str(config.checkpoint_every),
                "--device",
                config.device,
                "--run-dir",
                str(child_dir),
            ]
            print(
                f"Starting estimation_size={estimation_size}, "
                f"optimization_batch_size={config.optimization_batch_size}"
            )
            subprocess.run(command, check=True)

        summary = json.loads(child_summary.read_text(encoding="utf-8"))
        result = result_from_summary(
            summary,
            estimation_size=estimation_size,
            num_updates=config.num_updates,
        )
        results.append(result)
        write_aggregate(
            config,
            results,
            output=run_dir / "sweep_summary.json",
        )
        plot_aggregate(
            results,
            output=run_dir / "estimation_size_comparison.png",
        )
        print(json.dumps(result, indent=2))

    print(f"Sweep summary: {run_dir / 'sweep_summary.json'}")
    print(f"Comparison plot: {run_dir / 'estimation_size_comparison.png'}")


if __name__ == "__main__":
    main()
