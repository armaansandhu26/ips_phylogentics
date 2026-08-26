"""Post-training evaluation: training curves, sampling, sampling plots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "grpo_experiments" / "scripts"

DEFAULT_SAMPLE_SIZE = 10_000
DEFAULT_PLOT_METHOD = "learned-reverse"


def _ensure_scripts_path() -> None:
    scripts = str(SCRIPTS_DIR)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def _log(message: str) -> None:
    print(f"[post_train {time.strftime('%F %T')}] {message}", flush=True)


def _run_python(script: Path | str, args: list[str], *, cwd: Path = REPO_ROOT) -> None:
    cmd = [sys.executable, "-u", str(script), *args]
    _log("running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def plot_training_curves(
    run_dir: Path,
    *,
    output: Path | None = None,
    title: str | None = None,
) -> Path:
    _ensure_scripts_path()
    from plot_learned_reverse_training_curves import plot_training_panel  # noqa: E402

    run_dir = run_dir.resolve()
    output = output or (run_dir / "plots" / "training_curves.png")
    if title is None:
        title = f"Learned-reverse IPS-GRPO — {run_dir.name}"
    summary = plot_training_panel(run_dir, title=title, output=output)
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _log(f"wrote training curves: {output}")
    return output


def sample_checkpoint(
    run_dir: Path,
    *,
    num_trees: int = DEFAULT_SAMPLE_SIZE,
    batch_size: int = 4096,
    seed: int = 0,
    device: str | None = None,
    output: Path | None = None,
) -> Path:
    run_dir = run_dir.resolve()
    output = output or (run_dir / f"sampled_full_diagnostics_{num_trees}.npz")
    args = [
        "--checkpoint",
        str(run_dir),
        "-n",
        str(num_trees),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]
    if device is not None:
        args.extend(["--device", device])
    _run_python(SCRIPTS_DIR / "sample_learned_reverse_full_diagnostics.py", args)
    _log(f"wrote samples: {output}")
    return output


def plot_sampling(
    samples_path: Path,
    *,
    output_dir: Path | None = None,
    plot_method: str = DEFAULT_PLOT_METHOD,
    shared_reference: bool = True,
) -> Path:
    samples_path = samples_path.resolve()
    output_dir = output_dir or (samples_path.parent / "plots" / "sampling")
    args = [
        "--samples",
        str(samples_path),
        "--output-dir",
        str(output_dir),
        "--plot-method",
        plot_method,
    ]
    if shared_reference:
        args.append("--shared-reference")
    _run_python(SCRIPTS_DIR / "plot_full_checkpoint_vs_reward_reference.py", args)
    plot_path = output_dir / "model_probability_vs_reward.png"
    if not plot_path.exists():
        raise FileNotFoundError(f"expected sampling plot at {plot_path}")
    _log(f"wrote sampling plot: {plot_path}")
    return plot_path


def run_post_train_pipeline(
    run_dir: Path | None,
    *,
    num_trees: int = DEFAULT_SAMPLE_SIZE,
    sample_batch_size: int = 4096,
    seed: int = 0,
    device: str | None = None,
    skip_training_curves: bool = False,
    skip_sampling: bool = False,
    skip_sampling_plots: bool = False,
) -> dict[str, str]:
    if run_dir is None:
        raise ValueError("run_dir is required")
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "final_checkpoint.pt").exists():
        raise FileNotFoundError(f"missing final checkpoint: {run_dir / 'final_checkpoint.pt'}")
    if not (run_dir / "learned_reverse_state.pt").exists():
        raise FileNotFoundError(
            f"missing reverse state: {run_dir / 'learned_reverse_state.pt'}"
        )

    artifacts: dict[str, str] = {"run_dir": str(run_dir)}
    _log(f"starting post-train pipeline for {run_dir}")

    if not skip_training_curves:
        training_plot = plot_training_curves(run_dir)
        artifacts["training_curves"] = str(training_plot)
        artifacts["training_curves_meta"] = str(training_plot.with_suffix(".json"))

    samples_path = run_dir / f"sampled_full_diagnostics_{num_trees}.npz"
    if not skip_sampling:
        samples_path = sample_checkpoint(
            run_dir,
            num_trees=num_trees,
            batch_size=sample_batch_size,
            seed=seed,
            device=device,
            output=samples_path,
        )
    elif not skip_sampling_plots and not samples_path.exists():
        raise FileNotFoundError(
            f"--skip-sampling set but samples not found: {samples_path}"
        )
    artifacts["samples"] = str(samples_path)
    metadata_path = samples_path.with_suffix(".json")
    if metadata_path.exists():
        artifacts["samples_metadata"] = str(metadata_path)

    if not skip_sampling_plots:
        sampling_plot = plot_sampling(samples_path)
        artifacts["sampling_plot"] = str(sampling_plot)
        comparison_path = sampling_plot.parent / "comparison_metrics.json"
        if comparison_path.exists():
            artifacts["comparison_metrics"] = str(comparison_path)

    manifest_path = run_dir / "post_train_manifest.json"
    manifest_path.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    artifacts["manifest"] = str(manifest_path)
    _log(f"post-train pipeline complete: {manifest_path}")
    return artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-training evaluation for a learned-reverse IPS run: "
            "training curves, 10k sampling, probability-vs-reward plots."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed run directory containing final_checkpoint.pt.",
    )
    parser.add_argument(
        "-n",
        "--num-trees",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of terminal trees to sample (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument("--sample-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-training-curves", action="store_true")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-sampling-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_post_train_pipeline(
        args.run_dir,
        num_trees=args.num_trees,
        sample_batch_size=args.sample_batch_size,
        seed=args.seed,
        device=args.device,
        skip_training_curves=args.skip_training_curves,
        skip_sampling=args.skip_sampling,
        skip_sampling_plots=args.skip_sampling_plots,
    )


if __name__ == "__main__":
    main()
