#!/usr/bin/env python3
"""End-to-end full-model baseline: train -> verify -> 1M sample -> plots.

Runs plain GRPO or count IPS with ONLY_TRAIN_TREE_MODEL=false, then verifies
signature-level diversity before sampling and plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

SHARED_TRAIN_ARGS = [
    "--output",
    "grpo_experiments/learned_reverse_runs",
    "--seed",
    "0",
    "--device",
    "cuda:0",
    "--epochs",
    "10000",
    "--steps-per-epoch",
    "1",
    "--on-policy-batch-size",
    "4096",
    "--disable-replay",
    "--full-model",
    "--outcome-level",
    "signature",
    "--advantage-reward-mode",
    "log_reward",
    "--grpo-lr",
    "1e-4",
    "--grpo-entropy-coef",
    "0.0",
    "--grpo-num-iterations",
    "1",
    "--rollout-chunk-size",
    "4096",
    "--print-every",
    "25",
    "--checkpoint-every",
    "500",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("grpo", "count_ips"),
        required=True,
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Label appended to the timestamped run directory.",
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--num-trees", type=int, default=1_000_000)
    parser.add_argument("--sample-batch-size", type=int, default=4096)
    parser.add_argument(
        "--cfg",
        default=(
            "src/configs/benchmark_dna_cfgs/discrete_branch_lengths/"
            "cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml"
        ),
    )
    parser.add_argument(
        "--dataset",
        default="dataset/benchmark_datasets/DS1_reduced.pickle",
    )
    parser.add_argument(
        "--reward-shift",
        type=float,
        default=None,
        help="Sampling reward shift; defaults to ENV.LOG_SCORE_SHIFT from --cfg.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Continue training and the E2E pipeline in an existing run directory.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Checkpoint filename inside --resume-from (default: last committed checkpoint).",
    )
    parser.add_argument("--min-batch-unique-topologies", type=int, default=100)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional combined pipeline log path.",
    )
    return parser.parse_args()


def run(cmd: list[str], *, env: dict[str, str] | None = None, log_file: Path | None = None) -> None:
    printable = " ".join(cmd)
    print(f"[e2e] running: {printable}", flush=True)
    if log_file is not None:
        with log_file.open("a") as handle:
            handle.write(f"\n[e2e] {time.strftime('%F %T')} running: {printable}\n")
            handle.flush()
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                check=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        return
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def latest_run_dir(output_root: Path, run_name: str) -> Path:
    matches = sorted(output_root.glob(f"*_{run_name}_*"))
    if not matches:
        raise FileNotFoundError(f"no run directory matching *_{run_name}_* under {output_root}")
    return matches[-1]


def wait_for_run_dir(
    output_root: Path,
    run_name: str,
    timeout_s: int = 300,
    *,
    excluded: set[Path] | None = None,
) -> Path:
    excluded = {path.resolve() for path in (excluded or set())}
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        matches = sorted(
            path
            for path in output_root.glob(f"*_{run_name}_*")
            if path.resolve() not in excluded
        )
        if matches and (matches[-1] / "resolved_config.yaml").exists():
            return matches[-1]
        time.sleep(5)
    raise TimeoutError(f"run directory for {run_name} not created within {timeout_s}s")


def wait_for_first_metric(run_dir: Path, timeout_s: int = 600) -> dict:
    metrics_path = run_dir / "metrics.jsonl"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if metrics_path.exists() and metrics_path.stat().st_size > 0:
            return json.loads(metrics_path.read_text().splitlines()[0])
        time.sleep(5)
    raise TimeoutError(f"first metrics row not written within {timeout_s}s: {run_dir}")


def verify_resolved_config(run_dir: Path) -> None:
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    only_train = resolved["GFN"]["MODEL"]["ONLY_TRAIN_TREE_MODEL"]
    if only_train is not False:
        raise RuntimeError(
            f"{run_dir} resolved ONLY_TRAIN_TREE_MODEL={only_train!r}; expected false"
        )

    experiment = json.loads((run_dir / "experiment_config.json").read_text())
    if experiment.get("only_train_tree_model") is not False:
        raise RuntimeError(
            f"{run_dir} experiment_config only_train_tree_model="
            f"{experiment.get('only_train_tree_model')!r}; expected false"
        )
    if experiment.get("outcome_level") != "signature":
        raise RuntimeError(
            f"{run_dir} outcome_level={experiment.get('outcome_level')!r}; expected signature"
        )


def verify_training_run(
    run_dir: Path,
    first_metric: dict,
    *,
    min_batch_unique_topologies: int,
) -> None:
    verify_resolved_config(run_dir)

    batch_unique = int(first_metric.get("batch_unique_outcomes", 0))
    batch_topologies = int(first_metric.get("batch_unique_topologies", 0))
    if batch_unique <= 500:
        raise RuntimeError(
            f"{run_dir} epoch-0 batch_unique_outcomes={batch_unique}; "
            "expected >>500 for full-model signature training"
        )
    if batch_topologies < min_batch_unique_topologies:
        raise RuntimeError(
            f"{run_dir} epoch-0 batch_unique_topologies={batch_topologies}; "
            f"expected >= {min_batch_unique_topologies}"
        )

    print(
        f"[e2e] verified full-model signature training: "
        f"batch_unique_outcomes={batch_unique}, batch_unique_topologies={batch_topologies}",
        flush=True,
    )


def verify_sampling(run_dir: Path, samples_path: Path, metadata_path: Path) -> None:
    if not samples_path.exists():
        raise FileNotFoundError(f"missing samples: {samples_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("only_train_tree_model") is not False:
        raise RuntimeError(
            f"sampling metadata only_train_tree_model="
            f"{metadata.get('only_train_tree_model')!r}; expected false"
        )
    observed_topologies = int(metadata.get("observed_topologies", 0))
    if observed_topologies < 100:
        raise RuntimeError(
            f"sampling observed_topologies={observed_topologies}; expected near 105"
        )
    if samples_path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"{samples_path} is only {samples_path.stat().st_size} bytes; "
            "full-model 1M sample should be much larger"
        )
    print(
        f"[e2e] verified sampling: topologies={observed_topologies}, "
        f"npz_bytes={samples_path.stat().st_size:,}",
        flush=True,
    )


def build_train_command(
    method: str,
    run_name: str,
    *,
    cfg: str,
    dataset: str,
    resume_from: Path | None = None,
    resume_checkpoint: str | None = None,
) -> list[str]:
    cmd = [str(PYTHON), "-u"]
    if method == "grpo":
        cmd += ["-m", "grpo_experiments.train", "--method", "grpo", "--run-name", run_name]
    else:
        cmd += [
            "-m",
            "grpo_experiments.ips_grpo.train",
            "--run-name",
            run_name,
            "--ips-propensity-mode",
            "count",
            "--ips-prob-floor",
            "1e-6",
            "--policy-loss-mode",
            "ppo",
        ]
    cmd += ["--cfg", cfg, "--dataset", dataset, *SHARED_TRAIN_ARGS]
    if resume_from is not None:
        cmd += ["--resume-from", str(resume_from)]
    if resume_checkpoint is not None:
        cmd += ["--resume-checkpoint", resume_checkpoint]
    return cmd


def main() -> None:
    args = parse_args()
    if args.resume_checkpoint is not None and args.resume_from is None:
        raise ValueError("--resume-checkpoint requires --resume-from")
    if args.resume_from is not None and not args.resume_from.is_dir():
        raise FileNotFoundError(f"resume run directory not found: {args.resume_from}")

    output_root = REPO_ROOT / "grpo_experiments" / "learned_reverse_runs"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    cfg = yaml.safe_load((REPO_ROOT / args.cfg).read_text())
    reward_shift = (
        float(args.reward_shift)
        if args.reward_shift is not None
        else float(cfg["ENV"].get("LOG_SCORE_SHIFT", 3600.0))
    )

    log_file = args.log_file
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "resumed" if args.resume_from is not None else "started"
        with log_file.open("a") as handle:
            handle.write(f"\n[e2e] {mode} {time.strftime('%F %T')} method={args.method}\n")

    train_cmd = build_train_command(
        args.method,
        args.run_name,
        cfg=args.cfg,
        dataset=args.dataset,
        resume_from=args.resume_from,
        resume_checkpoint=args.resume_checkpoint,
    )
    printable = " ".join(train_cmd)
    print(f"[e2e] launching training: {printable}", flush=True)
    existing_run_dirs = set(output_root.glob(f"*_{args.run_name}_*"))
    stdout = log_file.open("a") if log_file is not None else subprocess.DEVNULL
    train_proc = subprocess.Popen(
        train_cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
    )

    run_dir = (
        args.resume_from.resolve()
        if args.resume_from is not None
        else wait_for_run_dir(
            output_root,
            args.run_name,
            excluded=existing_run_dirs,
        )
    )
    verify_resolved_config(run_dir)
    first_metric = wait_for_first_metric(run_dir)
    try:
        verify_training_run(
            run_dir,
            first_metric,
            min_batch_unique_topologies=args.min_batch_unique_topologies,
        )
    except Exception:
        train_proc.terminate()
        train_proc.wait(timeout=30)
        raise

    exit_code = train_proc.wait()
    if exit_code != 0:
        raise RuntimeError(f"training failed with exit code {exit_code}: {run_dir}")

    checkpoint = run_dir / "final_checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing final checkpoint: {checkpoint}")

    samples_path = run_dir / f"sampled_full_diagnostics_{args.num_trees}.npz"
    metadata_path = samples_path.with_suffix(".json")
    sample_cmd = [
        str(PYTHON),
        "-u",
        "grpo_experiments/scripts/sample_ppo_full_diagnostics.py",
        "--checkpoint",
        str(run_dir),
        "-n",
        str(args.num_trees),
        "--batch-size",
        str(args.sample_batch_size),
        "--seed",
        "0",
        "--device",
        "cuda:0",
        "--reward-shift",
        str(reward_shift),
        "--output",
        str(samples_path),
    ]
    run(sample_cmd, env=env, log_file=log_file)
    verify_sampling(run_dir, samples_path, metadata_path)

    plot_dir = run_dir / "plots" / "mlp_shifted_linear_reference_1000k"
    plot_cmd = [
        str(PYTHON),
        "grpo_experiments/scripts/plot_full_checkpoint_vs_reward_reference.py",
        "--samples",
        str(samples_path),
        "--output-dir",
        str(plot_dir),
        "--plot-method",
        "ppo",
        "--shared-reference",
    ]
    run(plot_cmd, log_file=log_file)

    comparison = json.loads((plot_dir / "comparison_metrics.json").read_text())
    unique_signatures = int(comparison["unique_observed_signatures"])
    if unique_signatures <= 500:
        raise RuntimeError(
            f"plot comparison_metrics unique_observed_signatures={unique_signatures}; "
            "expected many signatures for full-model eval"
        )

    print(
        f"[e2e] complete: run_dir={run_dir}\n"
        f"       plots={plot_dir}\n"
        f"       unique_signatures={unique_signatures:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
