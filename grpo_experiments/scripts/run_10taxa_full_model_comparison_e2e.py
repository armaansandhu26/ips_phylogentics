#!/usr/bin/env python3
"""Train -> verify -> 1M sample -> plot for 10-taxa full-model comparison runs.

Logs pipeline output under grpo_experiments/comparisons/10taxa/ and copies the
final probability-vs-reward plot into that folder when complete.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from taxa_comparison_utils import (  # noqa: E402
    load_num_taxa,
    num_labeled_rooted_tree_topologies,
    summarize_gflownet_health_from_log,
    summarize_learned_reverse_health,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
OG_CODE = REPO_ROOT / "og_code"
COMPARISONS_DIR = REPO_ROOT / "grpo_experiments/comparisons/10taxa"
DATASET = REPO_ROOT / "dataset/benchmark_datasets/DS1_reduced_10taxa.pickle"
OG_DATASET = OG_CODE / "dataset/benchmark_datasets/DS1_reduced_10taxa.pickle"

LEARNED_REVERSE_RUN_NAME = "learned_reverse_10taxa_mlp_shifted_linear_b4096"
GFLOWNET_RUN_NAME = "phylgfn_logreward_10taxa_g4096_1m_full_replay_op3277_r819_rb4096"
LOG_SCORE_SHIFT = 5000.0
LEARNED_REVERSE_CFG = (
    "src/configs/benchmark_dna_cfgs/discrete_branch_lengths/"
    "cfg_0.001binsize_50bins_temperature_anneal_0.4_10taxa_shift5000.yaml"
)
GFLOWNET_CFG = (
    "src/configs/benchmark_dna_cfgs/discrete_branch_lengths/"
    "cfg_ds1_full_g4096_replay_op3277_r819_rb4096_10taxa_shift5000.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("learned_reverse", "gflownet"),
        required=True,
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--num-trees", type=int, default=1_000_000)
    parser.add_argument("--sample-batch-size", type=int, default=4096)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse an existing run directory and only sample/plot.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run directory when using --skip-train.",
    )
    return parser.parse_args()


def ensure_dataset_copies() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"missing dataset: {DATASET}")
    OG_DATASET.parent.mkdir(parents=True, exist_ok=True)
    if not OG_DATASET.exists():
        shutil.copy2(DATASET, OG_DATASET)


def log_line(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.strftime('%F %T')}] {message}\n")
    print(message, flush=True)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    printable = " ".join(cmd)
    log_line(log_path, f"running: {printable}")
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def wait_for_learned_reverse_run(run_name: str, timeout_s: int = 600) -> Path:
    output_root = REPO_ROOT / "grpo_experiments/learned_reverse_runs"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        matches = sorted(output_root.glob(f"*_{run_name}_*"))
        if matches and (matches[-1] / "resolved_config.yaml").exists():
            return matches[-1]
        time.sleep(5)
    raise TimeoutError(f"learned-reverse run not created within {timeout_s}s")


def wait_for_gflownet_run(run_name: str, timeout_s: int = 600) -> Path:
    output_root = OG_CODE / "experiments/full_model"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        matches = sorted(output_root.glob(f"*_{run_name}"))
        if matches and (matches[-1] / "config.yaml").exists():
            return matches[-1]
        time.sleep(5)
    raise TimeoutError(f"gflownet run not created within {timeout_s}s")


def wait_for_first_metric(run_dir: Path, timeout_s: int = 900) -> dict:
    metrics_path = run_dir / "metrics.jsonl"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if metrics_path.exists() and metrics_path.stat().st_size > 0:
            return json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
        time.sleep(5)
    raise TimeoutError(f"first metrics row not written within {timeout_s}s: {run_dir}")


def verify_resolved_config(run_dir: Path) -> None:
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if resolved["GFN"]["MODEL"]["ONLY_TRAIN_TREE_MODEL"] is not False:
        raise RuntimeError("ONLY_TRAIN_TREE_MODEL is not false")
    experiment = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
    if experiment.get("only_train_tree_model") is not False:
        raise RuntimeError("experiment_config.only_train_tree_model is not false")
    if experiment.get("outcome_level") != "signature":
        raise RuntimeError("expected outcome_level=signature")
    if experiment.get("reverse_policy_type") != "mlp":
        raise RuntimeError("expected reverse_policy_type=mlp for 10 taxa")


def verify_epoch0_metrics(first_metric: dict, *, num_taxa: int) -> None:
    batch_unique = int(first_metric.get("batch_unique_outcomes", 0))
    batch_topologies = int(first_metric.get("batch_unique_topologies", 0))
    ess = float(first_metric.get("ips_ess_fraction", 0.0))
    if batch_unique <= 500:
        raise RuntimeError(
            f"epoch-0 batch_unique_outcomes={batch_unique}; expected >>500"
        )
    min_topologies = min(4096, max(500, num_taxa * 50))
    if batch_topologies < min_topologies:
        raise RuntimeError(
            f"epoch-0 batch_unique_topologies={batch_topologies}; "
            f"expected >= {min_topologies}"
        )
    if ess < 0.05:
        raise RuntimeError(f"epoch-0 ips_ess_fraction={ess:.4f}; expected healthy ESS")


def verify_sampling_metadata(metadata: dict, *, num_taxa: int) -> None:
    if metadata.get("only_train_tree_model") is not False:
        raise RuntimeError("sampling metadata indicates tree-only checkpoint")
    observed_topologies = int(metadata.get("observed_topologies", 0))
    min_topologies = max(1000, num_taxa * 100)
    if observed_topologies < min_topologies:
        raise RuntimeError(
            f"observed_topologies={observed_topologies}; expected >= {min_topologies}"
        )


def update_manifest(method: str, payload: dict) -> None:
    manifest_path = COMPARISONS_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "dataset": "DS1_reduced_10taxa.pickle",
            "num_taxa": load_num_taxa(DATASET),
            "theoretical_topologies": num_labeled_rooted_tree_topologies(
                load_num_taxa(DATASET)
            ),
            "methods": {},
        }
    manifest["methods"][method] = payload
    write_json(manifest_path, manifest)


def learned_reverse_train_cmd(run_name: str) -> list[str]:
    return [
        str(PYTHON),
        "-u",
        "-m",
        "grpo_experiments.learned_reverse_ips_grpo",
        "--run-name",
        run_name,
        "--cfg",
        LEARNED_REVERSE_CFG,
        "--dataset",
        "dataset/benchmark_datasets/DS1_reduced_10taxa.pickle",
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
        "--learn-edge-lengths",
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
        "--reward-target",
        "shifted_linear",
        "--reverse-policy-type",
        "mlp",
    ]


def gflownet_train_cmd(run_name: str) -> list[str]:
    return [
        str(PYTHON),
        "-u",
        "train.py",
        GFLOWNET_CFG,
        "dataset/benchmark_datasets/DS1_reduced_10taxa.pickle",
        f"experiments/full_model/{run_name}",
    ]


def sample_and_plot_learned_reverse(
    run_dir: Path,
    *,
    num_taxa: int,
    num_trees: int,
    sample_batch_size: int,
    env: dict[str, str],
    log_path: Path,
) -> Path:
    samples_path = run_dir / f"sampled_full_diagnostics_{num_trees}.npz"
    metadata_path = samples_path.with_suffix(".json")
    plot_dir = run_dir / "plots/mlp_shifted_linear_reference_1000k"

    run_cmd(
        [
            str(PYTHON),
            "-u",
            "grpo_experiments/scripts/sample_learned_reverse_full_diagnostics.py",
            "--checkpoint",
            str(run_dir),
            "-n",
            str(num_trees),
            "--batch-size",
            str(sample_batch_size),
            "--seed",
            "0",
            "--device",
            "cuda:0",
            "--output",
            str(samples_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        log_path=log_path,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    verify_sampling_metadata(metadata, num_taxa=num_taxa)

    run_cmd(
        [
            str(PYTHON),
            "grpo_experiments/scripts/plot_full_checkpoint_vs_reward_reference.py",
            "--samples",
            str(samples_path),
            "--output-dir",
            str(plot_dir),
            "--plot-method",
            "learned-reverse",
            "--shared-reference",
        ],
        cwd=REPO_ROOT,
        env=env,
        log_path=log_path,
    )
    return plot_dir / "model_probability_vs_reward.png"


def sample_and_plot_gflownet(
    run_dir: Path,
    *,
    num_trees: int,
    sample_batch_size: int,
    env: dict[str, str],
    log_path: Path,
) -> Path:
    plot_dir = run_dir / f"plots/reward_probability_eval_{num_trees}"
    run_cmd(
        [
            str(PYTHON),
            "-u",
            "grpo_experiments/scripts/eval_og_gflownet_reward_probability.py",
            "--run-dir",
            str(run_dir),
            "-n",
            str(num_trees),
            "--batch-size",
            str(sample_batch_size),
            "--seed",
            "0",
            "--device",
            "cuda:0",
            "--output-dir",
            str(plot_dir),
            "--shared-reference",
        ],
        cwd=REPO_ROOT,
        env=env,
        log_path=log_path,
    )
    return plot_dir / "og_gflownet_model_probability_vs_reward.png"


def main() -> None:
    args = parse_args()
    ensure_dataset_copies()
    num_taxa = load_num_taxa(DATASET)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    method_key = args.method
    log_path = COMPARISONS_DIR / "logs" / f"{method_key}_e2e.log"
    status_path = COMPARISONS_DIR / "status" / f"{method_key}.json"
    plot_copy_dir = COMPARISONS_DIR / method_key

    payload: dict = {
        "method": method_key,
        "dataset": str(DATASET.relative_to(REPO_ROOT)),
        "num_taxa": num_taxa,
        "log_score_shift": LOG_SCORE_SHIFT,
        "reward": f"R(x) = {LOG_SCORE_SHIFT:g} + log L(x)",
        "started_at": time.strftime("%F %T"),
        "cuda_device": args.cuda_device,
    }
    write_json(status_path, payload)
    log_line(log_path, f"starting 10-taxa {method_key} pipeline (log_score_shift={LOG_SCORE_SHIFT:g})")

    if args.method == "learned_reverse":
        run_name = LEARNED_REVERSE_RUN_NAME
        if args.skip_train:
            if args.run_dir is None:
                raise ValueError("--run-dir is required with --skip-train")
            run_dir = args.run_dir.resolve()
        else:
            train_proc = subprocess.Popen(
                learned_reverse_train_cmd(run_name),
                cwd=REPO_ROOT,
                env=env,
                stdout=log_path.open("a"),
                stderr=subprocess.STDOUT,
            )
            run_dir = wait_for_learned_reverse_run(run_name)
            payload["run_dir"] = str(run_dir)
            write_json(status_path, payload)
            verify_resolved_config(run_dir)
            first_metric = wait_for_first_metric(run_dir)
            verify_epoch0_metrics(first_metric, num_taxa=num_taxa)
            log_line(
                log_path,
                "epoch-0 ok: "
                f"batch_unique_outcomes={first_metric['batch_unique_outcomes']} "
                f"batch_unique_topologies={first_metric['batch_unique_topologies']} "
                f"ips_ess_fraction={first_metric['ips_ess_fraction']:.4f}",
            )
            exit_code = train_proc.wait()
            if exit_code != 0:
                raise RuntimeError(f"training failed with exit code {exit_code}")
        if not (run_dir / "final_checkpoint.pt").exists():
            raise FileNotFoundError(f"missing final checkpoint: {run_dir / 'final_checkpoint.pt'}")
        health = summarize_learned_reverse_health(run_dir)
        write_json(COMPARISONS_DIR / "health" / f"{method_key}.json", health)
        plot_path = sample_and_plot_learned_reverse(
            run_dir,
            num_taxa=num_taxa,
            num_trees=args.num_trees,
            sample_batch_size=args.sample_batch_size,
            env=env,
            log_path=log_path,
        )
        comparison = json.loads(
            (plot_path.parent / "comparison_metrics.json").read_text(encoding="utf-8")
        )
    else:
        run_name = GFLOWNET_RUN_NAME
        if args.skip_train:
            if args.run_dir is None:
                raise ValueError("--run-dir is required with --skip-train")
            run_dir = args.run_dir.resolve()
        else:
            train_proc = subprocess.Popen(
                gflownet_train_cmd(run_name),
                cwd=OG_CODE,
                env=env,
                stdout=log_path.open("a"),
                stderr=subprocess.STDOUT,
            )
            run_dir = wait_for_gflownet_run(run_name)
            payload["run_dir"] = str(run_dir)
            write_json(status_path, payload)
            exit_code = train_proc.wait()
            if exit_code != 0:
                raise RuntimeError(f"training failed with exit code {exit_code}")
        health = summarize_gflownet_health_from_log(log_path)
        write_json(COMPARISONS_DIR / "health" / f"{method_key}.json", health)
        plot_path = sample_and_plot_gflownet(
            run_dir,
            num_trees=args.num_trees,
            sample_batch_size=args.sample_batch_size,
            env=env,
            log_path=log_path,
        )
        comparison = json.loads(
            (plot_path.parent / "comparison_metrics.json").read_text(encoding="utf-8")
        )

    plot_copy_dir.mkdir(parents=True, exist_ok=True)
    copied_plot = plot_copy_dir / plot_path.name
    shutil.copy2(plot_path, copied_plot)

    payload.update(
        {
            "completed_at": time.strftime("%F %T"),
            "run_dir": str(run_dir),
            "plot_source": str(plot_path),
            "plot_copy": str(copied_plot),
            "comparison_metrics": comparison,
            "status": "complete",
        }
    )
    write_json(status_path, payload)
    update_manifest(
        method_key,
        {
            "run_dir": str(run_dir),
            "plot": str(copied_plot.relative_to(REPO_ROOT)),
            "unique_signatures": comparison.get("unique_observed_signatures"),
            "pearson_r_vs_ideal": comparison.get(
                "model_probability_vs_reward_pearson_vs_ideal",
                comparison.get("probability_vs_reward_pearson_vs_ideal"),
            ),
            "log_file": str(log_path.relative_to(REPO_ROOT)),
        },
    )
    log_line(
        log_path,
        f"complete: run_dir={run_dir} plot={copied_plot} "
        f"unique_signatures={comparison.get('unique_observed_signatures')}",
    )


if __name__ == "__main__":
    main()
