"""Run all four methods for one comparison suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from final.configs import load_suite, list_suites
from final.logging.wandb_logger import WandbSettings
from final.methods.base import write_json
from final.paths import METHODS, RESULTS_DIR, RUNS_DIR
from final.pipeline import _wandb_settings_from_args, run_pipeline
from final.preflight import run_preflight


def update_suite_manifest(suite_id: str, method: str, method_manifest: dict) -> Path:
    suite_dir = RUNS_DIR / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = suite_dir / "suite.json"
    if manifest_path.exists():
        suite_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        suite_manifest = {"suite_id": suite_id, "methods": {}}
    suite_manifest["methods"][method] = method_manifest
    write_json(manifest_path, suite_manifest)
    return manifest_path


def run_suite(
    suite_id: str,
    *,
    methods: list[str] | None = None,
    cuda_device: int = 0,
    device: str = "cuda:0",
    skip_train: bool = False,
    skip_sample: bool = False,
    skip_plots: bool = False,
    skip_preflight: bool = False,
    wandb_settings: WandbSettings | None = None,
) -> dict:
    suite = load_suite(suite_id)
    if not skip_preflight and not skip_train:
        run_preflight(suite)
    selected = methods or list(METHODS)
    suite_manifest: dict = {"suite_id": suite.id, "methods": {}}

    for method in selected:
        print(f"\n[final] === suite={suite.id} method={method} ===\n", flush=True)
        log_file = RUNS_DIR / suite.id / method / "pipeline.log"
        method_manifest = run_pipeline(
            suite,
            method,
            device=device,
            cuda_device=cuda_device,
            skip_train=skip_train,
            skip_sample=skip_sample,
            skip_plots=skip_plots,
            log_file=log_file,
            wandb_settings=wandb_settings,
        )
        suite_manifest["methods"][method] = method_manifest
        update_suite_manifest(suite.id, method, method_manifest)

    results_dir = RESULTS_DIR / suite.id
    results_dir.mkdir(parents=True, exist_ok=True)
    from final.aggregate import aggregate_suite

    table = aggregate_suite(suite.id)
    suite_manifest["results"] = table
    write_json(RUNS_DIR / suite.id / "suite.json", suite_manifest)
    return suite_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all four paper methods (GRPO, count IPS, learned-reverse, PhyloGFN) for one suite."
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Suite id or path (e.g. 5taxa_noreplay, 27taxa_replay).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=None,
        help="Subset of methods to run (default: all four).",
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List available suite configs and exit.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip m(x) and topology hash verification (not recommended).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training metrics and plots to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="phylogfn-final")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_suites:
        for path in list_suites():
            suite = load_suite(path)
            print(f"{suite.id:20s}  taxa={suite.taxa}  shift={suite.log_score_shift}")
        return
    manifest = run_suite(
        args.suite,
        methods=args.methods,
        cuda_device=args.cuda_device,
        device=args.device,
        skip_train=args.skip_train,
        skip_sample=args.skip_sample,
        skip_plots=args.skip_plots,
        skip_preflight=args.skip_preflight,
        wandb_settings=_wandb_settings_from_args(args),
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
