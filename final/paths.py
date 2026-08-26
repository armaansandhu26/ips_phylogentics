"""Shared paths for the final paper experiment harness."""

from __future__ import annotations

from pathlib import Path

FINAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FINAL_ROOT.parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DATASET_ROOT = REPO_ROOT / "dataset" / "benchmark_datasets"

CONFIGS_DIR = FINAL_ROOT / "configs" / "suites"
RUNS_DIR = FINAL_ROOT / "runs"
RESULTS_DIR = FINAL_ROOT / "results"

GRPO_SCRIPTS = REPO_ROOT / "grpo_experiments" / "scripts"
OG_CODE_ROOT = REPO_ROOT / "og_code"
PAPER_ROOT = REPO_ROOT / "phylogfn_paper"

METHODS = ("grpo", "count_ips", "learned_reverse", "phylgfn")
