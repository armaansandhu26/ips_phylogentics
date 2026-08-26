"""Pre-flight checks and suite preparation before launching final runs."""

from __future__ import annotations

import subprocess

from final.configs import SuiteConfig
from final.paths import FINAL_ROOT, PYTHON, REPO_ROOT


def run_preflight(suite: SuiteConfig) -> None:
    """Run cheap verification checks before training."""
    print(f"[final] preflight for suite={suite.id} taxa={suite.taxa}", flush=True)

    cfg_rel = str(suite.resolve_cfg_path("grpo").relative_to(REPO_ROOT))
    dataset_rel = str(suite.dataset.relative_to(REPO_ROOT))

    if suite.taxa == 5:
        subprocess.run(
            [
                str(PYTHON),
                "-m",
                "final.verify.verify_mx",
                "--suite",
                suite.id,
                "--cfg",
                cfg_rel,
                "--dataset",
                dataset_rel,
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    subprocess.run(
        [
            str(PYTHON),
            "-m",
            "final.verify.verify_topology_hash",
            "--cfg",
            cfg_rel,
            "--dataset",
            dataset_rel,
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    print("[final] preflight OK", flush=True)
