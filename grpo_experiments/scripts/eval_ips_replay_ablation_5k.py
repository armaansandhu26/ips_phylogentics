#!/usr/bin/env python3
"""Eval 5k validation — hyb_ips pfloor winners vs PhyloGFN @ replay64."""

from __future__ import annotations

from pathlib import Path

from grpo_experiments.eval_utils import choose_device, load_json, save_json
from grpo_experiments.scripts.eval_ips_replay_ablation_phase01 import (
    AB_ROOT,
    run_panel,
)

MANIFEST_5K = AB_ROOT / "manifest_5k.json"

PANELS: dict[str, list[tuple[str, str]]] = {
    "panel_5k_ips_winners": [
        ("phylgfn", "ablation_phylgfn_r64_5k"),
        ("hyb_ips_p002", "ablation_hyb_ips_pfloor_002_5k"),
        ("hyb_ips_p005", "ablation_hyb_ips_pfloor_005_5k"),
    ],
}


def resolve_runs(manifest: dict) -> dict[tuple[str, str], Path]:
    return {
        (row["outcome"], row["id"]): Path(row["run_dir"])
        for row in manifest["runs"]
    }


def main() -> None:
    manifest = load_json(MANIFEST_5K)
    if not manifest.get("runs"):
        raise SystemExit(f"No runs in {MANIFEST_5K}")

    lookup = resolve_runs(manifest)
    device = choose_device(None)
    outcome = manifest["runs"][0]["outcome"]

    print(f"\n=== outcome={outcome} ===")
    for panel_name, specs in PANELS.items():
        print(f"\n--- {panel_name} ---")
        missing = [run_id for _, run_id in specs if (outcome, run_id) not in lookup]
        if missing:
            raise SystemExit(f"Missing runs for {panel_name}: {missing}")
        run_panel(outcome, panel_name, specs, lookup, device, replay_label="replay64@5k")

    status = {
        "phase": "5k",
        "status": "complete",
        "eval_root": str(AB_ROOT / "eval" / outcome / "panel_5k_ips_winners"),
        "manifest": str(MANIFEST_5K),
    }
    save_json(AB_ROOT / "eval_5k_status.json", status)
    print(f"\nEval complete. Status: {AB_ROOT / 'eval_5k_status.json'}")


if __name__ == "__main__":
    main()
