#!/usr/bin/env python3
"""Eval Panel G — replay annealing vs fixed replay endpoints."""

from __future__ import annotations

from pathlib import Path

from grpo_experiments.eval_utils import choose_device, load_json, save_json
from grpo_experiments.scripts.eval_ips_replay_ablation_phase01 import (
    AB_ROOT,
    run_panel,
)

MANIFEST_01 = AB_ROOT / "manifest_phase01.json"
MANIFEST_1B = AB_ROOT / "manifest_phase1b.json"
MANIFEST_G = AB_ROOT / "manifest_panel_g.json"

PANELS: dict[str, list[tuple[str, str]]] = {
    "panel_G_replay_anneal": [
        ("phylgfn_r64", "ablation_phylgfn_r64"),
        ("hyb_grpo_r32", "ablation_hyb_grpo_r32"),
        ("hyb_grpo_r128", "ablation_hyb_grpo_r128"),
        ("hyb_ips_r32", "ablation_hyb_ips_r32"),
        ("hyb_ips_r128", "ablation_hyb_ips_r128"),
        ("hyb_grpo_anneal", "ablation_hyb_grpo_replay_anneal_128to32"),
        ("hyb_ips_anneal", "ablation_hyb_ips_replay_anneal_128to32"),
    ],
}


def resolve_runs(*manifests: dict) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for manifest in manifests:
        for row in manifest["runs"]:
            out[(row["outcome"], row["id"])] = Path(row["run_dir"])
    return out


def main() -> None:
    lookup = resolve_runs(
        load_json(MANIFEST_01),
        load_json(MANIFEST_1B),
        load_json(MANIFEST_G),
    )
    device = choose_device(None)
    manifest_g = load_json(MANIFEST_G)
    if not manifest_g.get("runs"):
        raise SystemExit(f"No runs in {MANIFEST_G}")

    outcome = manifest_g["runs"][0]["outcome"]
    print(f"\n=== outcome={outcome} ===")
    for panel_name, specs in PANELS.items():
        print(f"\n--- {panel_name} ---")
        missing = [run_id for _, run_id in specs if (outcome, run_id) not in lookup]
        if missing:
            raise SystemExit(f"Missing runs for {panel_name}: {missing}")
        run_panel(outcome, panel_name, specs, lookup, device, replay_label="anneal128to32")

    status = {
        "phase": "panel_g",
        "status": "complete",
        "eval_root": str(AB_ROOT / "eval" / outcome / "panel_G_replay_anneal"),
        "manifest": str(MANIFEST_G),
    }
    save_json(AB_ROOT / "eval_panel_g_status.json", status)
    print(f"\nEval complete. Status: {AB_ROOT / 'eval_panel_g_status.json'}")


if __name__ == "__main__":
    main()
