#!/usr/bin/env python3
"""Eval Phase 1b Panel F — replay anchors + pfloor carry-forward. PhyloGFN always first."""

from __future__ import annotations

from pathlib import Path

from grpo_experiments.eval_utils import choose_device, load_json, save_json
from grpo_experiments.scripts.eval_ips_replay_ablation_phase01 import (
    AB_ROOT,
    run_panel,
)

MANIFEST_1B = AB_ROOT / "manifest_phase1b.json"
MANIFEST_01 = AB_ROOT / "manifest_phase01.json"

PANELS: dict[str, list[tuple[str, str]]] = {
    "panel_F_r32": [
        ("phylgfn", "ablation_phylgfn_r32"),
        ("hyb_grpo", "ablation_hyb_grpo_r32"),
        ("hyb_ips", "ablation_hyb_ips_r32"),
        ("hyb_ips_p002", "ablation_hyb_ips_pfloor_002_r32"),
        ("hyb_ips_p005", "ablation_hyb_ips_pfloor_005_r32"),
    ],
    "panel_F_r128": [
        ("phylgfn", "ablation_phylgfn_r128"),
        ("hyb_grpo", "ablation_hyb_grpo_r128"),
        ("hyb_ips", "ablation_hyb_ips_r128"),
        ("hyb_ips_p002", "ablation_hyb_ips_pfloor_002_r128"),
        ("hyb_ips_p005", "ablation_hyb_ips_pfloor_005_r128"),
    ],
    "panel_F_ips_pfloor_x_replay": [
        ("phylgfn_r64", "ablation_phylgfn_r64"),
        ("hyb_ips_r32", "ablation_hyb_ips_r32"),
        ("hyb_ips_p002_r32", "ablation_hyb_ips_pfloor_002_r32"),
        ("hyb_ips_p005_r32", "ablation_hyb_ips_pfloor_005_r32"),
        ("hyb_ips_r64", "ablation_hyb_ips_r64"),
        ("hyb_ips_p002_r64", "ablation_hyb_ips_pfloor_002"),
        ("hyb_ips_p005_r64", "ablation_hyb_ips_pfloor_005"),
        ("hyb_ips_r128", "ablation_hyb_ips_r128"),
        ("hyb_ips_p002_r128", "ablation_hyb_ips_pfloor_002_r128"),
        ("hyb_ips_p005_r128", "ablation_hyb_ips_pfloor_005_r128"),
    ],
}

REPLAY_TAG = {
    "panel_F_r32": "replay32",
    "panel_F_r128": "replay128",
    "panel_F_ips_pfloor_x_replay": "replay32/64/128",
}


def resolve_runs(*manifests: dict) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for manifest in manifests:
        for row in manifest["runs"]:
            out[(row["outcome"], row["id"])] = Path(row["run_dir"])
    return out


def main() -> None:
    manifest_1b = load_json(MANIFEST_1B)
    manifest_01 = load_json(MANIFEST_01)
    lookup = resolve_runs(manifest_01, manifest_1b)
    device = choose_device(None)
    outcomes = sorted({row["outcome"] for row in manifest_1b["runs"]})
    if not outcomes:
        raise SystemExit(f"No runs in {MANIFEST_1B}")

    for outcome in outcomes:
        print(f"\n=== outcome={outcome} ===")
        for panel_name, specs in PANELS.items():
            print(f"\n--- {panel_name} ---")
            missing = [run_id for _, run_id in specs if (outcome, run_id) not in lookup]
            if missing:
                print(f"  skip: missing runs {missing}")
                continue
            replay_tag = REPLAY_TAG[panel_name]
            run_panel(
                outcome,
                panel_name,
                specs,
                lookup,
                device,
                replay_label=replay_tag,
            )

    status = {
        "phase": "1b",
        "status": "complete",
        "eval_root": str(AB_ROOT / "eval"),
        "manifest": str(MANIFEST_1B),
        "panels": {
            o: {p: str(AB_ROOT / "eval" / o / p) for p in PANELS}
            for o in outcomes
        },
    }
    save_json(AB_ROOT / "eval_phase1b_status.json", status)
    print(f"\nEval complete. Status: {AB_ROOT / 'eval_phase1b_status.json'}")


if __name__ == "__main__":
    main()
