#!/usr/bin/env python3
"""Eval Panel H — delayed IPS vs early IPS and pure GRPO."""

from __future__ import annotations

from pathlib import Path

from grpo_experiments.eval_utils import choose_device, load_json, save_json
from grpo_experiments.scripts.eval_ips_replay_ablation_phase01 import (
    AB_ROOT,
    run_panel,
)

MANIFEST_01 = AB_ROOT / "manifest_phase01.json"
MANIFEST_H = AB_ROOT / "manifest_panel_h.json"

PANELS: dict[str, list[tuple[str, str]]] = {
    "panel_H_delayed_ips": [
        ("phylgfn", "ablation_phylgfn_r64"),
        ("hyb_grpo", "ablation_hyb_grpo_r64"),
        ("hyb_ips", "ablation_hyb_ips_r64"),
        ("hyb_ips_d500", "ablation_hyb_ips_delayed500"),
        ("hyb_ips_d750", "ablation_hyb_ips_delayed750"),
    ],
}


def resolve_runs(*manifests: dict) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for manifest in manifests:
        for row in manifest["runs"]:
            out[(row["outcome"], row["id"])] = Path(row["run_dir"])
    return out


def main() -> None:
    lookup = resolve_runs(load_json(MANIFEST_01), load_json(MANIFEST_H))
    device = choose_device(None)
    manifest_h = load_json(MANIFEST_H)
    if not manifest_h.get("runs"):
        raise SystemExit(f"No runs in {MANIFEST_H}")

    outcome = manifest_h["runs"][0]["outcome"]
    print(f"\n=== outcome={outcome} ===")
    for panel_name, specs in PANELS.items():
        print(f"\n--- {panel_name} ---")
        missing = [run_id for _, run_id in specs if (outcome, run_id) not in lookup]
        if missing:
            raise SystemExit(f"Missing runs for {panel_name}: {missing}")
        run_panel(outcome, panel_name, specs, lookup, device, replay_label="replay64")

    status = {
        "phase": "panel_h",
        "status": "complete",
        "eval_root": str(AB_ROOT / "eval" / outcome / "panel_H_delayed_ips"),
        "manifest": str(MANIFEST_H),
    }
    save_json(AB_ROOT / "eval_panel_h_status.json", status)
    print(f"\nEval complete. Status: {AB_ROOT / 'eval_panel_h_status.json'}")


if __name__ == "__main__":
    main()
