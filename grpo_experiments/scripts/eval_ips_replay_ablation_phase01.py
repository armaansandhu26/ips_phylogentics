#!/usr/bin/env python3
"""Eval Phase 0+1 ablation panels — PhyloGFN always first."""

from __future__ import annotations

import json
from pathlib import Path

from grpo_experiments.eval_utils import choose_device, load_json, save_json
from grpo_experiments.scripts.compare_sampling import (
    compute_bin_edges,
    compute_bin_frequencies,
    json_ready_summary,
    plot_sampling_comparison,
    plot_sampling_distributions,
    plot_sampling_overlay,
    plot_score_density,
    sample_run,
    save_scores_cache,
)

AB_ROOT = Path("grpo_experiments/runs/ips_replay_ablation")
MANIFEST = AB_ROOT / "manifest_phase01.json"

PANELS: dict[str, list[tuple[str, str]]] = {
    "panel_A_method_ladder": [
        ("phylgfn", "ablation_phylgfn_r64"),
        ("hyb_grpo", "ablation_hyb_grpo_r64"),
        ("hyb_ips", "ablation_hyb_ips_r64"),
    ],
    "panel_C_ips_floor": [
        ("phylgfn", "ablation_phylgfn_r64"),
        ("hyb_grpo", "ablation_hyb_grpo_r64"),
        ("hyb_ips_r64", "ablation_hyb_ips_r64"),
        ("hyb_ips_p010", "ablation_hyb_ips_pfloor_010"),
        ("hyb_ips_p005", "ablation_hyb_ips_pfloor_005"),
        ("hyb_ips_p002", "ablation_hyb_ips_pfloor_002"),
    ],
    "panel_D1_entropy_ips": [
        ("phylgfn", "ablation_phylgfn_r64"),
        ("hyb_ips_r64", "ablation_hyb_ips_r64"),
        ("hyb_ips_ent0", "ablation_hyb_ips_ent_000"),
        ("hyb_ips_ent001", "ablation_hyb_ips_ent_001"),
    ],
    "panel_D2_entropy_grpo": [
        ("phylgfn", "ablation_phylgfn_r64"),
        ("hyb_grpo_r64", "ablation_hyb_grpo_r64"),
        ("hyb_grpo_ent0", "ablation_hyb_grpo_ent_000"),
        ("hyb_grpo_ent001", "ablation_hyb_grpo_ent_001"),
    ],
}


def resolve_runs(manifest: dict) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for row in manifest["runs"]:
        out[(row["outcome"], row["id"])] = Path(row["run_dir"])
    return out


def run_panel(
    outcome: str,
    panel_name: str,
    specs: list[tuple[str, str]],
    lookup: dict[tuple[str, str], Path],
    device: str,
    samples: int = 1000,
    replay_label: str = "replay64",
) -> None:
    out_dir = AB_ROOT / "eval" / outcome / panel_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for idx, (label, run_id) in enumerate(specs):
        run_dir = lookup[(outcome, run_id)]
        print(f"  sample {outcome}/{panel_name}: {label} ({run_dir.name})")
        summaries.append(
            sample_run(
                run_dir,
                label,
                device=device,
                samples=samples,
                batch_size=128,
                seed=idx,
                checkpoint_name="final_checkpoint.pt",
                estimate_mll=False,
            )
        )
    bin_edges = compute_bin_edges(summaries, 10)
    frequencies = compute_bin_frequencies(summaries, bin_edges)
    title = f"vs PhyloGFN — {panel_name} @ {replay_label} ({outcome})"
    payload = {
        "metadata": {
            "panel": panel_name,
            "outcome": outcome,
            "baseline": "phylgfn",
            "samples_per_run": samples,
            "reward_bin_edges": [float(x) for x in bin_edges],
        },
        "runs": [
            json_ready_summary(r, bin_edges=bin_edges, bin_frequencies=frequencies[r["label"]])
            for r in summaries
        ],
    }
    save_json(out_dir / "sampling_summary.json", payload)
    save_scores_cache(summaries, out_dir / "sampling_scores.npz")
    plot_sampling_comparison(
        summaries, bin_edges, frequencies,
        out_dir / "sampling_comparison.png", 20,
        samples=samples, n_bins=10, title_context=title,
    )
    plot_sampling_distributions(
        summaries, out_dir / "sampling_distributions.png",
        samples=samples, title_context=title,
    )
    plot_sampling_overlay(
        summaries, out_dir / "sampling_distributions_overlay.png",
        title_context=title,
    )
    plot_score_density(
        summaries, out_dir / "sampling_score_density.png",
        title_context=title,
    )
    baseline = summaries[0]["log_scores"]
    bmean = float(baseline.mean())
    lines = []
    for row in summaries:
        s = row["log_scores"]
        lines.append(
            f"  {row['label']}: mean={s.mean():.2f} ({s.mean()-bmean:+.2f}) "
            f"topo={row['unique_topologies']} dup={row['topology_duplicate_fraction']:.3f}"
        )
    (out_dir / "sampling_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    manifest = load_json(MANIFEST)
    lookup = resolve_runs(manifest)
    device = choose_device(None)
    for outcome in ("sig", "topo"):
        print(f"\n=== outcome={outcome} ===")
        for panel_name, specs in PANELS.items():
            print(f"\n--- {panel_name} ---")
            run_panel(outcome, panel_name, specs, lookup, device)
    status = {
        "phase": "0+1",
        "status": "complete",
        "eval_root": str(AB_ROOT / "eval"),
        "panels": {
            o: {p: str(AB_ROOT / "eval" / o / p) for p in PANELS}
            for o in ("sig", "topo")
        },
    }
    save_json(AB_ROOT / "eval_phase01_status.json", status)
    print(f"\nEval complete. Status: {AB_ROOT / 'eval_phase01_status.json'}")


if __name__ == "__main__":
    main()
