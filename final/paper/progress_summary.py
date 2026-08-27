#!/usr/bin/env python3
"""Print or patch overall experiment progress for EXPERIMENT_TODO.md."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TODO = Path(__file__).resolve().parent / "EXPERIMENT_TODO.md"
MARKER_START = "<!-- PROGRESS_START -->"
MARKER_END = "<!-- PROGRESS_END -->"


def latest_epoch_metrics(base: Path) -> int | None:
    if not base.exists():
        return None
    for d in sorted([x for x in base.iterdir() if x.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        mf = d / "metrics.jsonl"
        if mf.exists():
            return int(json.loads(mf.read_text().splitlines()[-1])["epoch"])
    return None


def phylo_epoch(seed: int) -> int | None:
    ckpts = list((REPO / f"final/runs/27taxa_noreplay_b4096_seed{seed}/phylgfn").rglob("checkpoint_*.pt"))
    if not ckpts:
        return None
    return max(int(re.search(r"(\d+)", p.name).group(1)) for p in ckpts)


def run_progress(train_frac: float, eval_done: bool = False, plots_done: bool = False) -> float:
    return 0.5 * train_frac + (0.25 if eval_done else 0.0) + (0.25 if plots_done else 0.0)


def bar(pct: float, width: int = 20) -> str:
    filled = round(width * pct / 100)
    return "█" * filled + "░" * (width - filled)


def compute() -> dict:
    epochs = {
        "27t_lrips_s0": (latest_epoch_metrics(REPO / "final/runs/27taxa_noreplay_b4096_seed0/learned_reverse"), 25000),
        "27t_lrips_s1": (latest_epoch_metrics(REPO / "final/runs/27taxa_noreplay_b4096_seed1/learned_reverse"), 25000),
        "27t_lrips_s2": (latest_epoch_metrics(REPO / "final/runs/27taxa_noreplay_b4096_seed2/learned_reverse"), 25000),
        "27t_phylo_s0": (phylo_epoch(0), 32000),
        "27t_phylo_s1": (phylo_epoch(1), 32000),
        "27t_phylo_s2": (phylo_epoch(2), 32000),
        "hg64_lrips_s1": (latest_epoch_metrics(REPO / "final/runs/hypergrid_64_seed1/learned_reverse_ips"), 10000),
        "hg64_tb_s1": (latest_epoch_metrics(REPO / "final/runs/hypergrid_64_seed1/trajectory_balance"), 10000),
        "hg64_grpo_s2": (latest_epoch_metrics(REPO / "final/runs/hypergrid_64_seed2/grpo"), 10000),
        "hg64_cips_s2": (latest_epoch_metrics(REPO / "final/runs/hypergrid_64_seed2/count_ips"), 10000),
        "hg64_lrips_s2": (latest_epoch_metrics(REPO / "final/runs/hypergrid_64_seed2/learned_reverse_ips"), 10000),
    }

    hg64_cells = [
        run_progress(1.0, True, True),   # grpo s0
        run_progress(1.0, True, True),   # grpo s1
        run_progress(epochs["hg64_grpo_s2"][0] / 10000 if epochs["hg64_grpo_s2"][0] else 0.0),  # grpo s2
        run_progress(1.0, True, True),   # cips s0
        run_progress(1.0, True, True),   # cips s1
        run_progress(epochs["hg64_cips_s2"][0] / 10000 if epochs["hg64_cips_s2"][0] else 0.0),  # cips s2
        run_progress(1.0, True, True),   # lrips s0
        run_progress(1.0, True, True),   # lrips s1
        run_progress(epochs["hg64_lrips_s2"][0] / 10000 if epochs["hg64_lrips_s2"][0] else 0.0),  # lrips s2
        run_progress(1.0, True, True),   # tb s0
        run_progress(1.0, True, True),   # tb s1
        run_progress(0.0),               # tb s2
    ]
    hg64_pct = 100 * sum(hg64_cells) / len(hg64_cells)

    hg8_pct = 100.0

    t27_fracs = [
        (epochs["27t_lrips_s0"][0] or 0) / 25000,
        (epochs["27t_lrips_s1"][0] or 0) / 25000,
        (epochs["27t_lrips_s2"][0] or 0) / 25000,
        (epochs["27t_phylo_s0"][0] or 0) / 32000,
        (epochs["27t_phylo_s1"][0] or 0) / 32000,
        (epochs["27t_phylo_s2"][0] or 0) / 32000,
    ]
    # PhyloGFN s0/s1: train + 1M eval + plots complete
    t27_fracs[3] = 1.0
    t27_fracs[4] = 1.0
    t27_pct = 100 * sum(run_progress(f) for f in t27_fracs) / 6

    analysis_done = 8
    analysis_total = 12
    analysis_pct = 100 * analysis_done / analysis_total

    # Primary path: hg64 (12) + hg8 (9) + 27t (6) + analysis (12)
    primary = (
        (sum(hg64_cells) / 12) * 12
        + 1.0 * 9
        + (sum(run_progress(f) for f in t27_fracs) / 6) * 6
        + (analysis_done / analysis_total) * 12
    ) / 39 * 100

    extras_done = 2 + 4 + 1  # hg4096 + 5t + 10t ablation
    extras_total = 2 + 4 + 1 + 4  # + full 10t suite (optional)
    extras_pct = 100 * extras_done / extras_total

    train_avg = 100 * sum(e / t for e, t in epochs.values() if e is not None) / len(epochs)

    return {
        "epochs": epochs,
        "primary_pct": primary,
        "hg64_pct": hg64_pct,
        "hg8_pct": hg8_pct,
        "t27_pct": t27_pct,
        "analysis_pct": analysis_pct,
        "extras_pct": extras_pct,
        "train_avg": train_avg,
    }


def render_block(data: dict) -> str:
    e = data["epochs"]
    primary = data["primary_pct"]

    lines = [
        MARKER_START,
        "",
        "## Overall progress",
        "",
        f"**Primary path (paper-critical): {primary:.0f}%** `{bar(primary)}`",
        "",
        "*Method: each run cell = 50% train + 25% eval + 25% plots. Active runs credit train fraction only. Refresh: `.venv/bin/python final/paper/progress_summary.py --patch`*",
        "",
        "| Track | Weight | Progress | Bar |",
        "|-------|-------:|---------:|:---:|",
        f"| Hyper-Grid 64 (4×3 seeds) | 12 | **{data['hg64_pct']:.0f}%** | `{bar(data['hg64_pct'])}` |",
        f"| Hyper-Grid 8 b256 | 9 | **{data['hg8_pct']:.0f}%** | `{bar(data['hg8_pct'])}` |",
        f"| Phylo 27t (LR-IPS + PhyloGFN ×3) | 6 | **{data['t27_pct']:.0f}%** | `{bar(data['t27_pct'])}` |",
        f"| Analysis / paper deliverables | 12 | **{data['analysis_pct']:.0f}%** | `{bar(data['analysis_pct'])}` |",
        f"| **Primary total** | **39** | **{primary:.0f}%** | `{bar(primary)}` |",
        "",
        "### Active training (epoch % only)",
        "",
        "| Run | Epoch progress |",
        "|-----|---------------:|",
        f"| 27t LR-IPS s0 | {e['27t_lrips_s0'][0] or 0:,} / {e['27t_lrips_s0'][1]:,} (**{100*(e['27t_lrips_s0'][0] or 0)/e['27t_lrips_s0'][1]:.1f}%**) |",
        f"| 27t LR-IPS s1 | {e['27t_lrips_s1'][0] or 0:,} / {e['27t_lrips_s1'][1]:,} (**{100*(e['27t_lrips_s1'][0] or 0)/e['27t_lrips_s1'][1]:.1f}%**) |",
        f"| 27t LR-IPS s2 | {e['27t_lrips_s2'][0] or 0:,} / {e['27t_lrips_s2'][1]:,} (**{100*(e['27t_lrips_s2'][0] or 0)/e['27t_lrips_s2'][1]:.1f}%**) |",
        f"| 27t PhyloGFN s0 | ✅ train + 1M eval + plots |",
        f"| 27t PhyloGFN s1 | ✅ train + 1M eval + plots |",
        f"| 27t PhyloGFN s2 | {e['27t_phylo_s2'][0] or 0:,} / {e['27t_phylo_s2'][1]:,} (**{100*(e['27t_phylo_s2'][0] or 0)/e['27t_phylo_s2'][1]:.1f}%**) |",
        f"| hg64 GRPO s2 | {e['hg64_grpo_s2'][0] or 0:,} / {e['hg64_grpo_s2'][1]:,} (**{100*(e['hg64_grpo_s2'][0] or 0)/e['hg64_grpo_s2'][1]:.1f}%**) |",
        f"| hg64 Count-IPS s2 | {e['hg64_cips_s2'][0] or 0:,} / {e['hg64_cips_s2'][1]:,} (**{100*(e['hg64_cips_s2'][0] or 0)/e['hg64_cips_s2'][1]:.1f}%**) |",
        f"| hg64 MIPS-GRPO s2 | {e['hg64_lrips_s2'][0] or 0:,} / {e['hg64_lrips_s2'][1]:,} (**{100*(e['hg64_lrips_s2'][0] or 0)/e['hg64_lrips_s2'][1]:.1f}%**) |",
        f"| *Avg across active jobs* | **{data['train_avg']:.1f}%** |",
        "",
        f"**Extras** (5t, 10t ablation, hg4096 — not in primary): **{data['extras_pct']:.0f}%** `{bar(data['extras_pct'])}`",
        "",
        MARKER_END,
    ]
    return "\n".join(lines) + "\n"


def patch_todo(block: str) -> None:
    text = TODO.read_text()
    pattern = re.compile(re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END) + r"\n?", re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"Markers not found in {TODO}; add {MARKER_START} / {MARKER_END} first.")
    TODO.write_text(pattern.sub(block, text, count=1))


def main() -> None:
    data = compute()
    block = render_block(data)
    if "--patch" in sys.argv:
        patch_todo(block)
        print(f"Patched {TODO} — primary progress {data['primary_pct']:.0f}%")
    else:
        print(block)


if __name__ == "__main__":
    main()
