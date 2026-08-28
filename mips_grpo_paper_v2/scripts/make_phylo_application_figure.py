#!/usr/bin/env python3
"""Generate the matched no-replay five-taxa summary from committed CSV data."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "appendix"
methods = ["GRPO", "Count IPS-GRPO", "GFlowNet TB", "MIPS-GRPO"]
csv_methods = {
    "GRPO": "grpo",
    "Count IPS-GRPO": "count_ips_grpo",
    "GFlowNet TB": "gflownet_tb",
    "MIPS-GRPO": "mips_grpo",
}

with (ROOT / "data" / "phylo_5taxa_no_replay.csv").open(newline="") as handle:
    rows = {row["method"]: row for row in csv.DictReader(handle)}

ordered_rows = [rows[csv_methods[method]] for method in methods]
topologies = np.array([float(row["topologies"]) for row in ordered_rows])
unique_pct = 100 * np.array([float(row["unique_fraction"]) for row in ordered_rows])
log_r = np.array([
    float(row["loglog_r"]) if row["loglog_r"] else np.nan
    for row in ordered_rows
])
colors = ["#D55E00", "#0072B2", "#CC79A7", "#009E73"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
x = np.arange(len(methods))

axes[0].bar(x, topologies, color=colors, edgecolor="black", linewidth=0.5)
axes[0].axhline(105, color="0.35", ls="--", lw=1)
axes[0].set_ylabel("Topologies covered (out of 105)")
axes[0].set_title("Exact topology support")

axes[1].bar(x, unique_pct, color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_ylabel("Distinct signatures (%)")
axes[1].set_title("Non-repetition in 1M draws")

valid = ~np.isnan(log_r)
axes[2].bar(x[valid], log_r[valid], color=np.asarray(colors)[valid],
            edgecolor="black", linewidth=0.5)
axes[2].set_ylim(0, 1.05)
axes[2].set_ylabel(r"Pearson $r$: $\log P(x)$ vs $\log R(x)$")
axes[2].set_title("Pathwise reward alignment")
axes[2].text(x[0], 0.03, "collapsed\n(undefined)", ha="center", va="bottom", fontsize=8)

for ax, values in zip(axes, [topologies, unique_pct, log_r]):
    ax.set_xticks(x, methods, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.2)
    for idx, val in enumerate(values):
        if np.isfinite(val):
            label = f"{val:.0f}" if ax is axes[0] else (f"{val:.1f}%" if ax is axes[1] else f"{val:.3f}")
            ax.text(idx, val + (2 if ax is axes[0] else 1.5 if ax is axes[1] else 0.02),
                    label, ha="center", va="bottom", fontsize=8)

fig.suptitle("Five-taxa phylogenetic sampling — matched no-replay evaluation", y=1.04)
fig.savefig(OUT / "figureS3_phylo_5taxa_no_replay.pdf", bbox_inches="tight")
plt.close(fig)
