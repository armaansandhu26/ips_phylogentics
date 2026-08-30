#!/usr/bin/env python3
"""Generate the main hypergrid regime summary from its committed CSV data."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)

colors = {
    "GRPO": "#D55E00",
    "Count IPS-GRPO": "#0072B2",
    "MIPS-GRPO": "#009E73",
    "GFlowNet TB": "#CC79A7",
}
methods = ["GRPO", "Count IPS-GRPO", "MIPS-GRPO", "GFlowNet TB"]
csv_methods = {
    "grpo": "GRPO",
    "count_ips_grpo": "Count IPS-GRPO",
    "mips_grpo": "MIPS-GRPO",
    "gflownet_tb": "GFlowNet TB",
}

with (ROOT / "data" / "hypergrid.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))

small_rows = [row for row in rows if row["regime"] == "counting_works"]
large_rows = [row for row in rows if row["regime"] == "counting_fails"]


def values_for(source_rows, method, field):
    csv_method = next(key for key, label in csv_methods.items() if label == method)
    return [float(row[field]) for row in source_rows if row["method"] == csv_method]


small_l1 = []
small_sd = []
small_modes = []
large_l1 = []
large_modes = []
for method in methods:
    l1_values = values_for(small_rows, method, "l1")
    mode_values = values_for(small_rows, method, "modes")
    small_l1.append(float(np.mean(l1_values)) if l1_values else np.nan)
    small_sd.append(float(np.std(l1_values, ddof=1)) if len(l1_values) > 1 else np.nan)
    small_modes.append(float(np.mean(mode_values)) if mode_values else np.nan)

    l1_values = values_for(large_rows, method, "l1")
    mode_values = values_for(large_rows, method, "modes")
    large_l1.append(float(np.mean(l1_values)))
    large_modes.append(float(np.mean(mode_values)))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)
x = np.arange(len(methods))
width = 0.36

for ax, small, large, ylabel, ylim in [
    (axes[0], small_l1, large_l1, r"$\ell_1$ distance to target", (0, 2.15)),
    (axes[1], small_modes, large_modes, "Modes recovered (out of 4)", (0, 4.55)),
]:
    valid = ~np.isnan(np.asarray(small, dtype=float))
    ax.bar(x[valid] - width / 2, np.asarray(small)[valid], width,
           color=[colors[m] for m, keep in zip(methods, valid) if keep],
           alpha=0.62, edgecolor="black", linewidth=0.5,
           label=r"Counts informative: $H=8,G=256$ (3 seeds)")
    ax.bar(x + width / 2, large, width, color=[colors[m] for m in methods],
           alpha=1.0, edgecolor="black", linewidth=0.5,
           label=r"Counts sparse: $H=64,G=32$ (1 seed)")
    ax.set_xticks(x, methods, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.22)

axes[0].errorbar(x[:3] - width / 2, small_l1[:3], yerr=small_sd[:3], fmt="none",
                 ecolor="black", capsize=3, linewidth=1)
axes[0].set_title("Distributional fidelity")
axes[1].set_title("Mode coverage")
axes[1].set_yticks([0, 1, 2, 3, 4])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2,
           frameon=False)

fig.savefig(OUT / "figure1_hypergrid_regimes.pdf", bbox_inches="tight")
plt.close(fig)
