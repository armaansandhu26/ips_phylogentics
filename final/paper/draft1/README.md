# Paper draft bundle (`draft1/`)

Self-contained LaTeX draft with figures and table fragments wired to completed
experiment outputs (Aug 2026).

## Compile

```bash
cd /home/armaan/phylogfn/final/paper/draft1
pdflatex draft1.tex
pdflatex draft1.tex   # resolve references
```

Requires a standard TeX install (`article`, `booktabs`, `graphicx`, `subcaption`, `hyperref`).

## Layout

```
draft1/
├── draft1.tex              # Main source (updated from ../draft1.tex)
├── figures/
│   ├── hypergrid/          # H=64 suite plots @ epoch 24999 (symlinks)
│   ├── hypergrid_b256/     # H=8, G=256 suite plots @ 2k epochs, seed 0
│   └── phylo/              # Matched-transform Figure 6 + 5t panels + training curves
└── tables/
    ├── tab_hypergrid64.tex       # Table: hyper-grid H=64 recovery (25k epochs)
    ├── tab_hypergrid8_b256.tex   # Table: H=8 G=256 mean±std + per-seed (3 seeds)
    ├── tab_phylo_setup.tex         # Benchmark settings
    ├── tab_phylo_matched.tex       # Headline matched-transform Pearson r
    ├── tab_phylo_extended.tex      # 5t extended results (final/runs)
    ├── tab_uniform_ablation.tex  # 10t uniform P_B ablation
    └── tab_reverse_training.tex    # Reverse-policy diagnostics
```

## Data sources

| Artifact | Source run / script |
|----------|---------------------|
| Hyper-grid table + figs | `final/runs/hypergrid_64/plots/epoch_24999/` |
| Hyper-grid $H=8$, $G=256$ | `final/runs/hypergrid_8_b256/plots/seed{0,1,2}/` |
| Matched-transform table | `final/paper/table2.json` (manifest-backed) |
| Figure 6 (10t, 27t) | `final/paper/figure6_*_matched_transform.png` |
| 5t phylo panels | `final/runs/5taxa_noreplay/{grpo,count_ips,learned_reverse,phylgfn}/` |
| Uniform ablation | `final/runs/10taxa_uniform_reverse_ablation/` |
| Training diagnostics fig | `grpo_experiments/comparisons/learned_reverse_training_curves_5_10_27.png` |

## Regenerate upstream artifacts

```bash
cd /home/armaan/phylogfn
.venv/bin/python grpo_experiments/scripts/build_paper_table2.py
.venv/bin/python grpo_experiments/scripts/plot_matched_transform_figure.py --taxa 27
.venv/bin/python grpo_experiments/scripts/plot_matched_transform_figure.py --taxa 10 \
  --output final/paper/figure6_10taxa_matched_transform.png
# Then recopy figure6 PNGs into draft1/figures/phylo/
```

Hyper-grid suite plots:

```bash
.venv/bin/python -m final.toy.plot_comparison --suite hypergrid_64 --all-methods --last-common-checkpoint
```

## Notes

- Figure symlinks under `figures/hypergrid/` and most `figures/phylo/` point at
  run directories; keep repo paths intact when compiling.
- 27t GFlowNet headline numbers use legacy comparison runs (checkpoint missing
  on disk for og_code baseline); see `final/paper/manifest.json`.
- Root `../draft1.tex` is the pre-bundle source; **`draft1/draft1.tex`** is the
  maintained version with `\input{tables/...}` and `figures/` paths.
