# Paper artifacts (P0: matched-transform comparisons)

Generated outputs for Issue 1 fix — same transform for both methods.

## Regenerate

```bash
cd /home/armaan/phylogfn

# Table 2 (CSV, Markdown, LaTeX, JSON)
.venv/bin/python grpo_experiments/scripts/build_paper_table2.py

# Figure 6 — 27 taxa (headline)
.venv/bin/python grpo_experiments/scripts/plot_matched_transform_figure.py --taxa 27

# Optional: 10 taxa supplementary figure
.venv/bin/python grpo_experiments/scripts/plot_matched_transform_figure.py \
  --taxa 10 \
  --output final/paper/figure6_10taxa_matched_transform.png
```

## Outputs

| File | Use in paper |
|------|----------------|
| `table2.tex` | Drop into LaTeX source as Table 2 |
| `table2.md` | Quick reference + caption draft |
| `table2.csv` | Spreadsheet / sanity check |
| `figure6_27taxa_matched_transform.png` | Replace old Figure 6 |
| `figure6_10taxa_matched_transform.png` | Supplementary (optional) |

## Data sources

Configured in `manifest.json`. Metrics are read from each method's
`comparison_metrics.json`; both linear and log-log Pearson keys are normalized
automatically.

## Figure layout

```
              Linear P vs R          Log-log log P vs log R
MIPS-GRPO     regenerated          regenerated
GFlowNet      prerendered PNG      prerendered PNG
```

MIPS-GRPO panels are regenerated from NPZ with **shared column axes**.
GFlowNet uses archived PNGs when sample NPZ / checkpoints are unavailable
(27t checkpoint missing on disk). Pearson values in legends match Table 2.

## Text to paste into draft (Results)

> We report Pearson correlation between pathwise implied terminal probability
> and terminal reward under **matched transforms** for both methods: linear
> \(P(x)\) vs \(R(x)\) and log-log \(\log P(x)\) vs \(\log R(x)\) (Table 2).
> At 27 taxa, MIPS-GRPO achieves \(r = 0.977\) on both scales, while GFlowNet
> remains near zero (\(r = 0.002\) linear, \(r = 0.024\) log-log). The prior
> figure mixed scales across methods; the matched comparison does not change
> the conclusion.

## Status

- [x] P0 Table 2 with four 27t numbers
- [x] P0 Figure 6 matched-transform layout
- [ ] 5t GFlowNet log-log \(r\) (NPZ not archived; linear only in table)
