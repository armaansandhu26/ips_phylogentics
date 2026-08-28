# MIPS-GRPO paper bundle

This is the clean, self-contained paper workspace. It contains no symlinks and
does not depend on the original repository layout.

## Build

```bash
MPLCONFIGDIR=/tmp/mips-paper-mpl python3 scripts/make_main_figure.py
MPLCONFIGDIR=/tmp/mips-paper-mpl python3 scripts/make_phylo_application_figure.py
latexmk -pdf main.tex
```

The single generated manuscript is `main.pdf`. It is intentionally ignored by
Git; all source needed to rebuild it is versioned. All supporting results are
included as appendices in the same document. To remove LaTeX build artifacts,
run `latexmk -c`.

## Paper story and evidence boundary

- **Act I — controlled proof:** the derivation of multiplicity-corrected IPS
  and both hypergrid regimes. H=8 is a three-seed result; H=64 is a labelled
  single-seed stress test.
- **Act II — phylogenetics:** the 27-taxon result is the main-paper scale
  demonstration. Its recoverable MIPS-GRPO artifact shows 999,987 distinct
  signatures per million draws and pathwise \(r=0.977\). The old GFlowNet
  checkpoint is unavailable, so the displayed contrast is explicitly
  preliminary and must be replaced by the running matched three-seed suite.
- **Act III — molecular synthesis:** the matched 500-update four-method result
  is in the main paper and explicitly labelled interim and single-seed.
- **Appendix:** the five-taxon exact-enumeration control, ten-taxon scaling
  plots, per-seed values, training curves, reverse-policy ablation, topology
  diagnostics, and molecular sampling convergence.
- **Local figure archive:** recoverable source plots are retained locally in
  `figures/archive/` even when not referenced from LaTeX. The archive is
  intentionally ignored by Git to keep the paper source bundle small.
- **Not final claims:** the current 27-taxon between-method comparison remains
  in the main paper because it is the correct application scale, but its
  missing legacy baseline checkpoint is disclosed next to the result. The sEH
  result is seed 0 at 500 updates and must be replaced after the 2,000-update
  matched evaluation.

The exact numbers behind the tables and generated figures are in `data/`; the
figure scripts read those CSV files directly. `provenance/ASSET_MANIFEST.md`
records where every copied result originated.
