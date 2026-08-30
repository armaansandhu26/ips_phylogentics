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
- **Act II — phylogenetics:** the main-paper figure is the matched five-taxon
  comparison of GRPO, IPS-GRPO, MIPS-GRPO, and PhyloGFN. The 27-taxon table is
  an explicitly asymmetric scaling diagnostic because the four-method suite is
  not complete at that scale.
- **Act III — molecular synthesis:** the main paper reports the matched
  three-seed, 2,500-update four-method result.
- **Appendix:** ten-taxon scaling plots, per-seed values, training curves,
  reverse-policy ablation, topology diagnostics, and legacy scatter plots.
- **Local figure archive:** recoverable source plots are retained locally in
  `figures/archive/` even when not referenced from LaTeX. The archive is
  intentionally ignored by Git to keep the paper source bundle small.
- **Not final claims:** the 27-taxon MIPS-GRPO row is a legacy artifact and the
  PhyloGFN row aggregates three new seeds; they are not a matched comparison.
  Molecular mode counts vary substantially across the three seeds, so their
  difference is reported descriptively.

The exact numbers behind the tables and generated figures are in `data/`; the
figure scripts read those CSV files directly. `provenance/ASSET_MANIFEST.md`
records where every copied result originated.
