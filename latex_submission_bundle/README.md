# Self-contained LaTeX paper bundle

This directory contains the complete manuscript source, bibliography, generated
tables, and every figure referenced by `main.tex`. It has no symlinks and does
not depend on paths outside this directory.

## Build

From this directory, run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The compiled manuscript is `main.pdf`. Clean temporary LaTeX files with:

```bash
latexmk -c
```

## Layout

- `main.tex`: manuscript and appendix.
- `references.bib` and `main.bbl`: bibliography source and prebuilt BibTeX output.
- `results_macros.tex`: shared numerical macros.
- `tables/`: all table fragments included by the manuscript.
- `figures/main/` and `figures/appendix/`: all referenced figure assets.
- `data/`: machine-readable values behind generated tables and figures.
- `scripts/`: figure-generation scripts retained for provenance.
- `provenance/ASSET_MANIFEST.md`: source map for bundled assets.

The molecular-synthesis result has been updated to the final three-seed,
2,500-update evaluation. Its figure uses arithmetic-mean bars, sample-standard-
deviation whiskers, and circles for all seeds; it contains no best-run or star
selection.

The main phylogenetics figure is the matched five-taxon comparison of GRPO,
IPS-GRPO, MIPS-GRPO, and PhyloGFN. The 27-taxon table remains an explicitly
asymmetric scaling diagnostic because complete GRPO, IPS-GRPO, and matched
MIPS-GRPO runs are not yet available at that scale.
