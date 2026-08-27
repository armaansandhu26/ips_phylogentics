# Interim paper package: seed 0 at 500 updates

This directory contains the compact, defensible presentation of the
full-space sEH seed-0 checkpoint after 500 policy updates (50,000 training
oracle calls). Every method is evaluated with 5,000 independent final-policy
samples.

## Main paper material

- `figure1_seed0_500_updates.{png,pdf}`: the complete interim result. Panels
  report sampled non-repetition, top-molecule mass, mean proxy, and diverse
  leader modes above proxy 7.
- `table1_seed0_500_updates.{md,csv}`: exact values behind the figure, with
  validity and raw high-proxy candidate counts.
- `CAPTIONS.md`: manuscript-ready captions for the main figure, table, and
  supplementary figure.

The supported interim conclusion is that MIPS-GRPO avoids the catastrophic
terminal collapse observed for GRPO and Count IPS-GRPO. Relative to RGFN,
MIPS-GRPO occupies a different quality-diversity operating point: greater
non-repetition, but lower mean sampled proxy and less mass above proxy 6.

This single seed does not establish run-to-run uncertainty, and the difference
between three MIPS leader modes and two RGFN leader modes is too small to claim
superiority.

## Supplementary material

- `figureS1_sampling_convergence.{png,pdf}`: accumulation of distinct sampled
  SMILES and stabilization of the running proxy mean for the two non-collapsed
  policies.

At the final 2,000-update checkpoint, replace this interim package with the
same layout using identical final-policy sample budgets for all four methods.
Add per-seed points and uncertainty only after seeds 1 and 2 are complete.

## Deliberately excluded

- `log_reward` versus proxy plots: `log_reward = 8 × proxy` by construction,
  so the perfect fit is tautological.
- The former training-discovery panel: it mixed RGFN leader modes with
  per-batch scaffold counts and placed `modes_0.xlsx` at zero oracle calls.
- Combined bars for modes, scaffolds, and ESS: the metrics have incompatible
  scales and ESS alone cannot detect missing support.
- Unlabelled top-frequency molecule bars: frequencies without structures or
  chemical annotations are not informative.
- Older 2,000-update GRPO/Count IPS-only figures: those are not a matched
  four-method checkpoint comparison.
- Mixed-length training curves: the available figure combined resumed
  2,000-update baselines with 500-update MIPS/RGFN histories.

## Reproduction

The input values and their provenance are frozen in `metrics_500.json`.

```bash
python -m molecule_synthesis.plot_seh_checkpoint_paper \
  --metrics-json molecule_synthesis/runs/seh_paper_medium/results/paper_500/metrics_500.json \
  --output-dir molecule_synthesis/runs/seh_paper_medium/results/paper_500
```
