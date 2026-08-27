# Molecule synthesis: RGFN + GRPO + MIPS-GRPO

For the complete CPU-to-GPU execution sequence, paper-scale sEH protocol, and
evaluation checklist, see [`EXPERIMENT_RUNBOOK.md`](EXPERIMENT_RUNBOOK.md).

This folder ports the comparison harness used for phylogenetics to the
[Reaction-GFlowNet](https://github.com/koziarskilab/RGFN) reaction environment.
Every learned method uses the same reaction templates, fragments, reward proxy,
forward-policy architecture, batch, and optimizer settings.

| Method | Training signal | Purpose |
|---|---|---|
| `rgfn` | Upstream trajectory balance | Paper implementation |
| `grpo` | Group-normalized linear reward + PPO surrogate | Plain RL baseline |
| `count_ips_grpo` | `R / p_hat(molecule)` + GRPO | Original IPS-GRPO (empirical count estimator) |
| `mips_grpo` | `R * q_phi(trajectory|molecule) / P_F(trajectory)` + GRPO | MIPS with learned reverse policy |

`count_ips_grpo` is the local, explicit name for the original
[IPS-GRPO](https://arxiv.org/abs/2601.21669): within each sampled group,
`p_hat(molecule) = count(molecule) / group_size`, and GRPO receives
`R / max(p_hat, epsilon)`. The aliases `ips_grpo` and `ips-grpo` select this
same method; it does not use an auxiliary model.

The MIPS weight matters because a terminal molecule can have several valid
synthesis trajectories. Count IPS estimates terminal propensity directly from
within-batch molecule counts. MIPS instead learns a normalized reverse proposal
`q_phi(trajectory|molecule)` by maximum likelihood and uses
`log R + log q_phi - log P_F` for its forward-policy advantages. The GRPO and
importance-weight formulas mirror the phylogenetics implementation, while the
adapter remains independent of phylogenetics-only ETE/fvcore dependencies.

## Fast CPU toy verification

This is the quickest check before installing RGFN, RDKit, or DGL. It only needs
a CPU PyTorch environment:

```bash
bash molecule_synthesis/scripts/toy_smoke.sh
```

To select a particular Python environment:

```bash
PYTHON_BIN=/path/to/python molecule_synthesis/scripts/toy_smoke.sh
```

The toy has four terminal “molecules” with rewards `(1, 2, 4, 8)` and
`(1, 2, 4, 8)` distinct synthesis routes. The correct molecule target is
`P(x) proportional to R(x)`. It checks that count IPS and learned-reverse MIPS
recover that target, that `q_phi` learns the forward route distribution
conditioned on each molecule, and that plain GRPO concentrates on the maximum.

A passing run takes a few seconds on CPU and ends with
`toy_verification=PASS`. Detailed curves and distributions are saved under
`molecule_synthesis/toy/runs/latest/`.

## Enumerable CPU chemistry pilot

The first real-chemistry comparison uses a deterministic subset of the paper's
released chemical language: eight acids, eight amines, the two amide synthesis
templates, and one reaction per molecule. It enumerates all reachable terminal
molecules before training, computes the exact target
`P*(molecule) = exp(4 * QED) / Z`, trains all four methods, then reports TV,
Jensen-Shannon, and L1 distance from final-checkpoint samples. It also reports
support and target-mass coverage, probability calibration, and top-k reward over
unique molecules (duplicates are removed before discovery metrics).

```bash
conda activate rgfn-molecules
python -m molecule_synthesis.pipeline --suite qed_cpu_pilot
```

The enumerated target and comparison table are written to
`molecule_synthesis/runs/qed_cpu_pilot/`. This is a one-seed pilot intended to
catch method and chemistry integration issues before longer multi-seed runs.

Generate PNG and PDF plots from a completed suite with:

```bash
python -m molecule_synthesis.plot_results \
  --suite-dir molecule_synthesis/runs/qed_cpu_pilot
```

This writes a compact method summary, per-molecule probability calibration,
terminal-distribution panels, support discovery curves, and the sampled-versus-
target QED distribution under `results/plots/`. Colors and panel styling follow
the accompanying paper: GRPO red, count IPS blue, MIPS green, and RGFN orange.

## Primary paper-scale experiment: sEH

The RGFN paper's main head-to-head uses four biological oracles rather than
eight datasets: public sEH, senolytic, and DRD2 proxies, plus direct ClpP
docking. For the analogue of a single primary phylogenetics dataset, use sEH.
It is the paper repository's default experiment, has public proxy weights, and
does not require the private senolytic checkpoints or Vina-GPU docking setup.

The `seh_paper_main` suite compares exactly our four methods over seeds 0, 1,
and 2. It uses the paper appendix's 4,000 updates, 100 new trajectories per
update, four-reaction maximum, learning rate 1e-3, and sEH reward beta 8. RGFN
also receives its paper-standard 20 replay trajectories per update; GRPO,
count IPS-GRPO, and MIPS-GRPO disable replay because their one-step policy ratio
and propensity correction are on-policy. The released upstream config defaults
to 5,002 updates, but the suite deliberately overrides this to the 4,000 stated
in the paper and its 400,000-oracle-call comparison.

Prepare and validate the public proxy before submitting GPU jobs:

```bash
python -m molecule_synthesis.prefetch_assets --seh
python -m molecule_synthesis.preflight --strict-commit
```

Run one seed first—the closest analogue of using phylogenetics DS1:

```bash
python -m molecule_synthesis.pipeline \
  --suite seh_paper_main \
  --seed 0 \
  --method all
```

Run all three paper seeds sequentially with:

```bash
python -m molecule_synthesis.pipeline --suite seh_paper_main --method all
```

Individual jobs can be run without losing earlier suite results; the manifest
merges completed method/seed pairs:

```bash
python -m molecule_synthesis.pipeline --suite seh_paper_main --seed 0 --method rgfn
python -m molecule_synthesis.pipeline --suite seh_paper_main --seed 0 --method mips_grpo
```

Each final checkpoint produces 100,000 evaluation molecules. Since the full
reaction space cannot be enumerated, TV to an exact target is unavailable. The
suite instead records normalized importance-weight ESS and log-weight spread as
reward-proportional diagnostics, plus the paper's leader-mode count (sEH > 7,
Tanimoto < 0.5), Murcko scaffold counts above 7 and 8, reward distributions,
and QED/MW/SA statistics over the top 500 modes. Results are aggregated per seed
and as mean ± standard deviation under `results/`.

Generate the matching paper-style figures after the run:

```bash
python -m molecule_synthesis.plot_paper_task \
  --suite-dir molecule_synthesis/runs/seh_paper_main
```

The paper reports approximately 24 GPU-hours for one proxy-task run on a Quadro
RTX 8000. The complete 4-method × 3-seed suite is therefore a substantial job;
the single-seed command is the recommended first GPU launch.

## 1. Install

RGFN is pinned to commit `6ce59169f855ed18f34ba4e8279de93bee306e4f` and kept
out of this Git repository under `molecule_synthesis/external/RGFN`.

```bash
# CPU environment
bash molecule_synthesis/scripts/setup_env.sh --accelerator cpu

# or CUDA 11.8
bash molecule_synthesis/scripts/setup_env.sh --accelerator cu118

conda activate rgfn-molecules
python -m molecule_synthesis.preflight --strict-commit
```

The CUDA option follows upstream's PyTorch 2.3.0/DGL 2.2.1 setup. CPU uses the
upstream DGL 1.1.2 recommendation. Python 3.11 is required; the repository's
system Python should not be used if it is newer.

### Fast Linux/A100 bootstrap

For the public sEH and QED experiments on a Linux A100 server, use the smaller
cache-aware environment instead of installing every optional upstream oracle:

```bash
bash molecule_synthesis/scripts/bootstrap_a100.sh
source .venv-rgfn-cu118/bin/activate
```

To reuse the environment and downloaded wheels across future clones, place
them on persistent scratch or shared storage:

```bash
bash molecule_synthesis/scripts/bootstrap_a100.sh \
  --venv /persistent/path/rgfn-cu118 \
  --cache-dir /persistent/path/pip-cache

source /persistent/path/rgfn-cu118/bin/activate
```

The first call installs the pinned stack; later calls reuse it and only refresh
the editable RGFN path for the current clone. The minimal environment supports
QED and public-sEH proxy runs. It intentionally excludes DRD2/PyTDC, private
senolytic models, direct docking, Jupyter, and development-only dependencies.

## 2. Smoke tests

The default script runs unit checks, validates imports and the pinned checkout,
and dry-runs all four commands:

```bash
bash molecule_synthesis/scripts/smoke_test.sh
```

Run an actual one-iteration CPU training pass for all methods:

```bash
FULL_SMOKE=1 bash molecule_synthesis/scripts/smoke_test.sh
```

The smoke suite uses RDKit QED, so it does not download the sEH proxy weights or
require docking software.

## 3. Run comparisons

```bash
# List suites
python -m molecule_synthesis.pipeline --list

# One method
python -m molecule_synthesis.pipeline --suite qed_smoke --method mips-grpo

# All comparable methods
python -m molecule_synthesis.pipeline --suite seh_small --method all --device cuda
```

The first sEH run downloads the pretrained proxy used by upstream RGFN. Full
docking suites additionally require the external Vina-GPU setup documented by
RGFN; they are intentionally not part of the offline smoke test.

Each pipeline trains, reloads the final checkpoint, samples terminal molecules,
and writes basic validity/diversity/reward metrics. Runs are written to:

```text
molecule_synthesis/runs/<suite>/<method>/<timestamp>/
├── manifest.json
├── train/checkpoints/last_gfn.pt
├── samples/samples.jsonl
├── samples/summary.json
├── unique_molecules/
├── all_molecules/
└── modes/
```

The suite-level method-to-run mapping is
`molecule_synthesis/runs/<suite>/suite.json`; sample summaries are aggregated to
`results/comparison.json` and `results/comparison.csv`.

## Standalone training and overrides

```bash
python -m molecule_synthesis.train \
  --method mips_grpo \
  --cfg molecule_synthesis/configs/rgfn_qed.gin \
  --iterations 100 \
  --forward-trajectories 64 \
  --replay-trajectories 16 \
  --batch-size 80 \
  --max-reactions 4 \
  --device cuda
```

Use `--reverse-loss-weight` to control the learned-reverse MLE contribution.
Count IPS exposes `--count-probability-floor` for numerical protection.

## Scope and comparison caveats

- `rgfn` optimizes trajectory balance and learns its backward policy.
- GRPO and count IPS update only the forward policy. MIPS additionally trains
  RGFN's reverse reaction policy by trajectory maximum likelihood; its forward
  importance weights use a detached `q_phi` from the current batch.
- Upstream RGFN retains its configured exploration mixture. GRPO-family runners
  bind the sampler directly to the learned forward policy because their PPO
  ratios and exact propensities must be on-policy.
- Replay data generated by upstream RGFN is supported by the shared trainer, but
  the one-step PPO ratio is on-policy. Start with replay disabled when validating
  a new proxy; treat replay-enabled GRPO results as an off-policy ablation.
- Compare methods with identical seeds, reaction depth, proxy, reward `beta`, and
  numbers of sampled trajectories. The suite files enforce those shared values.
