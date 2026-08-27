# Molecule-synthesis experiment runbook

This document is the practical runbook for comparing our molecule-synthesis
methods in the reaction environment from the
[RGFN paper](https://arxiv.org/abs/2406.08506) and
[RGFN repository](https://github.com/koziarskilab/RGFN).

The comparison contains exactly these four methods:

| CLI name | Meaning |
|---|---|
| `rgfn` | Reaction GFlowNet trained with trajectory balance |
| `grpo` | Plain group-relative policy optimization |
| `count_ips_grpo` | Original count-based IPS-GRPO from [arXiv:2601.21669](https://arxiv.org/abs/2601.21669) |
| `mips_grpo` | Our MIPS-GRPO using a learned reverse trajectory model |

There is no separate `trajectory_ips_grpo` method. `count_ips_grpo` is the
original IPS-GRPO baseline; `mips_grpo` is the learned-reverse version.

## 1. What experiment should be run for the paper?

The RGFN paper evaluates four biological objectives rather than eight datasets:

1. sEH using the released learned proxy;
2. senolytic activity using a learned proxy;
3. DRD2 activity using a learned proxy;
4. ClpP using direct docking.

For the first main result, use **sEH**. It is the closest analogue to selecting
the first dataset from the phylogenetics paper because it is RGFN's canonical
public task, its proxy checkpoint is available, and it does not require the
private senolytic assets or a Vina-GPU docking installation.

The publication-scale comparison is therefore:

- task: public sEH proxy;
- methods: RGFN, GRPO, count IPS-GRPO, and MIPS-GRPO;
- seeds: 0, 1, and 2;
- training: 4,000 updates and 100 new trajectories per update;
- maximum synthesis depth: four reactions;
- final-checkpoint evaluation: 100,000 independently sampled molecules;
- shared forward architecture, reaction templates, fragments, proxy, reward
  transform, optimizer budget, and seed set.

This is already encoded in
[`configs/suites/seh_paper_main.json`](configs/suites/seh_paper_main.json).
RGFN uses its paper-standard 20 replay trajectories per update. The three GRPO
variants are run on-policy with replay disabled because their policy ratios and
propensity corrections are defined using the current sampling policy.

## 2. Can this run on CPU?

Yes for correctness checks and small pilots; a GPU is strongly recommended for
the full sEH experiment.

| Stage | CPU suitability | Purpose |
|---|---|---|
| Four-state toy | Excellent; seconds | Verify objective math and learned reverse model |
| Enumerable MiniChem pilot | Excellent; roughly minutes | Verify real chemistry and exact reward-proportional sampling |
| One-update full-chemistry smoke | Good | Verify RGFN, RDKit, DGL, proxy, and all four trainers |
| Reduced sEH pilot | Possible, but slow | Find crashes and gross learning failures |
| Full sEH, one method and seed | Impractical on a laptop CPU | Publication run |
| Full four-method, three-seed suite | Use GPUs | Main result with uncertainty bars |

One paper-scale method/seed performs 400,000 training trajectories and then
draws 100,000 final samples. The complete suite performs 4.8 million training
trajectories plus 1.2 million evaluation samples. The RGFN paper reports roughly
24 GPU-hours for one proxy-task run on a Quadro RTX 8000, so CPU should be
treated as a development platform rather than the final experimental platform.
Hardware changes runtime, not the scientific definition of the method: CPU
results are valid if every method receives the same model and sampling budget.

On a laptop, run methods sequentially to avoid memory pressure and thermal
throttling. Do not mix reduced CPU budgets with full GPU budgets in the same
headline comparison.

## 3. One-time installation

Run all commands from the repository root:

```bash
cd /Users/armaansandhu/Desktop/projects/ips_phylogentics
```

### Recommended Linux/A100 setup

For a cloned repository on an A100 server, the environment definition is
already tracked in Git. Do **not** upload a virtual environment: it is large,
platform-specific, and contains absolute paths. The bootstrap recreates it
from the pinned CUDA requirements, pinned RGFN commit, and tracked patch.

The exact one-command path for the first paper run is:

```bash
git clone https://github.com/armaansandhu26/ips_phylogentics.git
cd ips_phylogentics

bash molecule_synthesis/scripts/run_paper_mips_a100.sh
```

Run that command inside a scheduler allocation or `tmux`, since the process
must survive for several hours. It creates/reuses a Python 3.11 virtual
environment, installs the pinned CUDA 11.8 PyTorch/DGL stack plus only the
dependencies required by QED and public sEH, checks out and patches the pinned
RGFN commit, downloads the public sEH checkpoint, verifies CUDA, runs the MIPS
tests and preflight, records `pip freeze` and the code commit, dry-runs the
exact command, and then launches **paper-scale MIPS seed 0**. It also draws the
100,000 final-policy samples after training and ends with `MIPS_HEALTH=PASS`
only if the frozen configuration, artifacts, finite diagnostics, and basic
non-collapse check all pass.

The only final launch command executed by the wrapper is:

```bash
/usr/bin/time -p .venv-rgfn-cu118/bin/python \
  -m molecule_synthesis.pipeline \
  --suite seh_paper_main \
  --seed 0 \
  --method mips_grpo \
  --device cuda \
  --wandb-mode offline
```

Use `--wandb-mode offline`: this preserves local training history without
requiring a W&B login or network connection. Run outputs, checkpoints, samples,
the exact environment snapshot, and code commit are kept under
`molecule_synthesis/runs/seh_paper_main/`.

For repeated clones or jobs, keep the environment and wheel cache on persistent
storage. Substitute paths that exist on the server:

```bash
bash molecule_synthesis/scripts/bootstrap_a100.sh \
  --venv /persistent/path/rgfn-cu118 \
  --cache-dir /persistent/path/pip-cache

source /persistent/path/rgfn-cu118/bin/activate
```

To combine those persistent paths with the one-command launch, use:

```bash
bash molecule_synthesis/scripts/run_paper_mips_a100.sh \
  --venv /persistent/path/rgfn-cu118 \
  --cache-dir /persistent/path/pip-cache
```

The first setup performs the downloads. Subsequent clones reuse the installed
packages and cache, then quickly repoint the editable RGFN install to the new
clone. Do not add the binary virtual environment to Git: it is large,
platform-specific, and contains absolute paths. The tracked pinned requirements
file is `molecule_synthesis/environments/requirements-seh-cu118.txt`.

### CPU environment

```bash
bash molecule_synthesis/scripts/setup_env.sh --accelerator cpu
conda activate rgfn-molecules
python -m molecule_synthesis.preflight --strict-commit
```

### NVIDIA GPU environment (CUDA 11.8)

```bash
bash molecule_synthesis/scripts/setup_env.sh --accelerator cu118
conda activate rgfn-molecules
python -m molecule_synthesis.preflight --strict-commit
```

The setup script installs Python 3.11, checks out RGFN at the pinned commit
`6ce59169f855ed18f34ba4e8279de93bee306e4f`, applies the minimal proxy import
patch, and installs the required chemistry and plotting packages.

Before the sEH experiment, fetch and validate the public checkpoint:

```bash
python -m molecule_synthesis.prefetch_assets --seh
python -m molecule_synthesis.preflight --strict-commit
```

## 4. Recommended staged workflow

Run these stages in order. Only advance after the current stage passes.

### Stage A: objective-only CPU toy

This does not require RGFN, RDKit, or DGL:

```bash
bash molecule_synthesis/scripts/toy_smoke.sh
```

A successful run ends with `toy_verification=PASS`. It checks a four-terminal
state space in which the exact target is known and verifies that count IPS and
learned-reverse MIPS recover reward-proportional sampling.

Outputs:

```text
molecule_synthesis/toy/runs/latest/
```

### Stage B: unit and integration smoke tests

Dry-run the command construction and run the automated tests:

```bash
bash molecule_synthesis/scripts/smoke_test.sh
```

Execute a real one-update CPU pass for all four methods:

```bash
FULL_SMOKE=1 bash molecule_synthesis/scripts/smoke_test.sh
```

This uses RDKit QED and the complete reaction environment, but it avoids the
sEH proxy and docking dependencies.

### Stage C: enumerable real-chemistry CPU pilot

```bash
conda activate rgfn-molecules
python -m molecule_synthesis.pipeline --suite qed_cpu_pilot
```

This experiment enumerates all 112 reachable terminal molecules and 224 routes,
so it can compare final-checkpoint samples against the exact target

```text
P*(molecule) = exp(4 * QED(molecule)) / Z.
```

The completed reference smoke used 30 updates and 5,000 evaluation samples per
method. It achieved the following total-variation distances in that run:

| Method | TV distance | Sampled support |
|---|---:|---:|
| RGFN | 0.311 | 100.0% |
| GRPO | 0.907 | 22.3% |
| Count IPS-GRPO | 0.313 | 97.3% |
| MIPS-GRPO | 0.096 | 100.0% |

These are smoke-test results, not final paper claims. Their role is to establish
that the implementation executes and that MIPS can approach the exact
reward-proportional target on a small chemical space.

Generate the paper-style plots with:

```bash
python -m molecule_synthesis.plot_results \
  --suite-dir molecule_synthesis/runs/qed_cpu_pilot
```

Plots are written as PNG and PDF files under:

```text
molecule_synthesis/runs/qed_cpu_pilot/results/plots/
```

The fixed method colors are GRPO red, count IPS blue, MIPS green, and RGFN
orange, matching the visual style selected for the paper draft.

### Stage D: optional reduced sEH CPU pilot

Use this only as an integration and learning-curve pilot:

```bash
python -m molecule_synthesis.pipeline \
  --suite seh_small \
  --method all \
  --device cpu
```

`seh_small` uses 500 updates and 1,000 final samples. It is substantially
smaller than the paper configuration and uses its own pilot learning rate, so
label all resulting tables and plots **reduced CPU pilot**, not paper-scale
sEH. If turnaround is more important than comparing all methods at once, run
one at a time:

```bash
python -m molecule_synthesis.pipeline --suite seh_small --method rgfn --device cpu
python -m molecule_synthesis.pipeline --suite seh_small --method grpo --device cpu
python -m molecule_synthesis.pipeline --suite seh_small --method count_ips_grpo --device cpu
python -m molecule_synthesis.pipeline --suite seh_small --method mips_grpo --device cpu
```

### Stage D2: reduced-space sEH experiment on A100 (4–5 hour target)

When the full paper chemical space is too expensive, use the dedicated
`seh_reduced_a100` suite. It retains the released sEH proxy and `beta = 8`, but
uses a deterministic chemical language with 50 fragments, 14 reaction
templates (28 anchored directions), four common reaction families, and at most
two reactions per molecule.

The per-method/seed budget is:

- 1,200 updates;
- 64 new forward trajectories per update (76,800 total);
- 13 replay trajectories for RGFN and zero for the on-policy GRPO methods;
- 20,000 final-checkpoint samples;
- three seeds for the final aggregate.

This is 19.2% of the full experiment's forward-trajectory count, before the
additional savings from halving the maximum synthesis depth and reducing the
fragment/action space. It targets no more than 4–5 hours per method/seed on a
dedicated A100, but the first job must be timed because RDKit speed and CPU
allocation vary between clusters.

Fetch the proxy and run seed 0 for one method first:

```bash
python -m molecule_synthesis.prefetch_assets --seh
python -m molecule_synthesis.preflight --strict-commit

/usr/bin/time -p python -m molecule_synthesis.pipeline \
  --suite seh_reduced_a100 \
  --seed 0 \
  --method mips_grpo \
  --wandb-mode disabled
```

If that job stays within budget, run the other three methods for seed 0. Run
them individually so the completed MIPS job is not repeated:

```bash
python -m molecule_synthesis.pipeline --suite seh_reduced_a100 --seed 0 --method rgfn
python -m molecule_synthesis.pipeline --suite seh_reduced_a100 --seed 0 --method grpo
python -m molecule_synthesis.pipeline --suite seh_reduced_a100 --seed 0 --method count_ips_grpo
```

After checking the seed-0 curves and samples, run all three seeds:

```bash
python -m molecule_synthesis.pipeline \
  --suite seh_reduced_a100 \
  --method all
```

Generate the sEH plots with:

```bash
python -m molecule_synthesis.plot_paper_task \
  --suite-dir molecule_synthesis/runs/seh_reduced_a100
```

In tables and captions, call this task **reduced-space sEH**, not the full RGFN
sEH benchmark. All four methods remain directly comparable within this task,
but its absolute rewards and mode counts should not be compared directly to the
paper's 350-fragment, four-reaction results.

### Stage E: paper-scale sEH on GPU

First list and dry-check the suites:

```bash
python -m molecule_synthesis.pipeline --list
python -m molecule_synthesis.preflight --strict-commit
```

Run corrected MIPS seed 0 first. This is deliberate: validate our method before
spending compute on any baseline.

```bash
bash molecule_synthesis/scripts/run_paper_mips_a100.sh
```

If setup has already passed and the environment is active, the exact equivalent
launch command is:

```bash
/usr/bin/time -p python -m molecule_synthesis.pipeline \
  --suite seh_paper_main \
  --seed 0 \
  --method mips_grpo \
  --device cuda \
  --wandb-mode offline
```

The paper-scale MIPS entry uses the corrected on-policy optimization recipe:
forward LR `1e-4`, reverse LR `1e-3`, four reverse MLE updates after each
forward update, reverse gradient clipping at 1, running importance-weight
normalization with decay `0.9`, advantage clipping at 10, and log-ratio
clipping at 20. Exploration and replay remain disabled for MIPS, so its weight
is still exactly `R(x) q_phi(tau|x) / P_F(tau)`.

Before starting baselines, require that seed 0 completed training and
final-checkpoint sampling, the loss and importance diagnostics are finite, the
reverse loss decreased, and the final policy did not collapse to a single
molecule. These are implementation-health checks, not a rule that MIPS must
beat every baseline on one seed.

The wrapper performs the machine-checkable portion automatically. To rerun it:

```bash
python -m molecule_synthesis.verify_mips_run \
  --suite-dir molecule_synthesis/runs/seh_paper_main \
  --seed 0
```

Inspect the offline W&B training history to confirm the reverse-loss trend;
the final health check additionally prints the final reverse loss, unique
molecules, modes, proxy mean, and importance ESS fraction.

Then run the other methods for seed 0:

```bash
python -m molecule_synthesis.pipeline --suite seh_paper_main --seed 0 --method rgfn --device cuda --wandb-mode offline
python -m molecule_synthesis.pipeline --suite seh_paper_main --seed 0 --method grpo --device cuda --wandb-mode offline
python -m molecule_synthesis.pipeline --suite seh_paper_main --seed 0 --method count_ips_grpo --device cuda --wandb-mode offline
```

If seed 0 completes and no configuration changes are made, retain it and run
seeds 1 and 2 rather than repeating seed 0:

```bash
python -m molecule_synthesis.pipeline \
  --suite seh_paper_main \
  --seed 1 \
  --method all

python -m molecule_synthesis.pipeline \
  --suite seh_paper_main \
  --seed 2 \
  --method all
```

If any hyperparameter is changed after inspecting seed 0, discard that pilot,
freeze the new configuration, and rerun all three seeds from scratch.

For a cluster, each method/seed can be submitted as an independent job. For
example:

```bash
python -m molecule_synthesis.pipeline --suite seh_paper_main --seed 0 --method rgfn
python -m molecule_synthesis.pipeline --suite seh_paper_main --seed 0 --method grpo
python -m molecule_synthesis.pipeline --suite seh_paper_main --seed 0 --method count_ips_grpo
python -m molecule_synthesis.pipeline --suite seh_paper_main --seed 0 --method mips_grpo
```

Repeat those four commands for seeds 1 and 2. The suite manifest merges the
completed method/seed jobs, so separate jobs do not discard previous results.

After all desired jobs finish, generate the main-task figures:

```bash
python -m molecule_synthesis.plot_paper_task \
  --suite-dir molecule_synthesis/runs/seh_paper_main
```

## 5. How the methods must be evaluated

Evaluation must sample from the **saved final checkpoint**, not reuse molecules
from training batches. All methods must use the same number of final samples,
sampling temperature, chemistry environment, and reward proxy.

### Enumerable MiniChem evaluation

Because every terminal molecule is known, report direct distributional fit:

- total-variation distance to `P*(x) proportional to R(x)`;
- Jensen-Shannon divergence;
- L1 distance;
- support coverage and target-mass coverage;
- sampled probability versus exact target probability;
- effective sample size and importance-weight spread;
- mean reward and top-k reward over unique molecules.

This experiment provides the cleanest direct evidence of reward-proportional
sampling.

### Full sEH evaluation

The full reaction space cannot be enumerated, so exact TV or JS distance is not
available. Report:

- the reward distribution from 100,000 final-checkpoint samples;
- normalized importance-weight effective sample size (ESS);
- log importance-weight spread and stability;
- number of distinct modes with sEH score above 7 and pairwise Tanimoto below
  0.5, following the RGFN protocol;
- Murcko scaffold counts above reward thresholds 7 and 8;
- top-500 unique-molecule reward statistics;
- QED, molecular weight, and synthetic-accessibility statistics for the top
  molecules;
- validity, uniqueness, diversity, and support discovery curves;
- mean and standard deviation across three seeds;
- training curves versus both update count and oracle calls.

Reward alone is not sufficient. A method that repeatedly generates a few
high-reward molecules can have a high average reward while failing to sample
proportionally or cover diverse modes. The ESS, mode, scaffold, uniqueness, and
diversity diagnostics detect that failure mode.

## 6. Output layout

Every run is written under:

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

Suite-level files are written under:

```text
molecule_synthesis/runs/<suite>/suite.json
molecule_synthesis/runs/<suite>/results/comparison.json
molecule_synthesis/runs/<suite>/results/comparison.csv
molecule_synthesis/runs/<suite>/results/plots/
```

Keep the run manifest, suite config, pinned RGFN commit, environment details,
random seed, checkpoint, raw samples, and aggregate metrics for every reported
result.

## 7. Minimum result set for the paper

The minimum defensible result is:

1. MiniChem exact-distribution validation showing whether each method samples
   proportionally to a known target;
2. public sEH results for all four methods at identical budgets;
3. three sEH seeds with mean and standard deviation;
4. final-checkpoint reward, mode, scaffold, uniqueness/diversity, and
   importance-weight diagnostics;
5. learning curves and paper-consistent plots;
6. a table documenting shared hyperparameters and the method-specific replay
   setting.

Run one full sEH seed before launching the remaining seeds. This catches proxy,
checkpoint, memory, and evaluation failures before committing the full compute
budget. The remaining three RGFN paper tasks can be added later as broader
evidence, but they are not required to establish the first sEH comparison.

## 8. Final pre-launch checklist

- [ ] `python -m molecule_synthesis.preflight --strict-commit` passes.
- [ ] The toy ends with `toy_verification=PASS`.
- [ ] Unit and integration smoke tests pass.
- [ ] MiniChem produces all four checkpoints, summaries, and plots.
- [ ] sEH proxy assets are present and load successfully.
- [ ] Every main method uses the same seed, proxy, trajectory budget, reaction
      depth, reward transform, and final sample count.
- [ ] RGFN replay is 20; GRPO-family replay is 0 and documented.
- [ ] Seed 0 completes and its final-checkpoint sampling is healthy.
- [ ] Seeds 1 and 2 complete before reporting mean and standard deviation.
- [ ] Tables distinguish CPU pilots from publication-scale GPU results.
- [ ] Raw samples and final checkpoints are archived with the aggregate plots.
