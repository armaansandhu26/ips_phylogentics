# seh_paper_medium — Experimental Results

Living results document for the full-chemistry sEH comparison (`seh_paper_medium`).
Update this file as each method/seed completes. Intended for paper tables and figures.

**Last updated:** 2026-08-27

---

## Task configuration

| Setting | Value |
|---|---|
| Suite | `seh_paper_medium` |
| Chemistry | Full RGFN sEH proxy (350 fragments, 66 reactions, 132 anchored) |
| Config | `rgfn/configs/rgfn_seh_proxy.gin` |
| Max reactions | 4 |
| Target iterations | 2,500 (paper medium); some runs stopped at 2,000 for pilot |
| Forward trajectories / update | 100 |
| Forward batch size | 100 |
| Final evaluation samples | 50,000 (paper); 5,000 used for quick collapse checks |
| Mode threshold | proxy ≥ 7.0 |
| Mode similarity | Tanimoto ≥ 0.5 |
| Cluster | NSCC Hopper (H200 / batch `small`) |
| Environment | Python 3.11, PyTorch 2.3.0+cu118, miniconda `py311` |

### Method-specific training settings

| Method | Replay traj/update | Forward LR | Notes |
|---|---:|---:|---|
| RGFN | 20 | 0.001 | Trajectory balance + replay |
| GRPO | 0 | 0.001 | On-policy, uniform reverse |
| Count IPS-GRPO | 0 | 0.001 | On-policy, count IPS estimator |
| MIPS-GRPO | 0 | 0.0001 | Learned reverse, running advantage norm |

---

## Collapse criteria (operational)

A run is considered **collapsed** at final-checkpoint sampling when:

- `n_unique ≤ 1` (strict collapse), or
- `n_unique` very small with **>95%** mass on a single SMILES (practical collapse)

Healthy runs (MIPS/RGFN expected): `n_unique` in the hundreds+ at 20k–50k samples, non-zero `n_modes`, competitive `mean_proxy`.

**Important:** Training-trajectory diversity (`num_unique_molecules` during training) is **not** a reliable collapse diagnostic. Always evaluate by **sampling from `last_gfn.pt`**.

---

## Summary table — seed 0 (final-checkpoint sampling)

Quick-check samples (5k) and paper-scale samples (50k) where available.

| Method | Iters | n_sampled | n_unique | unique_frac | mean_proxy | n_modes | Top-1 share | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GRPO | 500 | 5,000 | 3 | 0.06% | 4.15 | 0 | 98.9% | **Collapsed** |
| GRPO | 1,500 | 5,000 | 1 | 0.02% | 3.81 | 0 | 100.0% | **Collapsed** |
| GRPO | 2,000 | 50,000 | 1 | 0.002% | 3.73 | 0 | 100.0% | **Collapsed** |
| Count IPS-GRPO | 500 | 5,000 | 2 | 0.04% | 6.04 | 0 | 99.96% | **Collapsed** |
| Count IPS-GRPO | 2,000 | 50,000 | 10 | 0.02% | 4.80 | 0 | 83.9% | **Collapsed** |
| MIPS-GRPO | 500 | 5,000 | 4,923 | 99.4% | 4.11 | 3 | 0.08% | **Healthy** |
| RGFN | 500 | 5,000 | 3,965 | 79.3% | 5.15 | 2 | 0.96% | **Healthy** |

### Reduced-space pilot (reference only — `seh_reduced_a100`, 1,200 iters, 20k samples)

Not directly comparable to full chemistry; included for context.

| Method | n_unique | mean_proxy | n_modes | Top-1 share |
|---|---:|---:|---:|---:|
| GRPO | 152 | 7.50 | 11 | ~31% |
| MIPS-GRPO | 170 | 7.53 | 11 | ~37% |
| RGFN | 152 | 7.50 | 11 | ~31% |

Neither method collapsed on the **reduced** task at 1,200 iterations; both were mode-peaked but diverse.

---

## Seed 0 — detailed results

Run root: `molecule_synthesis/runs/seh_paper_medium/<method>/seed_0/batch/`  
(MIPS uses `.../mips_grpo/seed_0/20260827_112154/` from interrupted interactive pilot.)

---

### GRPO — seed 0

**Run directory:** `grpo/seed_0/batch`

#### Training

| Phase | Iterations | Wall time | Train unique (final batch era) | Train proxy mean (final) |
|---|---:|---|---:|---:|
| Phase 1 | 500 | ~11.6 min | 2,241 | 8.05 |
| Phase 2 | 500 → 1,500 | ~8.9 min | 9 | 5.17 |
| Phase 3 | 1,500 → 2,000 | ~8.9 min | 17 | 3.43 |

#### Final-checkpoint sampling

**500 iterations (5,000 samples)**

| Metric | Value |
|---|---:|
| n_unique | 3 |
| unique_fraction | 0.085% |
| mean_proxy | 4.15 |
| n_modes | 0 |
| valid_fraction | 1.0 |
| Top-1 SMILES share | 98.9% (4,928 / 5,000) |

Dominant SMILES (truncated): `O=C(Nc1ccc(-c2cccnc2)cc1)C1CCN1`

**1,500 iterations (5,000 samples)**

| Metric | Value |
|---|---:|
| n_unique | 1 |
| unique_fraction | 0.02% |
| mean_proxy | 3.81 |
| n_modes | 0 |
| Top-1 SMILES share | 100.0% |

Dominant SMILES: `Cc1cccc(C)c1NC(=O)N1CCC(OC(=O)c2cccnc2)CC1`

**2,000 iterations (50,000 samples)**

| Metric | Value |
|---|---:|
| n_unique | 1 |
| unique_fraction | 0.002% |
| mean_proxy | 3.73 |
| n_modes | 0 |
| valid_fraction | 1.0 |
| importance_ess_fraction | 1.0 |
| Top-1 SMILES share | 100.0% (50,000 / 50,000) |

Dominant SMILES: `C[C@@H](NC(=O)c1cc[nH]n1)c1ccccc1`

#### Interpretation

- Severe collapse by 500 iterations on full chemistry; **strict single-molecule collapse** by 1,500.
- More training **worsened** eval diversity and **lowered** sampled proxy (8.05 train batch → 3.7 sample).
- Zero modes above threshold 7.0 at all checkpoints tested.
- Canonical negative control for on-policy GRPO on this task.

---

### Count IPS-GRPO — seed 0

**Run directory:** `count_ips_grpo/seed_0/batch`

#### Training

| Phase | Iterations | Wall time | Train unique (final batch era) | Train proxy mean (final) | ips_dup_frac | ips_unique_out |
|---|---:|---|---:|---:|---:|---:|
| Phase 1 | 500 | ~11.2 min | 1,871 | 7.77 | 0.99 | 1.0 |
| Phase 2 | 500 → 2,000 | ~10 min | 23 | 5.43 | 0.99 | 1.0 |

#### Final-checkpoint sampling

**500 iterations (5,000 samples)**

| Metric | Value |
|---|---:|
| n_unique | 2 |
| unique_fraction | 0.04% |
| mean_proxy | 6.04 |
| n_modes | 0 |
| valid_fraction | 1.0 |
| importance_ess_fraction | 0.007 |
| Top-1 SMILES share | 99.96% (4,998 / 5,000) |

Dominant SMILES (truncated): `N#Cc1ccc(N2CCN(CCNC(=O)NCCN3CCN(c4ccc(-c5nnn(CC6CC6)n5)...`

**2,000 iterations (50,000 samples)**

| Metric | Value |
|---|---:|
| n_unique | 10 |
| unique_fraction | 0.02% |
| mean_proxy | 4.80 |
| n_modes | 0 |
| valid_fraction | 1.0 |
| importance_ess_fraction | 0.00093 |
| Top-1 SMILES share | 83.9% (41,942 / 50,000) |

Dominant SMILES (truncated): `COc1ccc(Cn2nnc(-c3ccc(N4CCC[C@H]4C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(N)=O)C(=O)O)nc3)n2)cc1`

All 10 unique SMILES are peptide side-chain variants of the same methoxyphenyl-tetrazole scaffold; none reach proxy ≥ 7.0.

#### Interpretation

- **Worse collapse than GRPO at 500 iterations** (2 vs 3 unique; 99.96% vs 98.9% top-1).
- Training diagnostics already showed collapse pressure (`ips_duplicate_fraction = 0.99`).
- **500 → 2,000 training degraded quality:** train unique fell 1,871 → 23, train proxy 7.77 → 5.43.
- **50k sampling at 2,000 iters:** slightly more eval diversity than GRPO (10 vs 1 unique, 83.9% vs 100% top-1); mean_proxy higher than GRPO (4.80 vs 3.73) but **down from 6.04 at 500 iters**.
- Cluster collapse around one peptide scaffold family, not strict single-molecule collapse.
- Importance weights nearly degenerate at eval (`importance_ess_fraction ≈ 0.0009`).
- Zero modes above threshold 7.0 at all checkpoints tested.
- Negative control for count IPS-GRPO on full chemistry; reduced pilot (170 unique / 20k) not reproduced at scale.

---

### MIPS-GRPO — seed 0

**Run directory:** `mips_grpo/seed_0/20260827_112154`

#### Training

| Phase | Iterations | Train unique (final batch) | Train proxy mean (final) |
|---|---:|---:|---:|
| Phase 1 | 500 | 33,781 | 5.30 |

#### Final-checkpoint sampling

**500 iterations (5,000 samples)**

| Metric | Value |
|---|---:|
| n_unique | 4,923 |
| unique_fraction | 99.4% |
| mean_proxy | 4.11 |
| n_modes | 3 |
| valid_fraction | 99.1% |
| importance_ess_fraction | 0.00079 |
| Top-1 SMILES share | 0.08% (4 / 4,953 valid) |
| proxy ≥ 7.0 | 4 samples (0.08%) |
| proxy ≥ 6.0 | 78 samples (1.6%) |
| max_proxy | 7.24 |

Top modes (proxy ≥ 7.0): bis-amide quinoline (7.24), fluorinated triazole variants (7.18, 7.14).

#### Interpretation

- **No collapse** at 500 iterations — qualitatively different from GRPO / Count IPS.
- High eval diversity (99.4% unique) but **low mean sampled proxy** (4.11 vs 5.30 train batch).
- Modes exist but are extremely sparse; learned-reverse importance weights are heavy-tailed at eval.
- Reduced pilot (170 unique / 20k, mean_proxy 7.53, 11 modes) not yet matched at full chemistry scale.

---

### RGFN — seed 0

**Run directory:** `rgfn/seed_0/batch`

#### Training

| Phase | Iterations | Train unique (final batch) | Train proxy mean (final) |
|---|---:|---:|---:|
| Phase 1 | 500 | 43,478 | 7.13 |

#### Final-checkpoint sampling

**500 iterations (5,000 samples)**

| Metric | Value |
|---|---:|
| n_unique | 3,965 |
| unique_fraction | 79.3% |
| mean_proxy | 5.15 |
| n_modes | 2 |
| valid_fraction | 100.0% |
| importance_ess_fraction | 0.00039 |
| Top-1 SMILES share | 0.96% (48 / 5,000) |
| proxy ≥ 7.0 | 2 samples (0.04%) |
| proxy ≥ 6.0 | 366 samples (7.3%) |
| max_proxy | 7.19 |

Top modes: quinoline–proline amide (7.19), methoxypyridine–tetrazole scaffold (7.05).

#### Interpretation

- **No collapse** at 500 iterations — diverse final policy like MIPS.
- **Best mean sampled proxy** among all four methods at 500 iters (5.15 vs 4.11 MIPS, 6.04 Count IPS collapsed, 4.15 GRPO collapsed).
- Largest train→sample gap (7.13 train → 5.15 sample); replay helps training diversity (43k train unique) but eval still spreads mass across mid-proxy region.
- Modes rarer than MIPS count (2 vs 3) but higher fraction above proxy 6 (7.3% vs 1.6%).

---

### MIPS vs RGFN at 500 iterations (head-to-head)

| Metric | MIPS-GRPO | RGFN | Winner |
|---|---:|---:|---|
| n_unique / 5k | 4,923 | 3,965 | MIPS |
| unique_fraction | 99.4% | 79.3% | MIPS |
| mean_proxy | 4.11 | **5.15** | RGFN |
| n_modes | **3** | 2 | MIPS |
| proxy ≥ 6 fraction | 1.6% | **7.3%** | RGFN |
| Top-1 share | **0.08%** | 0.96% | MIPS |
| train proxy mean | 5.30 | **7.13** | RGFN |
| train→sample gap | 1.19 | **1.98** | MIPS (smaller gap) |
| Collapsed? | No | No | — |

Both methods pass the operational collapse test. RGFN delivers higher proxy mass; MIPS delivers higher uniqueness and more discovered modes.

## Seeds 1 and 2

*Not yet started.*

| Method | Seed 1 | Seed 2 |
|---|---|---|
| GRPO | TBD | TBD |
| Count IPS-GRPO | TBD | TBD |
| MIPS-GRPO | TBD | TBD |
| RGFN | TBD | TBD |

When complete, add mean ± std rows here for paper tables.

---

## Hypothesis checklist (seed 0, in progress)

| Prediction | GRPO | Count IPS | MIPS | RGFN |
|---|---|---|---|---|
| Final-policy collapse on full chemistry | ✅ Yes | ✅ Yes (500/2000 it) | ❌ No | ❌ No |
| High training diversity misleading | ✅ Yes | ✅ Yes | ⚠️ Partial (train 5.3 → sample 4.1) | ⚠️ Partial (train 7.1 → sample 5.1) |
| Non-zero modes at proxy ≥ 7 | ❌ No | ❌ No | ✅ Yes (3) | ✅ Yes (2) |
| Stable proxy under sampling | ❌ No (↓ with training) | ❌ No (↓ with training) | ⚠️ Large train→sample gap | ⚠️ Large train→sample gap |

---

## Run commands (reference)

### Resume training

```bash
python -m molecule_synthesis.train \
  --method <grpo|count_ips_grpo|mips_grpo|rgfn> \
  --cfg molecule_synthesis/external/RGFN/configs/rgfn_seh_proxy.gin \
  --run-name seh_paper_medium/<method>/seed_0/batch \
  --checkpoint-path <run_dir>/train/checkpoints/last_gfn.pt \
  --iterations <FINAL_ITER> \
  ...
```

### Final-checkpoint sampling

```bash
python -m molecule_synthesis.sample \
  --run-dir <run_dir> \
  --n-samples 50000 \
  --batch-size 100 \
  --device cuda \
  --mode-threshold 7.0 \
  --similarity-threshold 0.5 \
  --scaffold-thresholds 7.0,8.0
```

### Batch submission (phase 1, parallel)

```bash
bash submit_phase1.sh   # four methods → 500 iter
bash check_phase1.sh    # verify epoch 499
bash submit_phase2.sh   # four methods → 2500 iter + sample
```

---

## File paths

| Method | Run directory |
|---|---|
| GRPO | `molecule_synthesis/runs/seh_paper_medium/grpo/seed_0/batch/` |
| Count IPS-GRPO | `molecule_synthesis/runs/seh_paper_medium/count_ips_grpo/seed_0/batch/` |
| MIPS-GRPO | `molecule_synthesis/runs/seh_paper_medium/mips_grpo/seed_0/20260827_112154/` |
| RGFN | `molecule_synthesis/runs/seh_paper_medium/rgfn/seed_0/batch/` |

Key artifacts per run:

```text
<run_dir>/
├── manifest.json
├── train/checkpoints/last_gfn.pt
└── samples/
    ├── summary.json      ← primary metrics for this document
    ├── progress.json     ← live sampling progress (during sample runs)
    ├── samples.jsonl
    └── modes.jsonl
```

Seed-0 paper-style figures (all four methods; 500-iter checkpoint):

```text
molecule_synthesis/runs/seh_paper_medium/results/figures/
├── main/
│   └── main_figure.{png,pdf}          # 3-panel RGFN-style figure
└── supplementary/
    ├── supp_training_curves.{png,pdf}
    ├── supp_ips_duplicate_fraction.{png,pdf}
    ├── supp_sampling_discovery.{png,pdf}
    ├── supp_top_molecules.{png,pdf}
    ├── supp_log_proxy_vs_log_reward.{png,pdf}
    └── supp_train_vs_sample_proxy.{png,pdf}
```

Regenerate paper figures:

```bash
python -m molecule_synthesis.plot_seh_figures \
  --suite-dir molecule_synthesis/runs/seh_paper_medium \
  --checkpoint-iter 500
```

Sample-based panels use on-disk 5k samples for MIPS-GRPO and RGFN at iter 500. Train-vs-sample panel includes all four methods (GRPO / Count IPS sample values from their documented 500-iter 5k runs).

---

## Changelog

| Date | Update |
|---|---|
| 2026-08-27 | Seed 0: GRPO collapse documented at 500/1500/2000 iters; Count IPS collapse at 500/2000 iters (50k sample); MIPS/RGFN batch in progress |
| 2026-08-27 | Seed 0 @500 iter: MIPS-GRPO and RGFN sampling complete (5k each); both healthy (no collapse); 500-iter figures regenerated |
