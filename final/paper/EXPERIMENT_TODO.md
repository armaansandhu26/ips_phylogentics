# Experiment progress tracker

Living checklist for paper/revision experiments. Update this file as runs finish.

**Last updated:** Aug 27, 2026 ~20:57 — **71% primary path**; hg64 s2 + 27t LR-IPS co-scheduled on GPUs 0–2  
**Related docs:** [`AUG25_COMPLETED_RUNS.md`](AUG25_COMPLETED_RUNS.md) · [`PAPER_REVISION_PRIORITIES.md`](PAPER_REVISION_PRIORITIES.md) · [`P1_RESULTS.md`](P1_RESULTS.md) · [`draft1/README.md`](draft1/README.md)

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Done — train + eval + plots |
| 🔄 | Active — training or eval in progress |
| ⏳ | Planned — not started |
| ❌ | Failed / blocked |
| 🔸 | Partial — some steps done (e.g. train only, or seed 0 only) |

Checkbox columns: `[x]` done · `[ ]` not done

---

## Dashboard (at a glance)

### Primary

| Section | Done | Active | Planned | Failed |
|---------|-----:|-------:|--------:|-------:|
| Hyper-Grid 64 | 8 | 3 | 1 | 0 |
| Hyper-Grid 8 b256 | 9 | 0 | 0 | 0 |
| Phylogenetics 27t | 2 | 4 | 0 | 1 |
| Analysis / paper | 8 | 1 | 3 | 0 |

### Extras

| Section | Done | Active | Planned | Failed |
|---------|-----:|-------:|--------:|-------:|
| Hyper-Grid 4096 | 2 | 0 | 0 | 0 |
| Phylogenetics 5t | 4 | 0 | 0 | 0 |
| Phylogenetics 10t | 1 | 0 | 4 | 0 |

<!-- PROGRESS_START -->

## Overall progress

**Primary path (paper-critical): 71%** `██████████████░░░░░░`

*Method: each run cell = 50% train + 25% eval + 25% plots. Active runs credit train fraction only. Refresh: `.venv/bin/python final/paper/progress_summary.py --patch`*

| Track | Weight | Progress | Bar |
|-------|-------:|---------:|:---:|
| Hyper-Grid 64 (4×3 seeds) | 12 | **68%** | `██████████████░░░░░░` |
| Hyper-Grid 8 b256 | 9 | **100%** | `████████████████████` |
| Phylo 27t (LR-IPS + PhyloGFN ×3) | 6 | **38%** | `████████░░░░░░░░░░░░` |
| Analysis / paper deliverables | 12 | **67%** | `█████████████░░░░░░░` |
| **Primary total** | **39** | **71%** | `██████████████░░░░░░` |

### Active training (epoch % only)

| Run | Epoch progress |
|-----|---------------:|
| 27t LR-IPS s0 | 15,004 / 25,000 (**60.0%**) ⏸️ |
| 27t LR-IPS s1 | 16,001 / 25,000 (**64.0%**) ⏸️ |
| 27t LR-IPS s2 | 10,001 / 25,000 (**40.0%**) ⏸️ |
| 27t PhyloGFN s0 | ✅ train + 1M eval + plots |
| 27t PhyloGFN s1 | ✅ train + 1M eval + plots |
| 27t PhyloGFN s2 | 30,999 / 32,000 (**96.9%**) |
| hg64 GRPO s2 | 2,135 / 10,000 (**21.4%**) |
| hg64 Count-IPS s2 | 1,626 / 10,000 (**16.3%**) |
| hg64 MIPS-GRPO s2 | 494 / 10,000 (**4.9%**) |
| *Avg across active jobs* | **63.9%** |

**Extras** (5t, 10t ablation, hg4096 — not in primary): **64%** `█████████████░░░░░░░`

<!-- PROGRESS_END -->

### Active training — rates & ETA (manual; refresh ~hourly)

| Run | GPU | Progress | Rate | ETA |
|-----|:---:|----------|-----:|:---:|
| hg64 GRPO s2 + 27t LR-IPS s0 | 0 | hg64 ~12% · 27t ~60% (resumed @15k) | ~520 / ~280 ep/h | hg64 Aug 28 ~14h · 27t Aug 29 |
| hg64 Count-IPS s2 + 27t LR-IPS s1 | 1 | hg64 ~9% · 27t ~66% (resumed @16k) | ~520 / ~280 ep/h | hg64 Aug 28 ~15h · 27t Aug 28 eve |
| hg64 MIPS s2 + 27t LR-IPS s2 | 2 | hg64 ~4% · 27t ~41% (resumed @10k) | ~480 / ~280 ep/h | hg64 Aug 28 ~17h · 27t Aug 29 |
| 27t PhyloGFN s2 | 3 | 29,999 / 32,000 (**93.7%**) | ~1,270 ep/h | **~22:20 tonight** |
| hg64 seed1 (all 4) | — | ✅ complete | — | — |
| 27t PhyloGFN s0/s1 | — | ✅ train + 1M eval + plots | — | — |

**VRAM co-schedule:** ~11–12 GB / 40 GB per GPU (hg64 ~0.5 GB + 27t ~11 GB). GPU 0 already at 100% util.

**Critical path:** PhyloGFN s2 (~1.5h) → parallel hg64 s2 + 27t LR-IPS finish over Aug 28–29.

---

## Hyper-Grid 64 (`hypergrid_64`)

**Config:** `final/configs/suites/hypergrid_64.json` · **Grid:** H=64, D=2 (64×64 → 4096 terminals) · **G=32** (batch / GRPO group size) · **Target:** 25k epochs · **Eval:** 50k terminal samples  
**Plots:** `final/runs/hypergrid_64/plots/epoch_24999/`

**Confirmed hyperparams (seed 0 runs):** `batch_size=32`, `steps_per_epoch=10`, `lr=0.001`, `clip_eps=0.2`, `checkpoint_every=200`, `eval_every=250`, `eval_samples=10000` (50k for final plots). See `resolved_config.json` in each run dir. *(Suite JSON lists 2000 epochs; actual headline runs were resumed/extended to 25k.)*

### Training grid

| Method | S0 | S1 | S2 | Notes |
|--------|:--:|:--:|:--:|-------|
| GRPO | ✅ | ✅ | 🔄 | S0: 1/4 modes, L1=1.97 · S1: 1/4 modes, L1=1.97 · S2: ~9% @ 10k |
| Count-IPS | ✅ | ✅ | 🔄 | S0: 1/4 modes, L1=1.84 · S1: 1/4 modes, L1=1.84 · S2: ~6% @ 10k |
| Learned-Reverse IPS | ✅ | ✅ | 🔄 | S0: **4/4 modes**, L1=0.26 · S1: **4/4 modes**, L1=0.52 · S2: ~2% @ 10k |
| GFlowNet TB | ✅ | ✅ | ⏳ | S0: **4/4 modes**, L1=0.27 · S1: **4/4 modes**, L1=0.32 |

### Final sampling stats — mean ± std across seeds (50k samples)

**Checkpoint:** S0 @ epoch 24999 · S1 @ epoch 9999 · **n = 2** (S2 pending — refresh when complete)

| Method | Modes S0 | Modes S1 | Modes μ ± σ | L1 S0 | L1 S1 | L1 μ ± σ | Peak mass μ ± σ | Unique terminals μ ± σ |
|--------|:--------:|:--------:|:-----------:|------:|------:|:--------:|:---------------:|:----------------------:|
| GRPO | 1 | 1 | 1.0 ± 0.0 | 1.966 | 1.968 | **1.967 ± 0.002** | 1.000 ± 0.000 | 14.5 ± 3.5 |
| Count-IPS | 1 | 1 | 1.0 ± 0.0 | 1.844 | 1.845 | **1.844 ± 0.001** | 1.000 ± 0.000 | 42.5 ± 0.7 |
| LR-IPS | 4 | 4 | 4.0 ± 0.0 | 0.262 | 0.323 | **0.293 ± 0.044** | 0.275 ± 0.007 | 4040 ± 23 |
| GFlowNet TB | 4 | 4 | 4.0 ± 0.0 | 0.270 | 0.324 | **0.297 ± 0.038** | 0.272 ± 0.008 | 4043 ± 0 |

*Sources: S0 → `final/runs/hypergrid_64/plots/epoch_24999/recovery_summary.json` (50k). S1 GRPO/Count-IPS → `hypergrid_64_seed1/plots/epoch_9999_grpo_countips/`; S1 TB → `…/epoch_9999_grpo_countips_tb/`; S1 LR-IPS → CPU 50k resample @ `checkpoint_epoch9999.pt` (GPU `plots/epoch_9999/` corrupt — do not use). σ = sample std, ddof=1 (n−1).*

**Refresh when S2 finishes:** add S2 column from trusted 50k `recovery_summary.json`, recompute μ ± σ over S0–S2 (n=3).

### Seed 0 runs (complete)

- [x] GRPO — `final/runs/hypergrid_64/grpo/20260824_184125_hypergrid_64_grpo`
- [x] Count-IPS — `final/runs/hypergrid_64/count_ips/20260824_191534_hypergrid_64_count_ips`
- [x] LR-IPS — `final/runs/hypergrid_64/learned_reverse_ips/20260824_212235_hypergrid_64_learned_reverse_ips`
- [x] TB — `final/runs/hypergrid_64/trajectory_balance/20260824_224747_hypergrid_64_trajectory_balance`

### Seed 1 (Aug 27)

**Config:** `final/configs/suites/hypergrid_64_seed1.json` · **10k epochs** · eval 10k@250 · seed 1

| Method | GPU | Train | Eval | Plots | Status | Log |
|--------|-----|:-----:|:----:|:-----:|--------|-----|
| GRPO | 1 | [x] | [x] | [x] | ✅ 1/4 modes, L1=1.97 | `final/runs/hypergrid_64_seed1/grpo/pipeline.log` |
| Count-IPS | 1 | [x] | [x] | [x] | ✅ 1/4 modes, L1=1.84 | `final/runs/hypergrid_64_seed1/count_ips/pipeline.log` |
| LR-IPS | 3 | [x] | [x] | [x] | ✅ **4/4 modes**, L1=0.52 | `final/runs/hypergrid_64_seed1/learned_reverse_ips/pipeline.log` |
| GFlowNet TB | 3 | [x] | [x] | [x] | ✅ **4/4 modes**, L1=0.32 | `final/runs/hypergrid_64_seed1/trajectory_balance/pipeline.log` |

- [x] Create suite config `hypergrid_64_seed1.json`
- [x] Launch 4 runs (seed 1, 10k epochs)
- [x] GRPO + Count-IPS partial suite plots @ epoch 9999 (50k samples) → `final/runs/hypergrid_64_seed1/plots/epoch_9999_grpo_countips/`
- [x] GRPO + Count-IPS + TB partial suite plots @ epoch 9999 (50k samples) → `final/runs/hypergrid_64_seed1/plots/epoch_9999_grpo_countips_tb/`
- [x] Full 4-method `plot_comparison` @ epoch 9999 → `final/runs/hypergrid_64_seed1/plots/epoch_9999/`

### Seed 2 (Aug 27 — in progress)

**Config:** `final/configs/suites/hypergrid_64_seed2.json` · **10k epochs** · eval 10k@250 · seed 2

| Method | GPU | Train | Eval | Plots | Status | Log |
|--------|-----|:-----:|:----:|:-----:|--------|-----|
| GRPO | 0 | [ ] | [ ] | [ ] | 🔄 ~870 / 10k (**8.7%**) | `final/runs/hypergrid_64_seed2/grpo/pipeline.log` |
| Count-IPS | 1 | [ ] | [ ] | [ ] | 🔄 ~620 / 10k (**6.2%**) | `final/runs/hypergrid_64_seed2/count_ips/pipeline.log` |
| MIPS-GRPO | 2 | [ ] | [ ] | [ ] | 🔄 ~235 / 10k (**2.4%**) | `final/runs/hypergrid_64_seed2/learned_reverse_ips/pipeline.log` |
| GFlowNet TB | — | [ ] | [ ] | [ ] | ⏳ not scheduled | — |

- [x] Create suite config `hypergrid_64_seed2.json`
- [x] Launch 3 runs (GRPO + Count-IPS + MIPS-GRPO on GPUs 0–2)
- [ ] Launch TB seed 2 when GPU available
- [ ] Suite plots @ epoch 9999 when seed 2 training finishes

### Plots & analysis

- [x] Suite comparison at epoch 24999 (`recovery_summary`, terminal distributions, L1/reward curves, modes vs samples)
- [x] Per-method GT vs sampled (`gt_vs_*.png`)
- [x] Per-method training diagnostics
- [x] In draft: H=64 table + figures (`draft1/tables/tab_hypergrid64.tex`, `draft1/figures/hypergrid/`)
- [x] Multi-seed mean ± std recovery table — **S0+S1 (n=2)** above; extend to n=3 when S2 + plots done
- [ ] Update `AUG25_COMPLETED_RUNS.md` §3 to reference `epoch_24999` (currently says `epoch_9999`)

---

## Hyper-Grid 8 (`hypergrid_8_b256`) — b256, 2k epochs ✅ COMPLETE

**3 methods × 3 seeds = 9 runs.** Batch 256, eval **5k samples every 200 epochs**, stopped @ **2k epochs** (Aug 26).

**Configs:** `hypergrid_8_b256_seed{0,1,2}.json`  
**Run dirs:** `final/runs/hypergrid_8_b256_seed{0,1,2}/{grpo,count_ips,learned_reverse_ips}/20260826_*`  
**Plots:** `final/runs/hypergrid_8_b256/plots/seed{0,1,2}/` (all generated)

| Method | S0 | S1 | S2 | Notes |
|--------|:--:|:--:|:--:|-------|
| GRPO | ✅ | ✅ | ✅ | 1/4 modes, L1=1.768 (all seeds) |
| Count-IPS | ✅ | ✅ | ✅ | **4/4 modes**, L1≈0.22–0.23 |
| MIPS-GRPO | ✅ | ✅ | ✅ | **4/4 modes**, L1≈0.049–0.062 |

### Final sampling stats — mean ± std across seeds (10k samples)

**Checkpoint:** all seeds @ epoch **1999** · **n = 3** (complete)

| Method | Modes S0 | Modes S1 | Modes S2 | Modes μ ± σ | L1 S0 | L1 S1 | L1 S2 | L1 μ ± σ | Peak mass μ ± σ | Unique terminals μ ± σ |
|--------|:--------:|:--------:|:--------:|:-----------:|------:|------:|------:|:--------:|:---------------:|:------------------------:|
| GRPO | 1 | 1 | 1 | 1.0 ± 0.0 | 1.768 | 1.768 | 1.768 | **1.768 ± 0.000** | 1.000 ± 0.000 | 1 ± 0 |
| Count-IPS | 4 | 4 | 4 | 4.0 ± 0.0 | 0.221 | 0.223 | 0.234 | **0.226 ± 0.007** | 0.532 ± 0.028 | 64 ± 0 |
| MIPS-GRPO | 4 | 4 | 4 | 4.0 ± 0.0 | 0.049 | 0.062 | 0.061 | **0.057 ± 0.007** | 0.465 ± 0.003 | 64 ± 0 |

*Sources: `final/runs/hypergrid_8_b256/plots/seed{0,1,2}/recovery_summary.json` (10k samples @ epoch 1999). σ = sample std, ddof=1 (n−1).*

**Key result:** With $G=256 > 64$ terminals, count IPS works (4/4 modes) but MIPS-GRPO still achieves ~4× lower $\ell_1$.

### Plots & analysis

- [x] Per-seed suite comparison (`recovery_summary`, terminal distributions, modes vs samples, training curves)
- [x] In draft: mean±std + per-seed tables (`draft1/tables/tab_hypergrid8_b256.tex`)
- [x] In draft: seed 0 figures (`draft1/figures/hypergrid_b256/`)

**GFlowNet TB:** not run in this suite (3 methods only).

---

## Phylogenetics — 27 taxa (`27taxa_noreplay_b4096_seed*`)

**Configs:** `final/configs/suites/27taxa_noreplay_b4096_seed{0,1,2}.json`  
**LR-IPS target:** 25k epochs · **PhyloGFN target:** 32k epochs · **Eval:** 1M samples (post-train)

### LR-IPS (3 seeds)

| Seed | Train | Eval | Plots | Progress | GPU |
|------|:-----:|:----:|:-----:|----------|-----|
| 0 | 🔄 | [ ] | [ ] | ~15k / 25k (**60%**) — co-sched w/ hg64 GRPO | 0 |
| 1 | 🔄 | [ ] | [ ] | ~16k / 25k (**66%**) — co-sched w/ hg64 Count-IPS | 1 |
| 2 | 🔄 | [ ] | [ ] | ~10k / 25k (**41%**) — co-sched w/ hg64 MIPS | 2 |

**Run dirs:** `final/runs/27taxa_noreplay_b4096_seed{0,1,2}/learned_reverse/20260825_*`

### PhyloGFN / GFlowNet TB (3 seeds)

| Seed | Train | Eval | Plots | Progress | GPU |
|------|:-----:|:----:|:-----:|----------|-----|
| 0 | [x] | [x] | [x] | ✅ train + 1M eval + plots | — |
| 1 | [x] | [x] | [x] | ✅ train + 1M eval + plots | — |
| 2 | 🔄 | [ ] | [ ] | ~30.0k / 32k (**93.7%**) | 3 |

**Run dirs:** `final/runs/27taxa_noreplay_b4096_seed{0,1,2}/phylgfn/`  
**Backend:** og_code (paper-faithful TB reproduction)

*Note (Aug 27 eve): **27t LR-IPS resumed on GPUs 0–2** alongside hg64 seed2 (co-scheduled; ~11 GB VRAM each). Resume checkpoints: s0 @ ep14999, s1 @ ep15999, s2 @ ep9999.*

### Other methods — planned / failed

| Method | Seed | Status | Notes |
|--------|------|--------|-------|
| GRPO | 0 | ❌ | OOM at epoch 0 (batch 4096); not restarted |
| GRPO | 1, 2 | ⏳ | Not scheduled |
| Count-IPS | 0, 1, 2 | ⏳ | Not in current seed configs |

- [ ] Decide whether to retry 27t GRPO with smaller batch (e.g. b1024 like LR-IPS)
- [ ] Post-train eval + plots for all 6 active runs when training completes
- [ ] Multi-seed aggregate Table 2 row for LR-IPS and PhyloGFN

### MLL validation (P1 Issue 4) — 🔸 partial

- [ ] Recover og_code 27t checkpoint (used in current Table 2) — **missing from disk**
- [x] Paper-faithful 27t early MLL eval — Δ −137 nats vs published at ~6% training
- [ ] Re-run MLL at convergence when PhyloGFN seed runs finish

---

## Analysis & paper deliverables

### P0 — Same-transform comparisons (Issue 1)

- [x] Table 2 with matched linear + log-log r (`table2.{tex,md,csv,json}`)
- [x] Figure 6 matched-transform PNGs (27t, 10t)
- [x] Draft bundle with tables + figures (`draft1/draft1.tex`, `draft1/figures/`, `draft1/tables/`)
- [x] Matched-transform caption + Results paragraph in `draft1/draft1.tex`

### P1 — q_φ ablation (Issue 3)

- [x] 10t uniform P_B run complete (train + eval + plots)
- [x] Ablation table in draft (`draft1/tables/tab_uniform_ablation.tex`)

### P2 — Fixed-benchmark eval (Issue 2) — ⏳ not started

- [ ] Run `eval_tree_logq_pearson.py` on frozen tree set — both methods, same trees
- [ ] Primary or appendix table with multi-sample estimator
- [ ] Demote 1M self-sampled scatter to "pathwise diagnostic" in text

### P3 — Writing & manifest

- [x] Metric A vs B draft text (pathwise vs fixed-tree)
- [x] Hypergrid H=64 figures + table in draft
- [x] Hypergrid H=8 b256 figures + tables in draft
- [x] Phylo setup / matched / extended / reverse-training tables in draft
- [ ] Update `manifest.json` to point 5t/10t at `final/runs/`
- [ ] Refresh Table 2 with completed 27t seed runs when available

---

## Extras

Supporting / smoke / ablation runs — not on the main experiment critical path.

### Hyper-Grid 4096 (`hypergrid_4096`)

**Config:** `final/configs/suites/hypergrid_4096.json` · **Target:** 100 epochs (smoke / pipeline validation only)

| Method | Train | Eval | Plots | Status |
|--------|:-----:|:----:|:-----:|--------|
| GRPO | [x] | [x] | [x] | ✅ 0/4 modes |
| Count-IPS | [x] | [x] | [x] | ✅ 0/4 modes |

**Notes:** Not comparable to 64-grid headline results. No multi-seed planned.

### Phylogenetics — 5 taxa (`5taxa_noreplay`)

**Config:** `final/configs/suites/5taxa_noreplay.json` · **Target:** 10k epochs · **Eval:** 1M samples · **Seed:** 0 only

| Method | Train | Eval | Plots | Status | Key result |
|--------|:-----:|:----:|:-----:|--------|------------|
| GRPO | [x] | [x] | [x] | ✅ | Mode collapse — 1 unique sig / 1M |
| Count-IPS | [x] | [x] | [x] | ✅ | 66k unique; log-log r=0.565 (empirical) |
| LR-IPS | [x] | [x] | [x] | ✅ | ESS 0.9998, log-log r=0.992, 951k unique |
| PhyloGFN (TB) | [x] | [x] | [x] | ✅ | log-log r=0.991, ESS **0.13**, 100k unique; 1M eval plots |

**Run dirs:** `final/runs/5taxa_noreplay/{grpo,count_ips,learned_reverse,phylgfn}/20260825_*`

- [x] LR-IPS sample plots regenerated (8 PNGs; pipeline plot fix applied)
- [x] PhyloGFN 1M reward-probability eval + plots
- [x] 5t panels in draft (`draft1/figures/phylo/5taxa_*`)
- [ ] Update Table 2 5t row to these `final/runs/` numbers (optional; legacy numbers in `table2.md`)

### Phylogenetics — 10 taxa

#### Uniform q_φ ablation (`10taxa_uniform_reverse_ablation`) — P1 Issue 3 ✅

**Config:** `final/configs/suites/10taxa_uniform_reverse_ablation.json` · **Target:** 10k epochs · **Seed:** 0

| Method | Train | Eval | Plots | Status |
|--------|:-----:|:----:|:-----:|--------|
| LR-IPS (uniform P_B) | [x] | [x] | [x] | ✅ |

**Run:** `final/runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_*`  
**Result:** ESS ~0.995 (same as fitted q_φ); linear r=0, log-log r=0.969

- [x] Ablation table in draft (`tab_uniform_ablation.tex`)

#### Full 10t comparison (`10taxa_noreplay`) — planned

**Config:** `final/configs/suites/10taxa_noreplay.json` · **Target:** 10k epochs · **No runs yet under `final/runs/`**

| Method | Train | Eval | Plots | Status |
|--------|:-----:|:----:|:-----:|--------|
| GRPO | [ ] | [ ] | [ ] | ⏳ |
| Count-IPS | [ ] | [ ] | [ ] | ⏳ |
| LR-IPS | [ ] | [ ] | [ ] | ⏳ |
| PhyloGFN | [ ] | [ ] | [ ] | ⏳ |

**Notes:** Fitted LR-IPS 10t baseline exists in legacy path (`grpo_experiments/learned_reverse_runs/20260803_*`); `final/runs/` reproduction not started.

- [ ] Optional: uniform q_φ ablation at 27t (after seed runs finish)

---

## GPU allocation (Aug 27 eve)

| GPU | Status | Jobs |
|-----|--------|------|
| 0 | 🔄 Busy | hg64 GRPO s2 + 27t LR-IPS s0 (~11 GB) |
| 1 | 🔄 Busy | hg64 Count-IPS s2 + 27t LR-IPS s1 (~11 GB) |
| 2 | 🔄 Busy | hg64 MIPS s2 + 27t LR-IPS s2 (~11 GB) |
| 3 | 🔄 Busy | 27t PhyloGFN s2 (~94%) |
| 4–7 | 🔄 Busy | External (qwenenv) |

**Queued:** hg64 TB seed2 when a GPU opens

---

## How to update this doc

1. When a run finishes training, check `[x]` under **Train** and note epoch / run dir.
2. After post-train eval + plots, check **Eval** and **Plots**; change status to ✅.
3. Update **Dashboard** counts and **Last updated** date.
4. Refresh **Overall progress** (epochs + weighted %):

```bash
.venv/bin/python final/paper/progress_summary.py --patch
```

5. For active runs, refresh inline progress % from `metrics.jsonl` or latest checkpoint:

```bash
# LR-IPS epoch
tail -1 final/runs/27taxa_noreplay_b4096_seed0/learned_reverse/*/metrics.jsonl | python3 -c "import sys,json; print(json.load(sys.stdin)['epoch'])"

# PhyloGFN epoch (max checkpoint)
ls final/runs/27taxa_noreplay_b4096_seed2/phylgfn/*/checkpoints/checkpoint_*.pt | tail -1

# Regenerate hypergrid plots (seed 0 @ 25k)
.venv/bin/python -m final.toy.plot_comparison --suite hypergrid_64 --all-methods --last-common-checkpoint

# hg64 seed1 — GRPO + Count-IPS only (done Aug 27)
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m final.toy.plot_comparison \
  --dataset final/datasets/hypergrid_64 \
  --grpo-run final/runs/hypergrid_64_seed1/grpo/20260826_224411_hypergrid_64_seed1_grpo \
  --ips-run final/runs/hypergrid_64_seed1/count_ips/20260826_224412_hypergrid_64_seed1_count_ips \
  --out-dir final/runs/hypergrid_64_seed1/plots/epoch_9999_grpo_countips \
  --num-samples 50000 --sample-device cuda:0

# hg64 seed1 — full 4-method (auto via watcher when LR-IPS + TB finish)
# final/runs/hypergrid_64_seed1_plot_watcher.py → plots/epoch_9999/
```

---

## Changelog

| Date | Change |
|------|--------|
| Aug 26, 2026 | Initial tracker created. Hypergrid S0 complete (25k, epoch_24999 plots). 27t seeds 0–2 active. |
| Aug 26, 2026 | Added Hyper-Grid 8 (H=8, G=256): 3 methods × 3 seeds = 9 runs. Confirmed H=64 batch_size=32. |
| Aug 26, 2026 | PhyloGFN seed2 paused on GPU 3; hypergrid_8 seed2 GRPO launched. Relaunch queue added. |
| Aug 26, 2026 | GPU cleanup: paused PhyloGFN S0–S2 + LR-IPS S1; moved hypergrid off GPU 0→3. 9× b256 runs on GPUs 1–3 only. |
| Aug 26, 2026 | Stop hypergrid @2k epochs; watcher auto-resumes 27t + queues plot_comparison per seed. |
| Aug 26, 2026 (eve) | **Hypergrid b256 complete** (9/9 train + plots). 27t jobs resumed (6 active). **draft1/** bundle: H=64 + H=8 b256 + phylo tables/figures. 5t suite fully complete incl. PhyloGFN eval. 10t uniform ablation complete. P0/P1 draft items largely done. |
| Aug 27, 2026 | **GPU 0 fix:** stopped 27t LR-IPS seed2 (was contending with seed0); launched `27taxa_lr_ips_seed2_gpu2_watcher.py` to relaunch on GPU 2 with `--resume-from` when PhyloGFN s0/s1 finish. Seed0 gets full GPU 0. |
| Aug 27, 2026 | **hg64 seed1 GRPO + Count-IPS complete** (train + eval + 50k-sample plots). LR-IPS/TB ~67–75%. 27t LR-IPS ~38–51%, PhyloGFN ~87.5%. Plot watcher queued for full seed1 comparison. EXPERIMENT_TODO refreshed. |
| Aug 27, 2026 | Added **Overall progress** section (66% primary) + `progress_summary.py --patch` auto-refresh. |
| Aug 27, 2026 (pm) | Progress refresh: PhyloGFN s0/s1 @94%, hg64 TB/LR-IPS ~87–78%, 27t LR-IPS ~49–56%. Added rates/ETA table. |
| Aug 27, 2026 16:10 | **67% primary.** PhyloGFN ~96%, hg64 TB ~92%, hg64 LR-IPS ~82%, 27t LR-IPS ~51–58%. hg64 TB finishes ~1.5h. |
| Aug 27, 2026 18:05 | **hg64 TB s1 complete** (4/4 modes, L1=0.32 @ 50k). 3-method seed1 plots → `plots/epoch_9999_grpo_countips_tb/`. PhyloGFN s0/s1 train done, 1M eval started. LR-IPS s2 resumed GPU 2. |
| Aug 27, 2026 20:37 | **hg64 seed2 launched** — GRPO/Count-IPS/MIPS on GPUs 0–2. 27t LR-IPS s0/s1/s2 paused. PhyloGFN s2 resumed GPU 3. |
| Aug 27, 2026 20:44 | **70% primary.** hg64 seed1 fully complete (LR-IPS 4/4 modes L1=0.52 + 4-method plots). PhyloGFN s0/s1 ✅. hg64 s2 ~2–9%. PhyloGFN s2 ~94%. |
| Aug 27, 2026 20:57 | **71% primary.** Co-scheduled 27t LR-IPS s0/s1/s2 on GPUs 0–2 alongside hg64 seed2 (~11 GB VRAM/GPU). All 3 resumed from checkpoint. |
