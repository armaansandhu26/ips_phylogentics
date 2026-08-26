# Experiment progress tracker

Living checklist for paper/revision experiments. Update this file as runs finish.

**Last updated:** Aug 26, 2026 (evening) — hypergrid b256 complete; draft1 bundle; 27t jobs resumed  
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
| Hyper-Grid 64 | 4 | 4 | 4 | 0 |
| Hyper-Grid 8 b256 | 9 | 0 | 0 | 0 |
| Phylogenetics 27t | 0 | 6 | 5 | 1 |
| Analysis / paper | 8 | 1 | 3 | 0 |

### Extras

| Section | Done | Active | Planned | Failed |
|---------|-----:|-------:|--------:|-------:|
| Hyper-Grid 4096 | 2 | 0 | 0 | 0 |
| Phylogenetics 5t | 4 | 0 | 0 | 0 |
| Phylogenetics 10t | 1 | 0 | 4 | 0 |

---

## Hyper-Grid 64 (`hypergrid_64`)

**Config:** `final/configs/suites/hypergrid_64.json` · **Grid:** H=64, D=2 (64×64 → 4096 terminals) · **G=32** (batch / GRPO group size) · **Target:** 25k epochs · **Eval:** 50k terminal samples  
**Plots:** `final/runs/hypergrid_64/plots/epoch_24999/`

**Confirmed hyperparams (seed 0 runs):** `batch_size=32`, `steps_per_epoch=10`, `lr=0.001`, `clip_eps=0.2`, `checkpoint_every=200`, `eval_every=250`, `eval_samples=10000` (50k for final plots). See `resolved_config.json` in each run dir. *(Suite JSON lists 2000 epochs; actual headline runs were resumed/extended to 25k.)*

### Training grid

| Method | S0 | S1 | S2 | Notes |
|--------|:--:|:--:|:--:|-------|
| GRPO | ✅ | ⏳ | ⏳ | S0: 1/4 modes, L1=1.97 |
| Count-IPS | ✅ | ⏳ | ⏳ | S0: 1/4 modes, L1=1.84 |
| Learned-Reverse IPS | ✅ | ⏳ | ⏳ | S0: **4/4 modes**, L1=0.26 |
| GFlowNet TB | ✅ | ⏳ | ⏳ | S0: **4/4 modes**, L1=0.27 |

### Seed 0 runs (complete)

- [x] GRPO — `final/runs/hypergrid_64/grpo/20260824_184125_hypergrid_64_grpo`
- [x] Count-IPS — `final/runs/hypergrid_64/count_ips/20260824_191534_hypergrid_64_count_ips`
- [x] LR-IPS — `final/runs/hypergrid_64/learned_reverse_ips/20260824_212235_hypergrid_64_learned_reverse_ips`
- [x] TB — `final/runs/hypergrid_64/trajectory_balance/20260824_224747_hypergrid_64_trajectory_balance`

### Seed 1 (active — Aug 26 evening)

**Config:** `final/configs/suites/hypergrid_64_seed1.json` · **10k epochs** · eval 10k@250 · seed 1

| Method | GPU | Status | Log |
|--------|-----|--------|-----|
| GRPO | 1 | 🔄 | `final/runs/hypergrid_64_seed1/grpo/pipeline.log` |
| Count-IPS | 1 | 🔄 | `final/runs/hypergrid_64_seed1/count_ips/pipeline.log` |
| LR-IPS | 3 | 🔄 | `final/runs/hypergrid_64_seed1/learned_reverse_ips/pipeline.log` |
| GFlowNet TB | 3 | 🔄 | `final/runs/hypergrid_64_seed1/trajectory_balance/pipeline.log` |

- [x] Create suite config `hypergrid_64_seed1.json`
- [x] Launch 4 runs (seed 1, 10k epochs)
- [ ] `plot_comparison` at epoch 9999 with 50k eval samples when complete

### Seed 1 & 2 (planned — seed 2 not started)

- [ ] Create suite config `hypergrid_64_seed2.json`
- [ ] Launch 4 runs (seed 2)

### Plots & analysis

- [x] Suite comparison at epoch 24999 (`recovery_summary`, terminal distributions, L1/reward curves, modes vs samples)
- [x] Per-method GT vs sampled (`gt_vs_*.png`)
- [x] Per-method training diagnostics
- [x] In draft: H=64 table + figures (`draft1/tables/tab_hypergrid64.tex`, `draft1/figures/hypergrid/`)
- [ ] Multi-seed mean ± std recovery table (after S1/S2 complete)
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
| 0 | 🔄 | [ ] | [ ] | ~7.3k / 25k (**29%**) | 0 |
| 1 | 🔄 | [ ] | [ ] | ~7.0k / 25k (**28%**) | 1 |
| 2 | 🔄 | [ ] | [ ] | ~5.8k / 25k (**23%**) | 0 |

**Run dirs:** `final/runs/27taxa_noreplay_b4096_seed{0,1,2}/learned_reverse/20260825_*`

### PhyloGFN / GFlowNet TB (3 seeds)

| Seed | Train | Eval | Plots | Progress | GPU |
|------|:-----:|:----:|:-----:|----------|-----|
| 0 | 🔄 | [ ] | [ ] | ~17.0k / 32k (**53%**) | 2 |
| 1 | 🔄 | [ ] | [ ] | ~17.0k / 32k (**53%**) | 2 |
| 2 | ⏸️ | [ ] | [ ] | ~28k / 32k (**88%**) — paused for hg64 GPU3 | 3 |

**Run dirs:** `final/runs/27taxa_noreplay_b4096_seed{0,1,2}/phylgfn/`  
**Backend:** og_code (paper-faithful TB reproduction)

*Note: PhyloGFN seed2 + LR-IPS seed1 paused (Aug 26 eve) so hg64 seed1 gets full GPUs 1 & 3; watchers auto-resume.*

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

## GPU allocation (Aug 26 evening)

| GPU | Status | Jobs |
|-----|--------|------|
| 0 | 🔄 Busy | 27t LR-IPS seed0 + seed2 |
| 1 | 🔄 Busy | **27t LR-IPS seed1 ⏸️ paused** + hg64 seed1 (grpo, count_ips) |
| 2 | 🔄 Busy | PhyloGFN seed0 + seed1 |
| 3 | 🔄 Busy | **PhyloGFN seed2 ⏸️ paused** + hg64 seed1 (LR-IPS, TB) |
| 4–7 | 🔄 Busy | External (qwenenv) |

**Hypergrid b256:** finished. No phylogfn jobs paused.

---

## How to update this doc

1. When a run finishes training, check `[x]` under **Train** and note epoch / run dir.
2. After post-train eval + plots, check **Eval** and **Plots**; change status to ✅.
3. Update **Dashboard** counts and **Last updated** date.
4. For active runs, refresh progress % from `metrics.jsonl` or latest checkpoint:

```bash
# LR-IPS epoch
tail -1 final/runs/27taxa_noreplay_b4096_seed0/learned_reverse/*/metrics.jsonl | python3 -c "import sys,json; print(json.load(sys.stdin)['epoch'])"

# PhyloGFN epoch (max checkpoint)
ls final/runs/27taxa_noreplay_b4096_seed2/phylgfn/*/checkpoints/checkpoint_*.pt | tail -1

# Regenerate hypergrid plots
.venv/bin/python -m final.toy.plot_comparison --suite hypergrid_64 --all-methods --last-common-checkpoint
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
