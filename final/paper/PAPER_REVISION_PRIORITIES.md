# Paper revision priorities

Feedback on `research_paper_v1.pdf` (Aug 2026). Ordered by **review damage** first, then by **effort vs payoff**.

**Last updated:** Aug 26, 2026 — reflects completed Aug 24–25 batch runs under `final/runs/`.

---

## Findings at a glance

| Issue | Status | Headline finding |
|-------|--------|------------------|
| **1** Same-transform comparisons | **Done** | At 27t, MIPS-GRPO \(r = 0.977\) on **both** linear and log-log scales; GFlowNet \(r \approx 0\) on both. Mixed-scale Figure 6 was misleading but conclusion unchanged. |
| **2** Circular / asymmetric eval | Open | Self-sampled pathwise ESS: MIPS 0.999 vs GFlowNet 0.847 at 27t. Fixed-tree multi-sample eval still needed. |
| **3** Role of fitted \(q_\phi\) | **Done (10t)** | Uniform \(P_B\) matches fitted \(q_\phi\) on ESS (~99.5%); fitted \(q_\phi\) wins linear calibration, uniform wins log-log. Reverse MLE calibrates scale, not IPS tractability. |
| **4** GFlowNet baseline validity | Partial | og_code 27t checkpoint missing; paper-faithful 27t at epoch 1999 gives MLL −7246 vs published −7109 (early training). |
| **5taxa sanity** | **Done** | LR-IPS best diversity + ESS; GRPO mode-collapses (1 sig / 1M); PhyloGFN strong log-log \(r\) but ESS 0.13. |
| **Hypergrid 64** | **Done** | LR-IPS and TB recover 4/4 modes; GRPO and Count-IPS find 1/4 only — supports mode-coverage story. |
| **27t reproduction** | In progress | LR-IPS seeds 0–2 at 19–25%; PhyloGFN seeds at 44–76%. seed0 GRPO OOM failed. |

Full plots and metrics: [`AUG25_COMPLETED_RUNS.md`](AUG25_COMPLETED_RUNS.md).

---

## Priority order

### P0 — Fix before any resubmit (no new compute)

**1. Same-transform comparisons (Issue 1)** — **DONE**

- **Problem:** Figure 6 compares MIPS-GRPO linear \(r\) vs GFlowNet log-log \(r\); headline is not apples-to-apples.
- **Fix applied:** Table 2 and Figure 6 now report linear and log-log \(r\) for **both** methods under matched transforms.
- **Outputs:** `final/paper/table2.{tex,md,csv,json}`, `figure6_27taxa_matched_transform.png`, `figure6_10taxa_matched_transform.png`
- **Regenerate:** see [`README.md`](README.md)

#### What we found (Table 2, Aug 2026)

| Taxa | Method | Linear \(r\) | Log-log \(r\) | ESS | Unique sig. / 1M |
|-----:|--------|------------:|-------------:|----:|-----------------:|
| 5 | GFlowNet | 0.982 | — | — | 960,850 |
| 5 | MIPS-GRPO | 0.994 | 0.994 | 1.000 | 951,175 |
| 10 | GFlowNet | 0.881 | 0.698 | 0.977 | 1,000,000 |
| 10 | MIPS-GRPO | 0.976 | 0.835 | 0.995 | 999,986 |
| 27 | GFlowNet | 0.002 | 0.024 | 0.847 | 1,000,000 |
| 27 | MIPS-GRPO | 0.977 | 0.977 | 0.999 | 999,987 |

**Interpretation for revision:**

- The **matched-transform fix removes the easiest reviewer attack** — we no longer compare incompatible scales.
- At **27 taxa the conclusion is unchanged**: MIPS-GRPO achieves near-perfect correlation on both scales; GFlowNet remains near zero.
- At **10 taxa**, log-log \(r\) is lower for MIPS-GRPO (0.84) than linear (0.98) — worth noting that log-log is the harder panel at moderate taxa counts.
- **5t GFlowNet log-log \(r\) is missing** from archived NPZ; linear only in table (known gap).

**Suggested paper text (Issue 1):**

> We report Pearson correlation between pathwise implied terminal probability and terminal reward under **matched transforms** for both methods: linear \(P(x)\) vs \(R(x)\) and log-log \(\log P(x)\) vs \(\log R(x)\). At 27 taxa, MIPS-GRPO achieves \(r = 0.977\) on both scales, while GFlowNet remains near zero (\(r = 0.002\) linear, \(r = 0.024\) log-log). A prior figure mixed transforms across methods; the corrected comparison does not change the headline conclusion.

**Remaining actions:**

- [x] Table 2 with four 27t numbers
- [x] Figure 6 matched-transform layout
- [ ] Paste revised caption + Results paragraph into draft
- [ ] Optional: regenerate 5t GFlowNet log-log panel if NPZ recovered

---

### P1 — Highest-value new experiments

**2. Controlled \(q_\phi\) ablation (Issue 3)** — **DONE**

- **Code:** `--reverse-policy uniform --reverse-train-epochs 0` in `learned_reverse_ips/`
- **Suite:** `final/configs/suites/10taxa_uniform_reverse_ablation.json`
- **Uniform run:** `final/runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_*`
- **Fitted baseline:** `grpo_experiments/learned_reverse_runs/20260803_124837_learned_reverse_10taxa_*`
- **Details:** [`P1_RESULTS.md`](P1_RESULTS.md) §2, [`AUG25_COMPLETED_RUNS.md`](AUG25_COMPLETED_RUNS.md) §2

#### What the ablation tests

\[
\hat P(x) \propto \frac{R(x)\, q(\tau \mid x)}{P_F(\tau)}
\]

Forward policy and PPO update are **fixed**; only the reverse term changes:

| Condition | Reverse term | Reverse training |
|-----------|--------------|------------------|
| **Fitted \(q_\phi\)** | MLP \(q_\phi(\tau \mid x)\), MLE on paths | 8 epochs / step |
| **Uniform \(P_B\)** | Frozen uniform backward from `log_paths_pb` | None |

#### Results (10 taxa, 1M eval)

| Metric | Fitted \(q_\phi\) | Uniform \(P_B\) |
|--------|:-----------------:|:---------------:|
| IPS ESS fraction | 0.995 | 0.995 |
| log-weight std | 0.209 | 0.217 |
| **Linear** P vs R | **0.976** | 0.0 |
| **Log-log** P vs R | 0.835 | **0.969** |
| Unique signatures / 1M | 999,986 | 999,974 |
| Topology TV | 0.024 | 0.024 |

#### Interpretation (Issue 3)

1. **ESS unchanged (~99.5%).** Fitting \(q_\phi\) does not reduce IPS weight variance at 10 taxa.
2. **Reverse learning = calibration, not tractability.** Sampling diversity is identical; only probability–reward alignment on a chosen scale differs.
3. **Linear vs log-log split** is the key nuance: fitted \(q_\phi\) for Table 2 linear panels; uniform \(P_B\) is competitive on log-log.
4. **Revise claims:** Do not say reverse MLE is needed to "fix degenerate IPS weights." Say it **calibrates implied probabilities** to the evaluation transform.

**Suggested paper text (Issue 3):** see [`P1_RESULTS.md`](P1_RESULTS.md) §2.

**Remaining actions:**

- [ ] Add uniform-\(P_B\) row to Table 2 or appendix ablation table (10t)
- [ ] Optional: repeat at 27 taxa when seed runs finish
- [ ] Clarify linear \(r = 0\) for uniform ≠ sampling failure (ESS and log-log panel)

---

**3. MLL check on 27-taxa GFlowNet baseline (Issue 4, part A)** — **PARTIAL**

- **og_code 27t checkpoint:** **missing from disk** — cannot validate headline Table 2 GFlowNet run via MLL.
- **paper-faithful 27t (epoch 1999 / 32000):** MLL = **−7246.3 ± 9.5** vs published −7108.95 (Δ −137 nats).
- **Script:** `grpo_experiments/scripts/eval_gflownet_mll.py`
- **Details:** [`P1_RESULTS.md`](P1_RESULTS.md) §1

#### What we found

- Partial paper-faithful training is **worse than published MLL**, as expected at ~6% of planned epochs.
- This **does not validate** the og_code baseline used in Table 2 / Figure 6.
- Reviewers may ask whether GFlowNet baseline is fair; we need either recovered og_code checkpoint or completed paper-faithful 27t run.

**Remaining actions:**

- [ ] Recover og_code 27t checkpoint or retrain
- [ ] Re-run MLL at convergence for paper-faithful PhyloGFN
- [ ] Report both og_code and paper-faithful baselines explicitly in text (don't conflate)

---

### P1b — New findings from Aug 25 batch (supporting evidence)

These runs were not explicit reviewer issues but strengthen (or constrain) revision narrative.

**8. Five-taxa four-method comparison** — **DONE**

Suite: `5taxa_noreplay` — 10k epochs, 1M eval. Runs: `final/runs/5taxa_noreplay/20260825_*`.

| Method | Train log reward | ESS | Log-log \(r\) | Unique sig / 1M | Finding |
|--------|----------------:|----:|-------------:|----------------:|---------|
| GRPO | 5.89 | — | — | **1** | **Mode collapse** — single signature dominates |
| Count-IPS | 5.91 | — | 0.565 (empirical) | 66,268 | Partial coverage; weak freq fit |
| LR-IPS | 5.58 | **0.9998** | **0.992** | 951,180 | Best IPS + diversity + calibration |
| PhyloGFN (TB) | — | 0.129 | **0.991** | 100,508 | Strong log-log \(r\) but **low ESS** (heavy weights) |

**Takeaways for paper:**

- GRPO alone is insufficient even at 5 taxa (collapse) — supports need for IPS normalization.
- LR-IPS combines high ESS with high correlation — validates method design on small instance.
- PhyloGFN can achieve high log-log \(r\) but with much lower ESS (0.13 vs 1.0) — supports Metric A vs B distinction (Issue 2).
- New `final/runs/` numbers are reproducible via `python -m final pipeline`; consider updating Table 2 5t row to these runs.

**9. Hyper-Grid 64 four-method benchmark** — **DONE**

25k epochs. Source: `final/runs/hypergrid_64/plots/epoch_9999/recovery_summary.json`.

| Method | Mode recovery | Modes / 4 | L1 to GT |
|--------|:-------------:|:---------:|---------:|
| GRPO | 25% | 1 | 1.97 |
| Count-IPS | 25% | 1 | 1.84 |
| LR-IPS | **100%** | **4** | **0.29** |
| GFlowNet TB | **100%** | **4** | 0.36 |

**Takeaways:**

- On toy multi-modal benchmark, **only LR-IPS and TB recover all modes**; GRPO-family without proper normalization finds one mode.
- Supports narrative that IPS + learned reverse helps **mode coverage**, not just correlation metrics on phylogeny.
- Good appendix figure: `final/runs/hypergrid_64/plots/epoch_9999/recovery_summary.png`.

**10. Hyper-Grid 4096 smoke** — **DONE** (100 epochs only)

Both GRPO and Count-IPS: 0/4 modes. Not headline-worthy; validates pipeline at scale only.

---

### P2 — Strengthen evaluation protocol

**4. Multi-sample estimator on fixed benchmark (Issue 2)** — **NOT STARTED**

- **Problem:** Pathwise \(P_F/q\) on self-samples is circular and estimator-asymmetric (ESS 0.999 vs 0.847 at 27t; 5t PhyloGFN ESS 0.13 vs LR-IPS 0.9998).
- **Evidence we now have:** 5t PhyloGFN shows high \(r\) (0.99) can coexist with terrible ESS (0.13) — exactly the reviewer concern.
- **Fix:** PhyloGFN-style eval on frozen tree set (`eval_tree_logq_pearson.py`). Both methods on **same** trees.
- **Deliverable:** Primary or appendix table; demote 1M self-sampled scatter to "pathwise diagnostic."

**5. Paper-faithful PhyloGFN 27-taxa baseline (Issue 4, part B)** — **IN PROGRESS**

- Active runs: `final/runs/27taxa_noreplay_b4096_seed{0,1,2}/phylgfn/` (~44–76% of 32k epochs, Aug 26).
- **Problem:** Table 2 GFlowNet baseline is og_code (shifted/clamped), not published PhyloGFN.
- **When done:** Compare paper-faithful TB to og_code and to MIPS-GRPO on same 27t suite.

---

### P3 — Paper text / clarity (parallel to above)

**6. Separate metrics in writing (Issues 2 & 4)** — **DRAFT READY**

State explicitly in Methods/Results:

- **Metric A (pathwise diagnostic):** Self-sampled 1M trajectories; Pearson \(r\) between implied terminal probability and reward on signature support. Report ESS alongside \(r\).
- **Metric B (PhyloGFN protocol):** Fixed-tree multi-sample \(\log \hat P(x)\) vs \(\log L(x)\); primary for cross-method fairness.

**Key sentences to add (from findings):**

> High pathwise correlation does not imply efficient estimation: at 5 taxa, PhyloGFN achieves log-log \(r = 0.99\) but IPS ESS fraction 0.13, whereas MIPS-GRPO achieves \(r = 0.99\) with ESS 0.9998.

> GFlowNet's low pathwise \(r\) at 27 taxa (\(r \approx 0.02\)) coexists with moderate ESS (0.85), indicating poor probability–reward calibration rather than weight degeneracy alone.

> Do **not** attribute GFlowNet's low \(r\) mainly to "bad \(P_B\)" when ESS \(\approx 0.85\).

**7. Table 2 + Figure 6 refresh** — **PARTIAL**

- [x] Matched-transform Table 2 generated (`table2.md`)
- [x] Figure 6 27t and 10t PNGs
- [x] 5t four-way runs complete under `final/runs/5taxa_noreplay/`
- [ ] Add uniform-\(P_B\) ablation row (10t)
- [ ] Update manifest.json to point 5t/10t at `final/runs/` where appropriate
- [ ] Consolidate appendix with hypergrid recovery figures

---

## Summary table

| Priority | Item | Issue # | Status | Key finding |
|----------|------|---------|--------|-------------|
| P0 | Same-transform Table 2 / Figure 6 | 1 | **Done** | 27t conclusion unchanged under fair comparison |
| P1 | Uniform vs fitted \(q_\phi\) (10t) | 3 | **Done** | ESS same; fitted wins linear, uniform wins log-log |
| P1 | MLL on 27t GFlowNet | 4 | **Partial** | og_code missing; paper 27t early MLL −7246 |
| P1b | 5t four-method suite | — | **Done** | GRPO collapses; LR-IPS best on all metrics |
| P1b | Hypergrid 64 | — | **Done** | LR-IPS + TB 4/4 modes; GRPO/IPS 1/4 |
| P2 | Fixed-benchmark multi-sample eval | 2 | Not started | 5t PhyloGFN ESS evidence motivates this |
| P2 | Paper-faithful PhyloGFN 27t | 4 | **Running** | ~44–76% training (Aug 26) |
| P3 | Metric definitions + narrative | 2, 4 | Draft ready | See §6 above |
| P3 | Table 2 / appendix refresh | 1, 3 | Partial | Add ablation row + hypergrid appendix |

---

## Existing artifacts (27 taxa — legacy comparison)

From `grpo_experiments/comparisons/27taxa/` (used in current Table 2):

| Method | Linear \(r\) | Log-log \(r\) | ESS |
|--------|-------------|---------------|-----|
| MIPS-GRPO (learned reverse) | 0.977 | 0.977 | 0.999 |
| GFlowNet (`og_code`) | 0.002 | 0.024 | 0.847 |

Will be superseded by `final/runs/27taxa_noreplay_b4096_seed*` when complete.

---

## Run status (Aug 26, 2026)

Full compiled results: [`AUG25_COMPLETED_RUNS.md`](AUG25_COMPLETED_RUNS.md)

| Suite | Method | Status | Progress / notes |
|-------|--------|--------|------------------|
| `5taxa_noreplay` | GRPO, Count-IPS, LR-IPS, PhyloGFN | **Done** | 1M eval + plots |
| `10taxa_uniform_reverse_ablation` | LR-IPS (uniform \(P_B\)) | **Done** | q\_φ ablation |
| `hypergrid_64` | GRPO, Count-IPS, LR-IPS, TB | **Done** | LR-IPS/TB 4/4 modes |
| `hypergrid_4096` | GRPO, Count-IPS | **Done** | 100-epoch smoke |
| `27taxa_*_seed0` | LR-IPS | Running | ~6.3k / 25k (25%) |
| `27taxa_*_seed1` | LR-IPS | Running | ~5.5k / 25k (22%) |
| `27taxa_*_seed2` | LR-IPS | Running | ~4.7k / 25k (19%) |
| `27taxa_*_seed0,1` | PhyloGFN | Running | ~14.2k / 32k (44%) |
| `27taxa_*_seed2` | PhyloGFN | Running | ~24.2k / 32k (76%) |
| `27taxa_*_seed0` | GRPO | **Failed** | OOM epoch 0 |

---

## Infrastructure fixes (Aug 25–26)

These unblocked post-train evaluation for completed runs:

1. **`learned_reverse_ips/post_train.py`** — curves-only mode no longer requires 10k samples file when `--skip-sampling-plots` is set.
2. **`phylogfn_paper/scripts/eval_reward_probability.py`** — fixed `build_gfn(cfg, env, device, ddp=False)` for 5t PhyloGFN eval.

---

## Suggested execution sequence

1. ~~P0: matched-transform Table 2 + Figure 6~~
2. ~~P1: uniform \(q_\phi\) 10t ablation~~
3. ~~P1b: 5t suite + hypergrid 64~~
4. **Now:** paste P0/P1/P3 draft text into paper; add uniform ablation row to Table 2
5. P1: recover og_code 27t checkpoint or MLL at convergence
6. Wait for 27t LR-IPS + PhyloGFN runs → update Table 2 with reproducible `final/runs/` paths
7. P2: 27t frozen benchmark multi-sample eval (both methods, same trees)
8. Optional: 27t uniform \(q_\phi\) ablation

---

## Related paths

| What | Where |
|------|-------|
| This document | `final/paper/PAPER_REVISION_PRIORITIES.md` |
| Completed runs + all plots | `final/paper/AUG25_COMPLETED_RUNS.md` |
| P1 detailed metrics | `final/paper/P1_RESULTS.md` |
| Table 2 / Figure 6 | `final/paper/table2.*`, `figure6_*.png` |
| Run outputs | `final/runs/` |
| Regenerate Table 2 | `grpo_experiments/scripts/build_paper_table2.py` |
| Regenerate Figure 6 | `grpo_experiments/scripts/plot_matched_transform_figure.py` |
| Multi-sample eval (P2) | `grpo_experiments/scripts/eval_tree_logq_pearson.py` |
