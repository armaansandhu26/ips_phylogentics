# P1 results — uniform q_phi ablation + MLL check

## 1. MLL check (Issue 4a)

### og_code 27-taxa baseline (paper comparison run)

- **Status:** checkpoint **not on disk** (`20260806_150835_phylgfn_logreward_27taxa_g1024_noreplay_shift12000`).
- Comparison metrics and Figure 6 PNGs remain in `grpo_experiments/comparisons/27taxa/gflownet/`.
- **Action if recovered:**  
  `python grpo_experiments/scripts/eval_gflownet_mll.py --run-dir <og_code_run> --device cuda:0`

### paper-faithful PhyloGFN 27-taxa (partial training)

| Field | Value |
|-------|------:|
| Run | `phylogfn_paper/experiments/full_model/20260812_145519_phylgfn_paper_27taxa_g1024_noreplay` |
| Checkpoint | `checkpoint_001999.pt` (epoch 1999 / 32000 planned) |
| MLL mean ± std | **−7246.27 ± 9.46** (3 replicates, 1024 trajs) |
| PhyloGFN paper ref | −7108.95 |
| MrBayes ref | −7108.42 |
| Δ vs paper | −137.3 nats |

**Interpretation:** This run is **early / incomplete** (~6% of planned epochs). MLL is worse than published −7109, which is expected mid-training. It does **not** yet validate the og_code baseline used in Table 2. Recover or retrain og_code 27t checkpoint for the headline MLL comparison.

Full JSON: `phylogfn_paper/.../mll_eval.json`

Regenerate:
```bash
python grpo_experiments/scripts/eval_gflownet_mll.py \
  --run-dir phylogfn_paper/experiments/full_model/20260812_145519_phylgfn_paper_27taxa_g1024_noreplay \
  --checkpoint checkpoints/checkpoint_001999.pt
```

---

## 2. Uniform vs fitted q_phi ablation (Issue 3) — **DONE**

### Code

- `--reverse-policy uniform` + `--reverse-train-epochs 0` in `learned_reverse_ips/`
- Uniform weight uses frozen `sum(log_paths_pb)` (= GFlowNet uniform \(P_B(\tau|x)\) on the forward path)
- Fitted baseline: existing 10t run  
  `grpo_experiments/learned_reverse_runs/20260803_124837_learned_reverse_10taxa_mlp_shifted_linear_b4096_learned_reverse_ips_grpo`
- Uniform ablation run (completed Aug 26):  
  `final/runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_learned_reverse_10taxa_uniform_reverse_ablation_learned_reverse_ips_grpo`

### Suite

`final/configs/suites/10taxa_uniform_reverse_ablation.json`

### Results table (10 taxa, 1M eval samples)

| Condition | reverse policy | linear r | log-log r | ESS | log-weight std | unique sig / 1M |
|-----------|----------------|---------:|----------:|----:|---------------:|----------------:|
| Fitted q_phi (baseline) | MLP MLE | **0.976** | 0.835 | 0.995 | 0.209 | 999,986 |
| Uniform P_B (ablation) | frozen uniform | 0.0 | **0.969** | 0.995 | 0.217 | 999,974 |

Metrics from each run's `plots/mlp_shifted_linear_reference_1000k/comparison_metrics.json`.

### What the ablation tests

MIPS-GRPO uses \(\hat P(x) \propto R(x)\, q(\tau|x) / P_F(\tau)\). The ablation holds the **forward policy and PPO update fixed** and swaps only the reverse term:

- **Fitted \(q_\phi\):** MLP trained by MLE on sampled paths (8 reverse epochs / step).
- **Uniform \(P_B\):** frozen GFlowNet uniform backward policy from `log_paths_pb`; no reverse MLE.

**Question:** Is learning \(q_\phi\) necessary, or does IPS normalization with uniform \(P_B\) suffice?

### Interpretation

1. **ESS is essentially unchanged (~99.5%).** Fitting \(q_\phi\) does not materially reduce IPS weight variance at 10 taxa. Uniform \(P_B\) already gives tractable IPS weights.

2. **Reverse learning mainly affects calibration, not IPS tractability.** Similar terminal diversity (topology TV ≈ 0.024 both; >999k unique signatures).

3. **Linear vs log-log split:**
   - Fitted \(q_\phi\) → strong linear P vs R (\(r = 0.98\)), weaker log-log (\(r = 0.84\)).
   - Uniform \(P_B\) → strong log-log P vs R (\(r = 0.97\)), collapsed linear (\(r = 0\)). Partition estimate differs sharply (log Z ≈ +94 vs −4585), likely explaining linear-scale mismatch.

4. **Takeaway:** Uniform backward policy is a strong baseline for IPS efficiency at 10 taxa. Learning \(q_\phi\) is still valuable for **linear-scale probability–reward calibration** (Table 2 headline), but the ablation does **not** support reverse MLE as necessary to fix degenerate IPS weights.

5. **Caveats:** Different run dates; 10 taxa only; 27-taxa ablation not yet run.

### Plots

See `final/paper/AUG25_COMPLETED_RUNS.md` §2 (training curves + full sampling panel).

Smoke test: `learned_reverse_ips/experiments/20260825_234833_smoke_uniform_5t_learned_reverse_ips_grpo`

### Paper text (draft)

> To isolate the effect of fitting \(q_\phi\), we hold the forward MIPS-GRPO policy and PPO update fixed and compare frozen uniform \(P_B(\tau|x)\) (GFlowNet-style backward policy on the sampled path) against MLE-fitted \(q_\phi\). At 10 taxa, both conditions achieve comparable IPS effective sample size (\(\mathrm{ESS}/N \approx 0.995\)) and similar terminal diversity (\(>999{,}000\) unique signatures per million samples), indicating that reverse MLE is not required for IPS tractability at this scale. Fitted \(q_\phi\) yields stronger linear probability–reward correlation (\(r = 0.98\) vs \(0.0\)), while uniform \(P_B\) yields stronger log-log alignment (\(r = 0.97\) vs \(0.84\)). We conclude that reverse-policy learning primarily calibrates implied terminal probabilities to the evaluation scale rather than reducing IPS weight variance.

### Remaining actions

- [ ] Add uniform-\(P_B\) row to Table 2 / ablation sub-table.
- [ ] Optional: repeat at 27 taxa after seed runs complete.
