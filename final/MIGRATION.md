# Migration plan: legacy code → `final/`

This document describes how to move from the current fragmented layout to a clean paper-ready repo.

## Target state

```
phylogfn/
├── final/                    # orchestration + configs + runs + results  ← NEW (keep)
├── src/                      # shared phylogenetic env / models          ← KEEP
├── dataset/                  # DS1 pickles                               ← KEEP
├── grpo_experiments/
│   ├── core/                 # GRPO trainer                              ← KEEP (minimal)
│   ├── ips_grpo/             # count IPS                                 ← KEEP (minimal)
│   └── scripts/              # sampling + plotting only                  ← KEEP (subset)
├── learned_reverse_ips/      # learned-reverse package                   ← KEEP
├── og_code/                  # shifted PhyloGFN                          ← KEEP (minimal)
├── phylogfn_paper/           # optional paper-faithful PhyloGFN          ← KEEP (optional)
└── .venv/
```

## Phase 1 — Run final suites (current)

1. Run all four methods for each suite you need in the paper:
   ```bash
   python -m final run_suite --suite 5taxa_noreplay
   python -m final run_suite --suite 10taxa_noreplay
   python -m final run_suite --suite 27taxa_noreplay
   python -m final run_suite --suite 27taxa_replay   # if replay row needed
   ```
2. Verify `final/results/<suite>/comparison_table.csv` looks correct.
3. Copy key plots from `final/runs/...` into your paper figures directory.

## Phase 2 — Archive legacy experiment outputs

Safe to archive once final runs are verified:

| Path | Size (approx) | Notes |
|------|---------------|-------|
| `grpo_experiments/learned_reverse_runs/` | ~3.6 GB | historical runs |
| `phylogfn_paper/experiments/` | ~1 GB | crashed paper run |
| `grpo_experiments/comparisons/` | ~100 MB | old comparison grids (keep if replot scripts needed) |
| `grpo_experiments/runs/` | varies | old default output root |

```bash
tar -czf ~/phylogfn_legacy_runs_$(date +%Y%m%d).tar.gz \
  grpo_experiments/learned_reverse_runs \
  grpo_experiments/runs \
  phylogfn_paper/experiments
```

## Phase 3 — Archive legacy code (after final runs complete)

### Keep (required by `final/`)

```
src/
grpo_experiments/core/
grpo_experiments/ips_grpo/
grpo_experiments/scripts/sample_ppo_full_diagnostics.py
grpo_experiments/scripts/sample_learned_reverse_full_diagnostics.py
grpo_experiments/scripts/plot_full_checkpoint_vs_reward_reference.py
grpo_experiments/scripts/plot_learned_reverse_training_curves.py
grpo_experiments/scripts/reward_probability_plot_reference.py
grpo_experiments/scripts/eval_og_gflownet_reward_probability.py
grpo_experiments/utils.py
grpo_experiments/metrics.py
grpo_experiments/resume.py
grpo_experiments/eval_utils.py
grpo_experiments/config.py
grpo_experiments/train.py
grpo_experiments/runner.py
grpo_experiments/learned_reverse_ips_grpo.py   # shim for sampling imports
grpo_experiments/phylo_learned_reverse_policy.py
learned_reverse_ips/
og_code/train.py
og_code/src/
og_code/sample_trees.py
phylogfn_paper/   # optional
final/
dataset/
```

### Archive / delete candidates

```
grpo_experiments/hybrid_grpo/
grpo_experiments/hybrid_ips_grpo/
grpo_experiments/marginal_ips_grpo/
grpo_experiments/tree_edge_ips_v2/
grpo_experiments/scripts/plot_5taxa_*
grpo_experiments/scripts/plot_10taxa_*
grpo_experiments/scripts/run_*_e2e.py      # superseded by final/
grpo_experiments/comparisons/            # after copying needed figures
rgflow/                                  # unrelated RGFN fork
```

```bash
tar -czf ~/phylogfn_legacy_code_$(date +%Y%m%d).tar.gz \
  grpo_experiments/hybrid_grpo \
  grpo_experiments/hybrid_ips_grpo \
  grpo_experiments/marginal_ips_grpo \
  grpo_experiments/tree_edge_ips_v2 \
  rgflow
```

## Phase 4 — Optional hardening before deletion

1. **PhyloGFN checkpoint pruning:** patch `og_code/train.py` or `phylogfn_paper/train.py` to save every 1000 epochs.
2. **Move `final/runs/` to `/data0/`** if root disk is tight:
   ```bash
   ln -s /data0/armaan/phylogfn_runs final/runs
   ```
3. **Pin suite configs** in git — they are the paper's experiment registry.
4. **Add CI smoke test:** `python -m final pipeline --suite 5taxa_noreplay --method grpo --skip-train --run-dir <fixture>`.

## Phase 5 — Minimal repo (end state)

After archiving, the repo shrinks to ~15 GB (.venv + .git + final runs):

- All paper numbers come from `final/results/*/comparison_table.csv`
- All paper figures come from `final/runs/*/ */plots/`
- One command reproduces a full comparison row: `python -m final pipeline --suite ... --method ...`

## Checklist before deleting anything

- [ ] All four methods complete for each paper suite
- [ ] `comparison_table.csv` matches expected ordering / magnitudes
- [ ] Key plots copied or referenced from `final/runs/`
- [ ] Legacy runs tarball verified (`tar -tzf ... | head`)
- [ ] Legacy code tarball verified
- [ ] No active tmux jobs pointing at old paths
