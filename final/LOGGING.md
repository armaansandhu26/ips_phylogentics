# Final logging spec — implementation status

## Per training step (`metrics_detailed.jsonl`)

Activated automatically when run dir is under `final/runs/`.

| Field group | Status |
|-------------|--------|
| Common (step, wall_clock, log_R stats, traj_len, pf_entropy, grad_norm, lr, loss) | **Implemented** via `FinalRunLogger` |
| PPO (policy_loss, clip_frac, approx_kl, ratio, advantage, group stats) | **Partial** — wired for learned-reverse forward pass; GRPO/count_ips hooks pending |
| Learned-reverse (log_w hist, psis_khat, ips_ess, reverse_nll) | **Implemented** in `learned_reverse_ips/runner.py` |
| Count IPS (p_hat stats, n_terminals ge 2) | **Pending** — needs ips_grpo runner hook |
| GFlowNet (tb_loss, log_Z, replay) | **Pending** — needs og_code train hook |

Legacy `metrics.jsonl` is still written unchanged for backward compatibility.

## Eval dumps (50k @ step 0, 500…5k, 1000…, final)

| Component | Status |
|-----------|--------|
| Eval schedule (`should_eval_dump`) | **Implemented** — marks `eval_dumps/eval_step_XXXXXX.pending` |
| Trajectory NPZ with backward K=32 arrays | **TODO** — `final/logging/eval_dump.py` |
| Checkpoints at eval steps | **TODO** |
| Final 1M sharded dump + Newick 10k subsample | **TODO** — use existing sample scripts + extend |

## meta.json

| Field | Status |
|-------|--------|
| seed, git commit, config, gpu model, log_w bin edges | **Implemented** |
| param counts, gpu hours | **Partial** — filled at finalize, param counts pending |

## Precomputed environment

| Artifact | Path | Status |
|----------|------|--------|
| 5-taxa terminal enumeration | `final/precomputed/5taxa_noreplay/` | **Implemented** |
| Exact m(x) per topology | `mx_exact.json` | **Implemented** (180 paths → 105 topologies) |

## Pre-launch verification

```bash
python -m final.verify.verify_mx          # 5 taxa: 180 trajectories, 105 topologies
python -m final.verify.verify_topology_hash  # deterministic IDs + Newick RF=0
```

Runs automatically before `python -m final run_suite` (unless `--skip-preflight`).

## Next implementation priority

1. `eval_dump.py` — trajectory-level NPZ at scheduled steps
2. Hook `FinalRunLogger` into `grpo_experiments/runner.py` and `ips_grpo/runner.py`
3. Hook GFlowNet TB metrics into `og_code/train.py`
4. Final 1M sharded export with backward K=64 on 10k terminals
