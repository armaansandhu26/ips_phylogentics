# Tree Edge IPS-GRPO v2

Phylo port of the toy `compound_action_rl/grid_4x4_varied_ips_v2` idea:

- exact / tempered inverse propensity weights from joint trajectory log-probs
- SNIPS + ESS targeting
- shared trajectory advantage for tree and edge
- split PPO with independent tree/edge ratios
- edge-rep detach via `EDGE_REP_GRAD_ALPHA=0`

## Recommended first sweep (real IPS test)

Defaults are now phylo-scale. Prefer **ESS targeting** over raw exact IPS:

```bash
python -m grpo_experiments.tree_edge_ips_v2.train \
  --run-name v2_ess05_g4096 \
  --device cuda:0 \
  --num-updates 10000 \
  --group-size 4096 \
  --num-groups 1 \
  --lr 1e-4 \
  --clip-eps 0.2 \
  --entropy-coef 1e-3 \
  --reward-mode log_reward \
  --ips-target-ess-fraction 0.5 \
  --detach-edge-rep \
  --eval-every 500 \
  --eval-episodes 2048 \
  --checkpoint-every 1000 \
  --print-every 10 \
  --cpu-threads 2
```

By default, runs land in `grpo_experiments/tree_edge_ips_v2/runs/<timestamp>_<run_name>/`. Override with `--output`.

Useful A/B variants:

| Run | Flag change | What it tests |
|---|---|---|
| ESS 0.5 (default) | `--ips-target-ess-fraction 0.5` | Adaptive tempered IPS |
| Fixed temp | `--ips-target-ess-fraction 0 --ips-weight-temperature 0.3` | Fixed tempering |
| Legacy exact | `--ips-target-ess-fraction 0 --ips-weight-temperature 1.0` | Often inert (uniform after cap) |
| Count IPS | `--propensity-mode count` | Outcome-count propensity |
| No detach | `--no-detach-edge-rep` | Edge grads into tree reps |

After training, use the existing full-model pipeline for large sampling plots:

```bash
python grpo_experiments/full_model/run_pipeline.py \
  --skip-training \
  --run-dir <this_run_dir> \
  --sample-trees 100000
```

## Smoke test

```bash
python -m grpo_experiments.tree_edge_ips_v2.train \
  --run-name sanity --num-updates 1 --num-groups 1 --group-size 2 \
  --eval-every 0 --checkpoint-every 0 --ips-target-ess-fraction 0.5
```

## Progress metrics to watch (`metrics.jsonl`)

### Reward quality
- `mean_log_score`, `p50_log_score`, `p95_log_score`, `max_log_score`
- `eval_mean_log_score`, `eval_p95_log_score`, `eval_max_log_score` (every `--eval-every`)

Rising mean with collapsing max/p95 diversity often means mode collapse.

### IPS health (critical)
- `ips_ess_fraction_mean` — should sit near `--ips-target-ess-fraction` (e.g. ~0.5)
- `ips_active_mean` — ≈0 means SNIPS is nearly uniform (IPS inert); larger means real reweighting
- `ips_solved_temperature_mean` / `ips_weight_temperature_mean` — solved beta; shrinks as policy sharpens
- `ips_snips_weight_min/max/std` — weight spread
- `ips_legacy_absolute_cap_mean` — 1.0 means you are on the inert absolute-cap path
- `ips_log_prop_min/max` — spread of `-log π(τ)`

### Policy update
- `tree_policy_loss`, `edge_policy_loss`
- `tree_ratio_mean`, `edge_ratio_mean`, clipped fractions
- `grad_norm`, `entropy_bonus`, `mean_entropy`
- `mean_log_pf_tree`, `mean_log_pf_edge`
- `mean_delta_log_pf_tree/edge` — should be ~0 on-policy with 1 epoch (sanity)

### Diversity
- `batch_unique_outcomes`, `batch_unique_topologies`
- `batch_duplicate_fraction`, `batch_duplicate_topology_fraction`
- `eval_unique_topologies`, `eval_unique_signatures`

Topology unique count crashing toward 1–3 while scores rise is the classic collapse signature.

## Intentional limitations

- Best-tree replay is not mixed into v2 updates.
- Toy edge input `[rep; one_hot(tree_action)]` and aux heads are not enabled on the phylo transformer.
- `edge_credit=counterfactual` is config-only (trajectory credit is used).
