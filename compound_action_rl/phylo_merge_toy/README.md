# Merge (phylo) toy — a DAG-structured toy that behaves like the tree env

This toy exists to test algorithms (GRPO / IPS-GRPO / marginal-IPS / GFlowNet-style
credit) on the *same structural problem* the phylo tree env has, but small enough to
enumerate exactly. Unlike the `grid_4x4_varied_ips_v2` toy — where trajectory ↔ outcome
is a **bijection** — here the map is a **many-to-one DAG**, exactly like phylogenetics.

## Why the grid toy is not enough (and this one is)

| Property | grid v2 | phylo tree env | **merge toy (this)** |
|---|---|---|---|
| trajectory → outcome | bijection | many-to-one DAG | **many-to-one DAG** |
| outcome | (moves, colors) = the path | topology (signature) | rooted topology (signature) |
| non-uniform multiplicity m(x) | no (m≡1) | yes | **yes** |
| reward dynamic range | [0,1] gentle | ~0..57 nats, `π ∝ e^{log_score}` peaked | **configurable: linear / exp(β·score) / log-tilt** |
| `log π(τ)` magnitude | small | ±huge, `exp(−log π)` meaningless | **configurable via β; log-space SNIPS** |
| exact ground truth | yes | no (too big) | **yes (enumerated)** |

The three things your query calls out are reproduced here on purpose:

1. **DAG marginalisation.** `n=5` gives **180 ordered merge sequences → 105 rooted
   topologies**, multiplicity `m(x) ∈ {1,2,3}`. Trajectory-level exact IPS drives
   `π(τ) ∝ R(τ)`, whose marginal is the **biased** `π(x) ∝ m(x)·R(x)`. The uniform
   backward policy (`1/num_parents`, validated so `Σ_{τ→x} P_B(τ|x)=1`) lets
   `marginal` IPS recover the **unbiased** `π(x) ∝ R(x)`.
2. **Reward dynamic range.** `--reward-mode exp --beta 57` makes the posterior
   astronomically peaked (best topology `≈ e^{57}` more likely), mirroring phylo.
   `log_score` reproduces the near-flat "uniform tilt" your current config uses.
3. **Broken `log π(τ)` at scale.** All IPS weighting is in **log-space with
   log-sum-exp SNIPS**. `--naive-expspace` reproduces the overflow failure (finite at
   β≈57–400, non-finite by β≈800), while the default log-space path stays finite.

## Mapping to the phylo code

| phylo | merge toy |
|---|---|
| `PhylogenticTreeState.subtrees` | `MergeEnv._forest` (list of `Subtree`) |
| `tree_action` into `C(num_trees,2)` pairs | `MergeEnv.step(action)` over `pair_index_table` |
| `PhylogeneticTree.signature` (topology id) | `Subtree.canon` (canonical rooted Newick id) |
| `rollout_worker` `num_parents = has_children.sum(-1)` | `MergeEnv._num_parents` = #internal subtrees |
| `log_paths_pf` (sum of step log-probs) | `Episode.log_pf` |
| `log_paths_pb = -log(num_parents)` | `Episode.log_pb` |
| `scale_rewards_exact_ips` (SNIPS) | `MergeTrainer._group_advantages` (log-space) |
| `marginal_ips_grpo` (P_B correction) | `--propensity-mode marginal` |

The env is single-head over merges to match the referenced **tree-only** run
(`ONLY_TRAIN_TREE_MODEL: true`); add a second discrete head later for the
branch-length / edge analogue if needed.

## Files

- `merge_env.py` — `MergeEnv`, `Subtree`, canonicalisation, `num_parents`, backward log-prob.
- `catalog.py` — `RewardModel` + exact DAG enumeration (`build_catalog`, `validate_catalog`) and the two exact targets `target_marginal()` / `target_ips()`.
- `config.py` — `TrainConfig` + `REWARD_PROFILES`.
- `networks.py` — masked categorical `MergePolicyNet`.
- `trainer.py` — `MergeTrainer`: rollout, log-space IPS weights, PPO update.
- `eval_sampling.py` — R² of empirical `q̂(x)` against both exact targets.
- `plots.py` — sampling scatter (vs marginal & vs biased), `log q̂ vs log R`, training curves.
- `run_experiment.py` / `train_common.py` / `run_output.py` — CLI + run artifacts.

## Reward profiles

| profile | mode | β | meaning |
|---|---|---|---|
| `phylo_peaked` | exp | 57 | astronomically peaked posterior (phylo-like) |
| `phylo_log_tilt` | log_score | 57 | near-flat "uniform tilt" (reward = log_score) |
| `mild_peaked` | exp | 8 | moderately peaked, easier to learn |
| `gentle` | linear | 1 | ~[0,1] target like the grid toy |

## Running

```bash
cd compound_action_rl/phylo_merge_toy

# Trajectory-exact IPS -> converges to the BIASED target π ∝ m(x)R(x)
python run_experiment.py --reward-profile phylo_peaked --propensity-mode exact

# Marginal (backward-corrected) IPS -> recovers the UNBIASED target π ∝ R(x)
python run_experiment.py --reward-profile phylo_peaked --propensity-mode marginal

# Plain GRPO baseline
python run_experiment.py --reward-profile gentle --propensity-mode none

# Demonstrate the exp-space overflow failure
python run_experiment.py --reward-profile phylo_peaked --propensity-mode exact --beta 800 --naive-expspace
```

### The key diagnostic

Each run prints and plots R² of empirical sampling `q̂(x)` against **both** exact
targets. The signature of the DAG bias:

- `exact` → higher **R²(biased ∝mR)** than **R²(marginal ∝R)**
- `marginal` → higher **R²(marginal ∝R)** (bias corrected)

`signature_qhat_vs_logreward.png` is the merge-toy analogue of your
`signature_qhat_vs_loglikelihood` plot: points colored by `log m(x)`; perfect
marginal sampling lies on the slope-1 line, and multiplicity bias shows up as
high-`m` topologies sitting above it.

## Extending to bigger models

- Increase `--n-leaves` (6 → 2700 histories / 945 topologies; 7 → 56700 / 10395) to
  stress the DAG bias and inverse-propensity variance; enumeration stays feasible to ~8.
- The `MergeEnv` / `MergeTrainer` interface intentionally mirrors the phylo
  `rollout_worker` fields (`log_pf`, `log_pb`, `signature`), so an algorithm that
  works here should port to `grpo_experiments/ips_grpo` and `marginal_ips_grpo`
  with the same weighting math.
