# Direction/step DAG — Count-IPS and GFlowNet comparison

This folder contains one small experiment:

```text
state (x, y)
  -> model 1 chooses RIGHT or UP
  -> model 2 chooses step count 1, 2, or 3
  -> terminal normal reward
```

The default DAG has `budget=3` and `max_step=3`:

- 10 coordinate states;
- 18 trajectories;
- 4 terminal outcomes;
- terminal multiplicities `(4, 5, 5, 4)`.

For arbitrary budgets, `find_default_terminal_states(budget)` returns the full
`x + y == budget` frontier and `reward_per_terminal_state(budget)` returns its
state-to-reward mapping. The budget-3 reward profile is linearly interpolated
over normalized terminal x-coordinate, preserving `(1.0, 0.8, 0.2, 0.05)`
exactly at budget 3.

All runners accept scalable environment arguments:

```bash
python compound_action_rl/dag_toy_dataset/run_count_ips.py \
  --budget 5 \
  --max-step 3
```

To override the interpolated defaults, provide exactly `budget + 1` rewards in
increasing terminal x-coordinate order:

```bash
python compound_action_rl/dag_toy_dataset/run_count_ips.py \
  --budget 5 \
  --terminal-rewards 1.0 0.9 0.7 0.4 0.15 0.05
```

## Rewards and ideal 10k sampling

All trajectories reaching the same outcome receive the same normal reward.

| outcome | trajectories | reward | ideal probability | ideal count |
|---|---:|---:|---:|---:|
| `(0,3)` | 4 | `1.00` | `48.78%` | `4,878` |
| `(1,2)` | 5 | `0.80` | `39.02%` | `3,902` |
| `(2,1)` | 5 | `0.20` | `9.76%` | `976` |
| `(3,0)` | 4 | `0.05` | `2.44%` | `244` |

Ideal sampling is outcome-level and reward proportional:

```text
p*(x) = R(x) / sum_x R(x)
```

## Count-based IPS

For each sampled group of size `G`, [count_ips.py](count_ips.py) computes:

```text
p_hat(x_i)       = count(x_i) / G
scaled_reward_i  = reward(x_i) / p_hat(x_i)
advantage_i      = normalize(scaled_reward_i)
```

The advantages feed a masked, token-averaged PPO clipped loss matching the
simple objective in `grpo_experiments/core/loss.py`. Each token is one compound
action with joint log-probability:

```text
log pi(direction | state) + log pi(step | state_rep, direction)
```

There is no backward policy, exact trajectory IPS, replay, or log-reward.

## Exact forward-path-probability IPS

[`exact_probability_ips.py`](exact_probability_ips.py) is the direct
exact-propensity version of Count-IPS. The rollout policy already records each
compound action log-probability, so it computes:

```text
log P_F(tau) = sum_t [log pi(direction_t | state_t)
                      + log pi(step_t | state_t, direction_t)]
scaled(tau)  = R(x(tau)) / P_F(tau)
```

No within-group count is used in the denominator. The saved probabilities are
from the behavior policy that actually sampled the path, before the PPO update.

This changes the target semantics: inverse weighting by a complete path
probability gives every path its own reward weight. If terminal `x` has `m(x)`
paths, its aggregate weight is proportional to `m(x) * R(x)`, not just `R(x)`.
Use `trajectory_ips.py` when exact multiplicities are available, or
`backward_kl_ppo.py` for a locally normalized backward correction, if the
desired terminal marginal is reward proportional.

Run the exact path-probability experiment with:

```bash
python compound_action_rl/dag_toy_dataset/exact_probability_ips.py \
  --budget 32 --group-size 128 --num-updates 500
```

## Backward-corrected exact IPS

[`backward_corrected_ips.py`](backward_corrected_ips.py) retains raw IPS while
removing the unknown path-multiplicity factor:

```text
scaled(tau) = R(x(tau)) * P_B(tau | x) / P_F(tau)
```

`P_F(tau)` is recorded exactly by the rollout behavior policy. `P_B` is fixed
and uniform over valid parent edges at every reverse step. Because this locally
normalized backward policy assigns total probability one across all reverse
paths from each terminal, the aggregate target mass for terminal `x` is
`R(x)` regardless of how many forward paths reach it. This is raw importance
weighting, not the log-ratio KL score used by `backward_kl_ppo.py` and not the
squared trajectory-balance objective.

```bash
python compound_action_rl/dag_toy_dataset/backward_corrected_ips.py \
  --budget 32 --group-size 128 --num-updates 500
```

## Trajectory-balance GFlowNet baseline

`gflownet.py` provides a deliberately small GFlowNet baseline modeled on the
core trajectory-balance setup in PhyloGFN:

```text
loss(tau) = (log Z + log P_F(tau) - log R(x) - log P_B(tau | x))^2
```

It uses the exact same environment, terminal rewards, hierarchical direction /
step network, batch sizes, and evaluation code as Count-IPS. The only added
model parameter is a scalar `log Z`. `P_B` is fixed and uniform over the valid
incoming compound edges at each state, so no outcome counts or trajectory
enumeration are needed.

Run matched Count-IPS and GFlowNet experiments from the repository root:

```bash
python compound_action_rl/dag_toy_dataset/run_count_ips.py \
  --budget 32 --group-size 128 --num-updates 500 --seed 0

python compound_action_rl/dag_toy_dataset/run_gflownet.py \
  --budget 32 --group-size 128 --num-updates 500 --seed 0
```

Both runners write the same main comparison artifacts: `history.json`,
`summary.json`, `checkpoint.pt`, `training_curves.png`,
`sampling_counts.png`, and `trajectory_sampling.png`. Compare
`summary.json -> final_sampling -> tv_reward_target` as the primary outcome
metric. GFlowNet runs are stored under `data/gflownet_runs/`.

## Annealed epsilon/temperature exploration

`epsilon_greedy_count_ips.py` is an exploration-focused version of the same
Count-IPS trainer. During training, both the direction and conditional step
policies sample from:

```text
q(a | s) = (1 - epsilon) * softmax(logits / temperature)
           + epsilon * Uniform(valid actions)
```

PPO old and new log-probabilities both use `q`, so the rollout and optimization
policies remain consistent. Evaluation uses the original learned policy with
no epsilon mixture or temperature scaling. By default epsilon anneals from
`0.30` to `0.02` and temperature from `2.0` to `1.0` on a cosine schedule.

```bash
python compound_action_rl/dag_toy_dataset/epsilon_greedy_count_ips.py \
  --budget 32 \
  --group-size 512 \
  --num-updates 4000 \
  --epsilon-start 0.30 \
  --epsilon-end 0.02 \
  --temperature-start 2.0 \
  --temperature-end 1.0 \
  --anneal-updates 3000 \
  --schedule cosine
```

The runner adds `exploration_schedule.png` and logs
`exploration_epsilon`/`exploration_temperature` in `history.json`.

## Known-multiplicity trajectory IPS oracle

`trajectory_ips.py` is a separate ideal-behavior experiment. It uses terminal
multiplicities computed by dynamic programming and targets complete trajectories with:

```text
score(tau) = R(x(tau)) / (m(x(tau)) * p_hat(tau))
```

This retains `p(x) proportional to R(x)` while targeting a uniform conditional
distribution over the known trajectories reaching each terminal. Its PPO ratio
uses the sum of action log-probabilities along the complete path, rather than a
token average.

Run it from the repository root with:

```bash
CUDA_VISIBLE_DEVICES=0 python compound_action_rl/dag_toy_dataset/trajectory_ips.py
```

The default group size is 512 because the rarest target trajectories have only
about 0.61% probability and trajectory-frequency estimates are noisier than
terminal-frequency estimates.

## Unknown-multiplicity trajectory balancing

`unknown_m_trajectory_ips.py` hides `m(x)` from the training rule. It retains
terminal count-IPS and adds a centered within-terminal path-surprisal advantage:

```text
A_terminal = normalize(R(x) / p_hat(x))
A_path     = normalize within terminals(-log p_hat(tau | x) - baseline(x))
A          = normalize(A_terminal + lambda * A_path)
```

The default schedule uses terminal-only training through update 100 and ramps
`lambda` from 0 to 1 over updates 101–200. No multiplicity is used by training;
conditional trajectory frequencies use an EMA count
table with default decay `0.95`, allowing rare terminals to pool path-frequency
evidence across recent batches without knowing how many paths exist.

```bash
CUDA_VISIBLE_DEVICES=0 python \
  compound_action_rl/dag_toy_dataset/unknown_m_trajectory_ips.py
```

## Backward-reference trajectory-KL PPO

`backward_kl_ppo.py` removes within-batch output counting entirely.  It trains
the forward policy with full-trajectory PPO on:

```text
score(tau) = beta * log R(x) + log P_B(tau | x) - log P_F(tau)
```

`P_B` is fixed and locally uniform over the valid incoming edges of each
visited DAG state.  These local probabilities define a normalized distribution
over every reverse path ending at a terminal, without enumerating those paths.
At `beta=1`, summing the optimal trajectory distribution over all paths to a
terminal gives `P_F(x) proportional to R(x)`, irrespective of how many paths
reach that terminal.

Run the small-batch experiment with:

```bash
python compound_action_rl/dag_toy_dataset/backward_kl_ppo.py \
  --budget 16 \
  --group-size 16 \
  --num-updates 1000
```

The reward exponent is annealed from `--reward-beta-start 0.25` to
`--reward-beta-end 1.0` by default.  Ordinary entropy regularization is disabled
because the `-log P_F(tau)` term already supplies the entropy in the exact KL
objective.

## Run

```bash
cd compound_action_rl/dag_toy_dataset

python run_count_ips.py \
  --num-updates 500 \
  --group-size 128 \
  --eval-every 100 \
  --final-samples 10000
```

The runner selects CUDA automatically when it is available. To select physical
GPU 0 explicitly from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python compound_action_rl/dag_toy_dataset/run_count_ips.py
```

Training groups are generated with vectorized rollouts: all active episodes in
a group advance together through batched direction and step policy calls on the
selected device. Evaluation uses the same sampler in chunks, avoiding one GPU
launch and CPU synchronization per action of each individual episode.

Runs write `config.json`, `history.json`, `checkpoint.pt`, `summary.json`,
`training_curves.png`, `sampling_counts.png`, and `trajectory_sampling.png`
under `data/count_ips_runs/`. Intermediate checkpoints are saved every 500
updates by default under the run's `checkpoints/` directory. Use
`--checkpoint-every N` to select a different interval. Automatically generated
run-directory names include the budget, for example
`20260718_160000_b16_gs512_seed0`.

### Logged diagnostics

- Sampling quality: per-outcome counts/probabilities, R², L1 distance, total-
  variation distance, and maximum probability error versus the ideal reward target.
- Count IPS: min/mean/max `p_hat`, unique outcomes, outcome count range, scaled-
  reward mean/std, IPS effective sample size, and ESS fraction.
- Optimization: gradient norm, parameter norm, policy entropy, PPO importance-
  ratio range, and clipping fraction.
- Behavior: mean reward, mean trajectory length, and terminal coverage.
- Trajectories: count and global probability for all 18 paths, per-terminal path
  coverage, conditional path probabilities, normalized conditional entropy,
  effective number of paths, and the largest within-terminal path share.

Total-variation distance is the primary convergence metric; it is zero only when
the learned terminal distribution matches the reward-proportional target. The
per-outcome probability curves reveal which outcome causes any remaining error.
The sampling scatter title and `final_sampling.r2_reward_target` report the R²
of the learned probabilities regressed on the ideal reward-proportional
probabilities; values closer to one indicate a tighter linear fit.

`trajectory_sampling.png` shows every action sequence (for example `R1-U2`),
its global sample count, its probability conditional on the terminal it reaches,
and path-diversity curves over training. Uniform allocation within a terminal is
shown only as a diagnostic reference: the terminal-reward objective itself does
not require trajectories leading to the same outcome to be equally likely.

## Test

```bash
python test_count_ips.py
```

## Minimal file map

- `dag_env.py` — monotone DAG and normal terminal rewards.
- `networks.py` — direction model and conditional step model.
- `config.py` — environment and training settings.
- `count_ips.py` — count-IPS advantages and PPO training algorithm.
- `gflownet.py` — minimal trajectory-balance GFlowNet training algorithm.
- `run_gflownet.py` — matched GFlowNet CLI, artifacts, and plots.
- `trajectory_ips.py` — separate known-multiplicity trajectory-IPS oracle and CLI.
- `unknown_m_trajectory_ips.py` — terminal IPS plus hidden-multiplicity conditional path entropy.
- `backward_kl_ppo.py` — count-free trajectory-KL PPO with a normalized reverse-path reference.
- `evaluate_trajectory_coverage.py` — sample a checkpoint and report/plot terminal sampling and exact path coverage.
- `run_count_ips.py` — CLI, artifacts, evaluation, and final count plot.
- `test_count_ips.py` — focused unit and training smoke tests.
- `test_gflownet.py` — trajectory-balance and checkpoint smoke tests.
- `dag_budget3_step3.png` — visual map of states, edges, and multiplicities.
- `ideal_reward_sampling_10000.png` — ideal reward-versus-count scatter for 10k samples.
