# Marginal (backward-corrected) exact IPS-GRPO

Self-contained experiment that makes RL sampling more reward-proportional by
fixing the **trajectory-multiplicity bias** in exact inverse-propensity scaling.
Nothing outside this folder is modified — to revert, delete
`grpo_experiments/marginal_ips_grpo/`.

## The problem this fixes

In the phylo env a terminal tree `x` is built by a sequence of pairwise joins,
and **many different join-orderings (trajectories) produce the same tree**. The
policy's marginal probability of a tree is therefore a sum over all orderings:

```
P(x) = sum_{tau -> x} P_F(tau)
```

The existing exact IPS weight uses a single sampled ordering,
`exp(-log P_F(tau))`, which estimates `1/P(tau)`, **not** `1/P(x)`. Because the
number of orderings differs across trees, this weight is a biased propensity and
the IPS fixed point is not `q(x) ∝ R(x)`. (Count-based IPS avoids the bias in
principle but degenerates at signature granularity, where almost every sample is
unique — so within-group counts carry no signal.)

## The change

The env already emits a uniform backward policy `log P_B(tau|x)`
(`RolloutWorker` -> `log_paths_pb`, equal to `-log N(x)`). We feed

```
log P_F(tau) - log P_B(tau|x)
```

into the exact weight, so it becomes

```
weight = exp(-(log P_F(tau) - log P_B(tau|x)))  ~  P_B(tau|x) / P_F(tau)
```

At the IPS fixed point `log P_F(tau) - log P_B(tau|x) = log R(x) - log Z` is
constant across all orderings of a given tree, so the marginal satisfies
`P_F(x) ∝ R(x)`. This is ordinary importance-weighting / reward shaping — **not**
a GFlowNet loss. It operates on the full trajectory (tree + edge log-probs), so
it targets **signature-level** proportionality (topology *and* branch lengths).

Only one line of behaviour differs from `grpo_experiments/ips_grpo`
(see `MARGINAL CORRECTION` in `runner.py`); everything else (trainer, PPO loss,
rollout, replay, logging, checkpointing) is imported unchanged.

## Recommended defaults (baked into `train.py`, all overridable)

| Setting | Value | Why |
|---|---|---|
| `ips_propensity_mode` | `exact` | backward correction only applies to the exact weight |
| `backward_correction` | `True` | the fix above |
| `advantage_reward_mode` | `exp_linear` | IPS must scale a **linear** reward, not a log-reward |
| `ips_target_ess_fraction` | `0.5` | auto-temper the weight each batch to hold ESS (tames heavy tails) |
| `grpo_entropy_coef` | `0.01` | entropy pressure so the fixed point is proportional, not a mode |
| `outcome_level` | `signature` | matches the final eval; only used for logging in exact mode |

## Run

Full-model 1M-step run (mirrors the existing replay run, on-policy + data-loader replay):

```bash
.venv/bin/python -m grpo_experiments.marginal_ips_grpo.train \
    --cfg src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml \
    --dataset dataset/benchmark_datasets/DS1_reduced.pickle \
    --output grpo_experiments/full_model \
    --run-name marginal_g4096_1m_full_replay \
    --on-policy-batch-size 3277 --replay-batch-size 819 --replay-buffer-size 4096 \
    --epochs 10000 --steps-per-epoch 1 \
    --device cuda:0 --checkpoint-every 1000
```

Then reuse the existing pipeline for sampling + plots:

```bash
.venv/bin/python grpo_experiments/full_model/run_pipeline.py --skip-training \
    --run-dir grpo_experiments/full_model/<timestamp>_marginal_g4096_1m_full_replay_marginal_ips_grpo \
    --sample-trees 1000000 --group-by both
```

## Ablation (isolates the effect of the correction)

Run the identical setup with the correction off — this recovers plain exact IPS
(`exp(-log P_F(tau))`) and should reproduce the collapsed scatter:

```bash
.venv/bin/python -m grpo_experiments.marginal_ips_grpo.train ... --no-backward-correction
```

Compare `signature_qhat_vs_loglikelihood` (and the topology-level plot) between
the two runs, and against the PhyloGFN (TB) baseline.

## Reverting

Delete this folder. No other files were changed.
```bash
rm -r grpo_experiments/marginal_ips_grpo
```
