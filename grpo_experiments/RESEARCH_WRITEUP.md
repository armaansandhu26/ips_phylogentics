# Learned-Reverse IPS-GRPO for Reward-Proportional Phylogenetic Sampling

**Working draft — August 2026**

> Detailed LaTeX version (derivation + algorithm + toy DAG PoC + PhyloGFN results): [`RESEARCH_WRITEUP_DETAILED.tex`](RESEARCH_WRITEUP_DETAILED.tex)

---

## Abstract

We study how to train a phylogenetic tree-building policy whose sampling distribution matches a terminal reward \(R(x)\). The same forward policy and environment are trained under four objectives: **GRPO**, **IPS-GRPO**, **GFlowNet (PhyloGFN)**, and our proposed **Learned-Reverse IPS-GRPO**. The core difficulty is *trajectory multiplicity*: many merge orderings reach the same terminal tree, so naive policy-gradient or count-based importance weights fail at signature granularity. Our method assigns each trajectory an importance weight \(R(x)\, q_\phi(\tau \mid x) / P_F(\tau)\), where \(q_\phi\) is a learned backward policy fit by maximum likelihood. On 5-, 10-, and 27-taxa benchmarks, Learned-Reverse IPS-GRPO matches or exceeds GFlowNet on reward-proportionality metrics, while GRPO and IPS-GRPO collapse or fail to correlate with the target distribution.

---

## 1. Introduction and problem setting

### 1.1 Motivation

Phylogenetic inference requires sampling trees from a distribution proportional to their likelihood under a sequence alignment model. **PhyloGFN** (ICLR 2024) frames this as a GFlowNet problem: train a generative policy over merge trajectories so that terminal trees are sampled with frequency proportional to reward. We ask whether comparable sampling quality can be achieved — or exceeded — with policy-gradient methods, and what modifications are necessary when the outcome space is large and each terminal tree has many valid construction paths.

### 1.2 Phylogenetic tree construction as an MDP

A tree is built by repeatedly merging subtrees until one tree remains. At each of \(N-1\) merge steps the forward policy \(P_F\) selects:

1. **Tree action:** which pair of subtrees to merge (from \(\binom{k}{2}\) options when \(k\) subtrees remain).
2. **Edge actions:** branch lengths on the new edge, sampled from the edge network.

The episode terminates when all taxa are joined. The terminal reward is a shifted log-likelihood:

\[
R(x) = c + \log L(x)
\]

with shift \(c \in \{3600, 5000, 12000\}\) for 5, 10, and 27 taxa respectively. The shift ensures \(R(x) > 0\), which is required for the importance weights used by IPS-GRPO and Learned-Reverse IPS-GRPO.

**Outcome granularity.** We evaluate at **signature level**: each outcome is a unique (topology, branch lengths) pair. This is finer than topology-only evaluation and matches the full generative model. At 10 taxa there are \(\sim 3.4 \times 10^7\) theoretical topologies; in 1M samples we observe near-unique signatures for the well-performing methods.

**Forward path probability.** For a complete trajectory \(\tau = (a_1, \ldots, a_{N-1})\):

\[
\log P_F(\tau) = \sum_{t=1}^{N-1} \log P_F(a_t \mid s_{t-1})
\]

This is recorded exactly at rollout time by the behavior policy, before any gradient step.

### 1.3 Target distribution

We want the policy's terminal marginal to satisfy:

\[
P_F(x) \propto R(x) \qquad \text{equivalently} \qquad P_F(x) = \frac{R(x)}{Z}
\]

where \(Z = \sum_x R(x)\) is the partition function. We do not need to estimate \(Z\) explicitly for evaluation; proportionality is what matters.

### 1.4 Trajectory multiplicity

The same terminal tree \(x\) can be built by many merge orderings \(\tau\). Let \(T(x)\) denote the set of valid trajectories to \(x\), and \(m(x) = |T(x)|\). The forward marginal is:

\[
P_F(x) = \sum_{\tau \in T(x)} P_F(\tau)
\]

For 5 taxa, \(m(x)\) ranges from 8 to 48 depending on topology. At 27 taxa, \(m(x)\) is enormous and cannot be computed. This is the central difficulty:

- **GRPO** treats each trajectory independently and has no correction for the fact that many paths lead to the same terminal.
- **IPS-GRPO** tries to estimate \(P_F(x)\) from within-batch counts, but at signature granularity almost every sample is unique — counts carry no signal.
- **GFlowNet** and **Learned-Reverse IPS-GRPO** both incorporate a backward policy that distributes probability mass over paths to each terminal, cancelling the unknown \(m(x)\).

### 1.5 Evaluation

All methods are evaluated by drawing 1M samples from the trained policy and comparing the empirical distribution to a reference. The primary metric is **Pearson correlation** between estimated terminal probability and reward (vs an ideal reference distribution from high-sample reference runs).

**Main diagnostic plot:** `model_probability_vs_reward` — scatter of estimated probability vs reward for each observed signature, with Pearson r annotated. A perfect reward-proportional sampler produces a tight linear relationship.

**Additional metrics:**
- **Unique signatures / 1M samples** — coverage; mode collapse shows up as low counts.
- **Topologies observed** — at 5 taxa, 105/105 means full support coverage.
- **IPS effective sample size (ESS) fraction** — for importance-weighted methods, measures weight concentration (1.0 = all weights equal).
- **Total variation (TV) distance** — used on the toy DAG where the ideal distribution is known exactly.

---

## 2. Methods

All four methods share the same forward policy architecture (tree network + edge network), rollout procedure (`rollout_worker_phylo.py`), and evaluation pipeline. They differ only in how the training signal is computed from rollouts.

| Method | Training signal | Handles multiplicity? | Partition function \(Z\) |
|--------|----------------|----------------------|--------------------------|
| **GRPO** | Group-normalized rewards → PPO | No | Not used |
| **IPS-GRPO** | \(R(x) / \hat{p}(x)\) within batch → PPO | No | Not used |
| **GFlowNet** | TB squared loss with fixed uniform \(P_B\) | Yes | Explicitly learned |
| **Learned-Reverse IPS-GRPO** | \(R(x)\, q_\phi(\tau \mid x) / P_F(\tau)\) → PPO | Yes | Absorbed by normalization |

### 2.1 GRPO

Standard group-relative policy optimization. After rollout, rewards are grouped (default group size matches batch structure) and z-scored within each group:

\[
A_i = \frac{R(x_i) - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}
\]

These advantages feed a clipped PPO surrogate on per-step log-probabilities. There is no importance weighting and no backward correction.

**Why it fails here.** GRPO optimizes for high reward, not reward-proportional sampling. With no mechanism to upweight rare high-reward terminals relative to common low-reward ones, the policy collapses to a small set of modes. At 5 taxa we observe a single unique signature in 1M samples; at 10 taxa, 2,850 out of 1M.

### 2.2 IPS-GRPO

Importance-sampling GRPO: terminal propensity is estimated by counting within a batch of \(G\) rollouts:

\[
\hat{p}(x) = \frac{\text{count}(x)}{G}, \qquad w_i = \frac{R(x_i)}{\hat{p}(x_i)}
\]

Advantages are derived from \(w\) (batch z-score or running EMA normalization) and passed to the same PPO surrogate as GRPO.

**Why it fails at signature level.** IPS-GRPO requires repeated observations of the same terminal within a batch to estimate \(\hat{p}(x)\). With batch size 4096 and millions of possible signatures, each terminal appears 0 or 1 times per batch. Then \(\hat{p}(x) \in \{0, 1/G\}\) is pure noise: weight saturates at \(G \cdot R(x)\) or is undefined. Observed: only 66K unique signatures in 1M samples at 5 taxa (vs 951K for Learned-Reverse IPS-GRPO), covering just 3 of 105 topologies.

**When IPS-GRPO could work.** At topology-level granularity with smaller outcome spaces and large batches, repeated terminals are more likely and count-based propensity estimates become viable. Our signature-level evaluation is deliberately harder and more realistic for the full generative model.

### 2.3 GFlowNet (PhyloGFN baseline)

The ICLR 2024 PhyloGFN approach. Training minimizes the trajectory-balance (TB) squared residual over sampled trajectories:

\[
\mathcal{L}_{\text{TB}} = \left(\log Z + \log P_F(\tau) - \log R(x) - \log P_B(\tau \mid x)\right)^2
\]

where:
- \(P_F(\tau)\) is the forward path probability (recorded at rollout).
- \(P_B(\tau \mid x)\) is a **fixed uniform backward policy** over valid reverse merge actions at each step.
- \(\log Z\) is a learned scalar partition function.

The TB constraint, when satisfied, implies \(P_F(x) \propto R(x)\). GFlowNet also uses a replay buffer (for 5/10 taxa runs) to reuse past trajectories. This is our strongest non-proposed baseline.

**Evaluation note.** GFlowNet's sampling quality is assessed via the TB identity: the pathwise quantity \(P_F(\tau)/P_B(\tau \mid x)\) should correlate with \(R(x)\) when the model is balanced. We plot this as `model_probability_vs_reward` and `log_probability_vs_log_reward`.

### 2.4 Learned-Reverse IPS-GRPO (proposed)

We retain the IPS-GRPO training framework (importance-weighted advantages → PPO) but replace batch counting with a **trajectory-level backward-corrected weight**:

\[
w(\tau) = R(x) \cdot \frac{q_\phi(\tau \mid x)}{P_F(\tau)}
\]

In log space:

\[
\log w = \log R(x) + \log q_\phi(\tau \mid x) - \log P_F(\tau)
\]

where:
- \(P_F(\tau)\): exact forward path probability, recorded at rollout time.
- \(q_\phi(\tau \mid x)\): **learned backward policy** — a normalized distribution over paths reaching terminal \(x\).

Advantages derived from \(\log w\) (running EMA normalization in phylogeny runs) are passed to path-level PPO. The forward update is equivalent to a stochastic gradient step toward making \(P_F\) match the target trajectory distribution \(p^*(\tau) \propto R(x)\, q_\phi(\tau \mid x)\).

**Relation to GFlowNet.** At initialization, \(q_\phi\) is uniform — identical to GFlowNet's fixed \(P_B\). As training proceeds, \(q_\phi\) is fit to the forward policy's conditional path distribution \(P_F(\tau \mid x)\), reducing weight variance. GFlowNet instead learns \(Z\) and enforces balance via a squared loss; we learn \(q_\phi\) and enforce proportionality via importance-weighted policy gradients.

---

## 3. Why a backward policy?

### 3.1 The multiplicity problem, concretely

Consider terminal tree \(x\) with \(m(x) = 5\) distinct merge orderings. If we use a naive weight \(R(x)/P_F(\tau)\):

- Each of the 5 paths gets a different weight.
- Summing over paths: total mass \(\propto m(x) \cdot R(x)\), not \(R(x)\).
- A terminal reachable by 100 paths receives 100× the target mass of one reachable by 1 path.

**GRPO** sidesteps this by ignoring paths entirely — but then cannot enforce proportionality.

**IPS-GRPO** tries to divide by an estimate of \(P_F(x)\) rather than \(P_F(\tau)\), but the count-based estimate is unusable at signature granularity.

### 3.2 Backward correction

A backward policy \(q_\phi(\tau \mid x)\) assigns probability over reverse paths from terminal \(x\) back to the root, normalized per terminal:

\[
\sum_{\tau \in T(x)} q_\phi(\tau \mid x) = 1 \qquad \forall x
\]

The corrected weight is:

\[
w(\tau) = R(x) \cdot \frac{q_\phi(\tau \mid x)}{P_F(\tau)}
\]

Summing the target mass over all paths to \(x\):

\[
\sum_{\tau \in T(x)} R(x) \cdot q_\phi(\tau \mid x) = R(x) \sum_{\tau \in T(x)} q_\phi(\tau \mid x) = R(x)
\]

The unknown \(m(x)\) never appears. The terminal marginal of the target is \(R(x)/Z\) for **any** normalized \(q_\phi\).

### 3.3 Why learn \(q_\phi\) rather than fix it uniform?

GFlowNet uses a fixed uniform \(P_B\), which is valid but typically far from the forward conditional \(P_F(\tau \mid x)\). When \(q_\phi \neq P_F(\cdot \mid x)\), weights vary across paths to the same terminal, inflating variance and reducing ESS.

When \(q_\phi = P_F(\tau \mid x)\), the weight simplifies:

\[
w(\tau) = \frac{R(x) \cdot P_F(\tau \mid x)}{P_F(\tau)} = \frac{R(x)}{P_F(x)}
\]

which is **constant across all paths to \(x\)** — zero within-terminal variance, maximal ESS.

We fit \(q_\phi\) by maximum likelihood on forward rollouts:

\[
\mathcal{L}_{\text{rev}}(\phi) = -\frac{1}{N}\sum_i \log q_\phi(\tau_i \mid x_i)
\]

This minimizes a \(P_F(x)\)-weighted KL from \(P_F(\tau \mid x)\) to \(q_\phi\). No rewards enter the reverse loss — it is purely "given terminal \(x\), predict which path was taken."

**Correctness vs variance.** A poorly fit \(q_\phi\) increases weight variance but **cannot bias the terminal marginal**, because normalization over paths to \(x\) holds for any \(\phi\). Learning \(q_\phi\) is purely a variance-reduction strategy on top of a valid estimator.

### 3.4 Reverse policy architecture

**Toy DAG:** MLP over 6 features per reverse step (child coordinates, terminal coordinates, remaining distance to terminal). Masked categorical over valid parent edges. Zero-initialized → uniform at step 0.

**Phylogeny (5 taxa):** Tabular — one learned logit per structural merge history (180 histories mapping to 105 topologies). Exact enumeration is feasible at this scale.

**Phylogeny (10/27 taxa):** Per-step MLP (`PhyloLearnedReversePolicy`, 256 hidden units, 3 layers) over 9 features: forest size, step index, merge pair indices, shifted terminal log-score, topology hash features. Masked categorical over valid merge actions.

### 3.5 Training loop

Each iteration:

1. **Roll out** trajectories with current \(P_F\). Record per-step log-probs, action sequences, terminals, rewards.
2. **Score** with **frozen** \(q_\phi\) (fitted on prior batches only). Compute \(\log w\), normalize to advantages.
3. **Update** \(P_F\) via path-level PPO on the pooled batch.
4. **Update** \(q_\phi\) by MLE on the same batch (for use in the *next* iteration).

The freeze-then-fit ordering is critical: if \(q_\phi\) were updated before scoring the current batch, it would be inflated on paths that happen to appear, creating a self-reinforcing bias. The one-update lag (\(q_\phi\) fit to \(\theta_{t-1}\), used to weight samples from \(\theta_t\)) is small because the forward policy moves slowly.

**Hyperparameters (27 taxa best run):**
- Forward: batch 1024, 32k epochs, no replay
- Reverse: lr \(10^{-3}\), 8 MLE epochs per update, gradient clip 1.0
- Advantages: running EMA normalization (decay 0.99)

---

## 4. Toy DAG experiments

Before scaling to phylogeny, we validated the backward-correction logic on a controlled grid DAG environment (`compound_action_rl/dag_toy_dataset/`). The toy shares the same forward/reverse policy structure and IPS-GRPO training loop as the phylogeny code, but with a small (or large) enumerable state space where the ideal distribution is known exactly.

### 4.1 Environment

States are grid points \((x, y)\) with depth \(x + y\). Compound actions choose direction (RIGHT/UP) and step length \(1..K\). Episode terminates when \(x + y = B\) (budget). Reward is terminal-only, a function of the \(x\)-coordinate on the frontier.

**Small instance** (\(B=3, K=3\)):
- 10 states, 18 trajectories, 4 terminals
- Multiplicities \((4, 5, 5, 4)\)
- Rewards \((1.0, 0.80, 0.20, 0.05)\)
- Ideal probabilities: \((48.8\%, 39.0\%, 9.8\%, 2.4\%)\)

**Large instance** (\(B=1024, K=3\)):
- 1025 terminals on the frontier
- Path probabilities as small as \(e^{-100}\)
- Intractable path enumeration — analogous to phylogeny

![DAG structure (budget=3, max_step=3)](../compound_action_rl/dag_toy_dataset/dag_budget3_step3.png)

![Ideal reward-proportional sampling (10k samples, budget=3)](../compound_action_rl/dag_toy_dataset/ideal_reward_sampling_10000.png)

On the toy we compare **Learned-Reverse IPS-GRPO** and **GFlowNet** directly; GRPO and IPS-GRPO exhibit the same failure modes as on phylogeny (collapse and count noise respectively), which is why we focus on the two multiplicity-aware methods here.

### 4.2 Training configuration (large scale)

Run: `data/learned_reverse_ips_runs/20260730_183022_b1024_gs16_seed0/`

| Parameter | Value |
|-----------|-------|
| Budget \(B\) | 1024 |
| Max step \(K\) | 3 |
| Group size | 16 |
| Num updates | 10,000 |
| Forward LR | \(3 \times 10^{-4}\) |
| Reverse LR | \(10^{-3}\) |
| Reverse train epochs | 4 |
| Reverse hidden / layers | 128 / 2 |
| Advantage normalization | running |

### 4.3 Results

**Primary metric:** TV distance between sampled terminal distribution and \(R(x)/Z\) (`tv_reward_target`). Lower is better; 0 = exact match.

| Method | Setting | TV to target |
|--------|---------|-------------|
| Learned-Reverse IPS-GRPO | \(B=1024\), 10k updates | **0.189** (100k eval) |
| GFlowNet | \(B=128\), 2k updates | **0.193** (final eval) |

Training progression (Learned-Reverse IPS-GRPO, B=1024): TV improved from **0.867** at step 1 to **0.323** at step 10,000.

#### Training plots

![Training curves — TV, reverse loss, ESS, reward](../compound_action_rl/dag_toy_dataset/data/learned_reverse_ips_runs/20260730_183022_b1024_gs16_seed0/training_curves.png)

![Reverse policy diagnostics — edge accuracy, entropy, within-terminal std](../compound_action_rl/dag_toy_dataset/data/learned_reverse_ips_runs/20260730_183022_b1024_gs16_seed0/reverse_diagnostics.png)

The reverse diagnostics track how well \(q_\phi\) matches the forward conditional. **Within-terminal std** of \(\log[P_F(\tau)/q_\phi(\tau \mid x)]\) is the key quantity: it approaches zero as \(q_\phi \to P_F(\tau \mid x)\), and its decrease drives ESS up.

![Running normalization diagnostics](../compound_action_rl/dag_toy_dataset/data/learned_reverse_ips_runs/20260730_183022_b1024_gs16_seed0/running_normalization_diagnostics.png)

#### Sampling plots (100k eval)

![Terminal sampling vs ideal](../compound_action_rl/dag_toy_dataset/data/learned_reverse_ips_runs/20260730_183022_b1024_gs16_seed0/evaluation_100000_terminal_sampling.png)

![Trajectory coverage](../compound_action_rl/dag_toy_dataset/data/learned_reverse_ips_runs/20260730_183022_b1024_gs16_seed0/evaluation_100000_trajectory_coverage.png)

**Takeaway.** Learned-Reverse IPS-GRPO achieves GFlowNet-level terminal distribution quality on the toy, in a setting where both methods are viable. This validates the backward-correction logic before applying it to phylogeny, where IPS-GRPO fails entirely and GFlowNet's performance degrades at scale.

---

## 5. Phylogenetic experiments

### 5.1 Experimental setup

| | 5 taxa | 10 taxa | 27 taxa |
|---|---|---|---|
| Dataset | DS1_reduced | DS1_reduced_10taxa | DS1 (full) |
| Reward | \(R(x) = 3600 + \log L(x)\) | \(R(x) = 5000 + \log L(x)\) | \(R(x) = 12000 + \log L(x)\) |
| Topologies (exact) | 105 | \(\sim 3.4 \times 10^7\) | very large |
| Batch size | 4096 | 4096 | 1024 |
| Epochs | 10,000 | 10,000 | 32,000 |
| Model | full (tree + edge) | full | full |
| Outcome level | signature | signature | signature |
| Replay | no (all four where applicable) | no | no |

**Evaluation:** 1M samples from the trained checkpoint unless noted. All comparison plots and metrics are in `grpo_experiments/comparisons/`.

**GFlowNet specifics:** 5/10 taxa runs use replay buffer (4096); 27 taxa comparison uses no replay to match Learned-Reverse IPS-GRPO. All GFlowNet runs use the original PhyloGFN training code in `og_code/`.

**Learned-Reverse IPS-GRPO specifics:** Reverse policy is MLP (256 hidden, 3 layers) for 10/27 taxa; 8 reverse MLE epochs per update at 27 taxa. No replay.

### 5.2 Five taxa

The smallest setting: only 105 topologies, all enumerable. This is a sanity check — both multiplicity-aware methods should perform well; GRPO and IPS-GRPO should fail.

| Method | Pearson r vs ideal | Unique signatures / 1M | Topologies observed |
|--------|-------------------:|-----------------------:|--------------------:|
| **Learned-Reverse IPS-GRPO** | **0.994** | 951,175 | 105 / 105 |
| GFlowNet | 0.982 | 960,850 | 105 / 105 |
| IPS-GRPO | 0.319 | 66,268 | 3 / 105 |
| GRPO | — (collapsed) | 1 | 1 / 105 |

Both Learned-Reverse IPS-GRPO and GFlowNet cover all 105 topologies with near-perfect correlation to the ideal. IPS-GRPO explores only 3 topologies; GRPO finds a single mode.

#### Learned-Reverse IPS-GRPO

![5 taxa — Learned-Reverse IPS-GRPO training](comparisons/5taxa/learned_reverse_training_curves.png)

Training curves show reverse NLL (fit of \(q_\phi\)), IPS ESS fraction (weight reliability), and mean log score (forward policy improvement).

![5 taxa — Learned-Reverse IPS-GRPO sampling](comparisons/5taxa/learned_reverse_ips_model_probability_vs_reward.png)

#### GFlowNet

![5 taxa — GFlowNet training](comparisons/5taxa/gflownet_training_curves.png)

![5 taxa — GFlowNet sampling](comparisons/5taxa/gflownet_model_probability_vs_reward.png)

#### Baselines and comparison

![5 taxa — IPS-GRPO](comparisons/5taxa/count_ips_model_probability_vs_reward.png)

![5 taxa — GRPO (collapsed to single mode)](comparisons/5taxa/grpo_model_probability_vs_reward.png)

![5 taxa — four-method side-by-side comparison](comparisons/5taxa/sampling_comparison_best_fit_grid.png)

### 5.3 Ten taxa

| Method | Pearson r | R² | Unique signatures / 1M |
|--------|----------:|---:|-----------------------:|
| **Learned-Reverse IPS-GRPO** | **0.976** | 0.952 | 999,986 |
| GFlowNet | 0.881 | 0.777 | 1,000,000 |
| IPS-GRPO | 0.050 | 0.003 | 140,109 |
| GRPO | 0.033 | 0.001 | 2,850 |

Learned-Reverse IPS-GRPO achieves higher correlation and R² than GFlowNet while maintaining comparable coverage (1M unique signatures). GRPO and IPS-GRPO show near-zero correlation.

#### Learned-Reverse IPS-GRPO

![10 taxa — Learned-Reverse IPS-GRPO training](comparisons/10taxa/learned_reverse_training_curves.png)

![10 taxa — Learned-Reverse IPS-GRPO sampling](comparisons/10taxa/learned_reverse/model_probability_vs_reward.png)

#### GFlowNet

![10 taxa — GFlowNet training](comparisons/10taxa/gflownet_training_curves.png)

![10 taxa — GFlowNet sampling](comparisons/10taxa/gflownet/model_probability_vs_reward.png)

![10 taxa — GFlowNet (partition-calibrated sampling)](comparisons/10taxa/gflownet/partition_calibrated_probability_vs_reward.png)

#### Four-method comparison

![10 taxa — four-method comparison](comparisons/10taxa/sampling_comparison_best_fit_grid.png)

#### Early training (10 taxa, 100k samples)

To compare convergence speed, we evaluate checkpoints before full training completes:

| Epoch | Method | Pearson r (prob) | Pearson r (log) | ESS fraction |
|------:|--------|-----------------:|----------------:|-------------:|
| 1000 | Learned-Reverse IPS-GRPO | 0.953 | 0.801 | 0.991 |
| 1000 | GFlowNet | 0.861 | 0.674 | 0.974 |
| 4000 | Learned-Reverse IPS-GRPO | 0.971 | 0.877 | 0.994 |
| 4000 | GFlowNet | 0.880 | 0.709 | 0.977 |

Learned-Reverse IPS-GRPO reaches strong sampling quality earlier. Early-checkpoint plots are in `comparisons/10taxa/early_epoch1000_100k/` and `early_epoch4000_100k/`.

### 5.4 Twenty-seven taxa

The most challenging setting: full DS1 dataset, 27 taxa, no replay, 32k epochs. Only Learned-Reverse IPS-GRPO and GFlowNet were run to completion at this scale.

| Method | Pearson r (prob) | Pearson r (log) | ESS fraction |
|--------|-----------------:|----------------:|-------------:|
| **Learned-Reverse IPS-GRPO** | **0.977** | 0.977 | **0.999** |
| GFlowNet | 0.002 | 0.024 | 0.847 |

Additional metrics (Learned-Reverse IPS-GRPO):
- Unique signatures: 999,987 / 1M
- Topology TV to ideal: 0.010
- Signature TV on observed support: 0.010
- Log-weight std: 0.027

The gap at 27 taxa is stark: Learned-Reverse IPS-GRPO maintains near-perfect correlation with the target distribution, while GFlowNet's sampling shows no meaningful relationship between estimated probability and reward.

#### Training dynamics (Learned-Reverse IPS-GRPO)

The reverse policy is the bottleneck at this scale. At 10k epochs, reverse loss was still 4.82 and ESS fraction only 0.49. Extending to 32k epochs with `reverse_train_epochs=8` brought reverse loss to \(\sim 4 \times 10^{-5}\) and ESS to 0.999.

| Taxa | Epochs | Final reverse loss | Final IPS ESS | Final mean log score |
|------|-------:|-------------------:|--------------:|---------------------:|
| 5 | 10,000 | 2.04e-05 | 0.9998 | 267.0 |
| 10 | 10,000 | 6.46e-05 | 0.9944 | 335.6 |
| 27 (10k) | 10,000 | 4.82 | 0.488 | 587.3 |
| **27 (32k)** | **32,000** | **3.87e-05** | **0.999** | **2026.2** |

![Training curves — 5 / 10 / 27 taxa (Learned-Reverse IPS-GRPO)](comparisons/learned_reverse_training_curves_5_10_27.png)

#### Learned-Reverse IPS-GRPO

![27 taxa — Learned-Reverse IPS-GRPO training](comparisons/27taxa/learned_reverse_noreplay_training_curves.png)

![27 taxa — Learned-Reverse IPS-GRPO sampling](comparisons/27taxa/learned_reverse/model_probability_vs_reward.png)

![27 taxa — Learned-Reverse IPS-GRPO (log scale)](comparisons/27taxa/learned_reverse/log_model_probability_vs_log_reward.png)

![27 taxa — Learned-Reverse IPS-GRPO (partition-calibrated)](comparisons/27taxa/learned_reverse/partition_calibrated_model_probability_vs_reward.png)

#### GFlowNet

![27 taxa — GFlowNet training](comparisons/27taxa/gflownet_noreplay_training_curves.png)

![27 taxa — GFlowNet sampling (log--log; linear prob vs reward is flat)](comparisons/27taxa/gflownet/log_probability_vs_log_reward.png)

![27 taxa — GFlowNet (linear scale, for reference)](comparisons/27taxa/gflownet/model_probability_vs_reward.png)

![27 taxa — GFlowNet (partition-calibrated)](comparisons/27taxa/gflownet/partition_calibrated_probability_vs_reward.png)

Despite a decreasing TB loss and a stable learned partition estimate, the GFlowNet sampler does not recover a reward-proportional terminal distribution under our signature-level evaluation.

#### Diagnostic plots (27 taxa, Learned-Reverse IPS-GRPO)

![Signature q̂ vs log-likelihood (1M samples)](comparisons/27taxa/learned_reverse/signature_qhat_vs_loglikelihood_1000k.png)

![Topology checkpoint vs reward reference](comparisons/27taxa/learned_reverse/topology_checkpoint_vs_reward_reference.png)

![Log-likelihood checkpoint vs reward reference](comparisons/27taxa/learned_reverse/loglikelihood_checkpoint_vs_reward_reference.png)

---

## 6. Discussion

### 6.1 Summary of findings

1. **GRPO** collapses to a handful of modes at all taxa counts. Without importance weighting or backward correction, there is no pressure to sample proportionally across the outcome space.

2. **IPS-GRPO** fails at signature granularity because within-batch count estimates are noise. It explores a tiny fraction of the outcome space (3/105 topologies at 5 taxa; 140K/1M signatures at 10 taxa) and shows near-zero correlation with the target.

3. **GFlowNet** is a strong baseline at 5 and 10 taxa (Pearson r = 0.98 and 0.88) but breaks down at 27 taxa (Pearson r ≈ 0.002). The TB loss continues to decrease during training, but the resulting sampler does not produce reward-proportional terminal distributions under our evaluation.

4. **Learned-Reverse IPS-GRPO** matches or exceeds GFlowNet at every scale (Pearson r = 0.994, 0.976, 0.977 at 5/10/27 taxa). The learned backward policy \(q_\phi\) is the key: it provides the multiplicity correction that IPS-GRPO lacks, and adapts to the forward policy's path distribution in a way that GFlowNet's fixed uniform \(P_B\) does not.

### 6.2 Why Learned-Reverse IPS-GRPO scales better than GFlowNet (hypothesis)

At 27 taxa, GFlowNet must simultaneously learn the forward policy, a scalar partition function \(Z\), and satisfy a per-trajectory balance constraint with a fixed backward policy. Learned-Reverse IPS-GRPO instead:

- Absorbs \(Z\) into batch/running advantage normalization (never estimated explicitly).
- Learns a flexible backward policy \(q_\phi\) that tracks the forward conditional, keeping ESS high.
- Uses a policy-gradient update that directly pushes sampled trajectories toward the reward-proportional target.

The reverse policy training is the main cost (32k epochs, 8 MLE steps per update at 27 taxa), but once \(q_\phi\) converges, the importance weights become nearly constant and training stabilizes.

### 6.3 Limitations

- **Path-space freedom.** The terminal marginal is constrained, but the conditional path distribution \(P_F(\tau \mid x)\) is not — any conditional is a fixed point if \(q_\phi\) matches it. Path collapse is possible, as with GFlowNet's free backward policy.
- **Reverse policy coverage.** \(q_\phi\) is fit by MLE weighted by \(P_F(x)\), so rare high-reward terminals may have poorly estimated reverse probabilities early in training.
- **Non-stationary target.** As \(q_\phi\) improves, the target trajectory distribution moves. Only its terminal marginal is invariant.
- **Compute.** 27-taxa training required 32k epochs and multiple days of GPU time.
- **GFlowNet 27t comparison.** We have not yet fully diagnosed why GFlowNet fails at 27 taxa (optimization difficulty, partition estimation, lack of replay, or evaluation mismatch). This remains an open question.

### 6.4 Next steps

- Ablation: learned \(q_\phi\) vs fixed uniform \(P_B\) under the IPS-GRPO framework (isolating the value of learning vs backward correction alone).
- Investigate GFlowNet failure mode at 27 taxa.
- Additional benchmark datasets (DS2–DS8) and sample-efficiency / wall-clock comparisons.
- Path-entropy monitoring and regularization during training.

---

## Appendix

### A. Reference runs

| Method | 5 taxa | 10 taxa | 27 taxa |
|--------|--------|---------|---------|
| Learned-Reverse IPS-GRPO | `learned_reverse_runs/20260730_160341_learned_reverse_5taxa_mlp_shifted_linear_b4096_learned_reverse_ips_grpo` | `learned_reverse_runs/20260803_124837_learned_reverse_10taxa_mlp_shifted_linear_b4096_learned_reverse_ips_grpo` | `learned_reverse_runs/20260806_144004_learned_reverse_27taxa_mlp_shifted_linear_b1024_rlr1e-3_rev8x_learned_reverse_ips_grpo` |
| GFlowNet | `og_code/experiments/full_model/20260703_172421_phylgfn_logreward_g4096_1m_full_replay_op3277_r819_rb4096` | `og_code/experiments/full_model/20260804_112555_20260803_124841_phylgfn_logreward_10taxa_g4096_1m_full_replay_op3277_r819_rb4096_resumed` | `og_code/experiments/full_model/20260806_150835_phylgfn_logreward_27taxa_g1024_noreplay_shift12000` |
| IPS-GRPO | `learned_reverse_runs/20260802_174242_count_ips_5taxa_full_sig_b4096_ips_grpo` | `learned_reverse_runs/20260804_011029_count_ips_10taxa_full_sig_b4096_shift5000_ips_grpo` | — |
| GRPO | `learned_reverse_runs/20260802_174242_grpo_5taxa_full_sig_b4096_grpo` | `learned_reverse_runs/20260804_133452_plain_grpo_10taxa_full_sig_b4096_shift5000_grpo` | — |

### B. Code and documentation

| Purpose | Path |
|---------|------|
| Toy DAG prototype | `compound_action_rl/dag_toy_dataset/learned_reverse_ips.py` |
| Phylo trainer | `grpo_experiments/learned_reverse_ips_grpo.py` |
| Phylo reverse policy (MLP) | `grpo_experiments/phylo_learned_reverse_policy.py` |
| GRPO baseline | `grpo_experiments/train.py --method grpo` |
| IPS-GRPO baseline | `grpo_experiments/ips_grpo/` |
| GFlowNet baseline | `og_code/` + `src/gfn/tb_gfn_phylo.py` |
| Comparison plots + metrics | `grpo_experiments/comparisons/` |
| Detailed method derivation | `compound_action_rl/dag_toy_dataset/learned_reverse_ips_writeup.md` |

### C. Comparison plot index

**5 taxa:** `comparisons/5taxa/` — Learned-Reverse and GFlowNet `*_training_curves.png`, per-method `model_probability_vs_reward`, `sampling_comparison_best_fit_grid`

**10 taxa:** `comparisons/10taxa/` — Learned-Reverse and GFlowNet training + sampling plots, `sampling_comparison_best_fit_grid`, early checkpoints in `early_epoch1000_100k/` and `early_epoch4000_100k/`

**27 taxa:** `comparisons/27taxa/` — Learned-Reverse and GFlowNet training (no-replay) + sampling plots, partition-calibrated variants, signature/topology diagnostics

**Cross-taxa:** `comparisons/learned_reverse_training_curves_5_10_27.png`

**LaTeX figures:** `writeup_package/figures/` — per-taxa `training.png`, `sampling.png`, `gflownet_training.png`, `gflownet_sampling.png`; cross-taxa `training_all_taxa.png`

**Toy DAG:** `compound_action_rl/dag_toy_dataset/data/learned_reverse_ips_runs/20260730_183022_b1024_gs16_seed0/`
