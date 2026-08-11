# Learned-Reverse Trajectory IPS — how the method works

Reference implementation: `compound_action_rl/dag_toy_dataset/learned_reverse_ips.py`
(plus `dag_env.py`, `count_ips.py`, `trajectory_ips.py`, `networks.py`, `config.py`).

This is a first-draft write-up meant for logic verification. Everything below is
stated in text; figures can be added later.

---

## 1. The environment and what we want

**States.** A state is an integer grid point `s = (x, y)` with `x, y >= 0`. Its
*depth* is `x + y`. The start state is the root `(0, 0)`.

**Compound action.** One action is a pair `a = (d, l)`:

- direction `d ∈ {RIGHT, UP}`,
- step length `l ∈ {1, ..., K}` where `K = max_step` (default 3).

`RIGHT` with length `l` moves `(x, y) -> (x + l, y)`; `UP` moves
`(x, y) -> (x, y + l)`. So depth increases by exactly `l` per action.

**Termination.** There is a fixed budget `B`. A state is terminal iff
`x + y == B`. The step length is masked so that `l <= min(K, B - depth(s))`,
hence every episode lands *exactly* on the terminal frontier `x + y = B` and no
explicit stop action is needed. Episode length is variable: between `ceil(B/K)`
and `B` actions.

**Forward policy (two hierarchical heads).** The behaviour/target policy is
factorised as

```
P_F(a | s) = pi_dir(d | s) * pi_step(l | s, d)
```

`pi_dir` is an MLP on a one-hot encoding of `(x, y, remaining)`; `pi_step` reuses
`pi_dir`'s hidden representation concatenated with the one-hot chosen direction
(`networks.py`). Both heads are masked categoricals, so every legal action has
strictly positive probability. For a complete trajectory
`tau = (a_1, ..., a_T)`:

```
log P_F(tau) = sum_t [ log pi_dir(d_t | s_{t-1}) + log pi_step(l_t | s_{t-1}, d_t) ]
```

This number is recorded **at rollout time** by the behaviour policy, before any
gradient step, so it is exact (not re-estimated).

**Reward.** Reward is terminal-only: `R(x) ∈ (0, 1]` for each terminal state `x`
on the frontier, indexed by its `x`-coordinate. Let `Z = sum_x R(x)`.

**Goal.** We want the forward policy's *terminal marginal* to be
reward-proportional:

```
P_F(x)  ->  p*(x) = R(x) / Z
```

Notation used throughout: `T(x)` = the set of complete trajectories ending at
terminal `x`, and `m(x) = |T(x)|` its (unknown, astronomically large)
multiplicity.

---

## 2. Why the two obvious estimators fail

**(a) Count-based IPS** (`count_ips.py`) estimates the terminal propensity by
counting inside a batch of `G` rollouts:

```
p_hat(x) = count(x) / G,     weight = R(x) / p_hat(x)
```

This needs each sampled terminal to appear several times in a batch. With
`B = 128` the frontier has 129 terminals and, later in training, most batches see
each terminal once or not at all, so `p_hat` is pure noise (and `1/p_hat`
saturates at `G`).

**(b) Exact trajectory IPS** (`exact_probability_ips.py`) uses the exactly known
`P_F(tau)`:

```
weight = R(x) / P_F(tau)
```

This is exact but targets the wrong distribution. Summing the induced target
mass over all paths to `x` gives `m(x) * R(x)`, not `R(x)`, so terminals reachable
by more paths are over-sampled. Fixing this requires `m(x)`, which is available
only by dynamic programming in this toy (`trajectory_ips.py`) and is unavailable
in the real setting (phylogenetic trees).

**The method in this file removes the `m(x)` factor without ever computing it,**
by replacing the unknown multiplicity with a *learned, normalised reverse
proposal over paths*.

---

## 3. The estimator

### 3.1 The weight

For a rollout `tau` ending at terminal `x`:

```
w(tau) = R(x) * q_phi(tau | x) / P_F(tau)
```

- `R(x)` — terminal reward.
- `P_F(tau)` — exact frozen behaviour-policy path probability (recorded at rollout).
- `q_phi(tau | x)` — a **learned reverse policy**: a distribution over the paths
  that reach `x`, conditioned on `x`.

In code (`learned_reverse_ips_advantages`, lines ~351-352):

```python
implied_terminal_log_probability = log_p_f - log_q          # log[P_F(tau)/q(tau|x)]
log_weights = np.log(reward_array) - implied_terminal_log_probability
```

which is exactly `log R + log q - log P_F`.

### 3.2 The one property `q_phi` must have

**`q_phi(. | x)` must be a normalised probability distribution over `T(x)`:**

```
sum_{tau in T(x)} q_phi(tau | x) = 1     for every terminal x
```

Section 4 shows this holds **by construction, for arbitrary network parameters
`phi`, at every point in training** — it is not something training has to
achieve.

### 3.3 Why the weight is correct (the exact statement)

Define the *target trajectory distribution*

```
p*_phi(tau) = R(x(tau)) * q_phi(tau | x(tau)) / Z
```

It is a valid distribution, because

```
sum_tau p*_phi(tau) = (1/Z) sum_x R(x) * sum_{tau in T(x)} q_phi(tau|x)
                    = (1/Z) sum_x R(x) * 1
                    = 1
```

and — the key point — its **terminal marginal is exactly the goal**:

```
sum_{tau in T(x)} p*_phi(tau) = R(x)/Z * 1 = p*(x)      for any phi
```

And `w` is precisely the importance weight of this target against the behaviour
policy:

```
w(tau) = Z * p*_phi(tau) / P_F(tau)
```

Therefore, with `tau ~ P_F`,

```
E[w(tau)] = sum_tau P_F(tau) * R(x) q_phi(tau|x) / P_F(tau)
          = sum_x R(x) * sum_{tau in T(x)} q_phi(tau|x)
          = sum_x R(x)
          = Z          (exactly, for any phi)
```

and for any function `f`, the self-normalised ratio
`sum_i w_i f(tau_i) / sum_i w_i` is consistent for `E_{p*_phi}[f]`. Two
consequences worth stressing:

1. **`Z` is never needed.** It is a constant multiplier on all weights and is
   removed by the batch normalisation in Section 6.
2. **`m(x)` is never needed, and no path enumeration or DP occurs.** The
   multiplicity is cancelled implicitly, because `q_phi` spreads a total mass of
   exactly 1 over however many paths reach `x`.

Also note that at the solution `P_F = p*_phi` we get `w(tau) = Z` for *every*
trajectory: the weights become constant, all advantages vanish, and ESS is
maximal. So the spread of `w` within a batch is a direct measure of the remaining
gap to the target.

### 3.4 The "implied terminal propensity" reading (intuition, with a caveat)

Rewrite the weight as

```
w(tau) = R(x) / P_hat(x; tau),    where  P_hat(x; tau) = P_F(tau) / q_phi(tau | x)
```

so the method *looks* like ordinary terminal-level IPS, `R(x)/P_F(x)`, with
`P_hat` standing in for the intractable `P_F(x) = sum_{tau in T(x)} P_F(tau)`.
This is the right intuition, and it is exact in one important case:

- **If `q_phi(. | x) = P_F(. | x)`** (the true forward conditional over paths),
  then `P_hat(x; tau) = P_F(tau) / [P_F(tau)/P_F(x)] = P_F(x)` for *every*
  `tau ∈ T(x)`. The plug-in becomes the true propensity, it no longer depends on
  which path was taken, and the within-terminal variance of the weight is zero.

Caveat to be explicit about, since it is the one place the naming can mislead:
`P_hat` is unbiased for `P_F(x)` under `tau ~ q_phi(.|x)`, but our rollouts come
from `tau ~ P_F(.|x)`, under which

```
E[P_hat(x; tau)] = P_F(x) * (1 + chi^2( P_F(.|x) || q_phi(.|x) ))  >=  P_F(x)
```

i.e. the plug-in propensity is biased *upward* whenever `q_phi` differs from the
true conditional. This does **not** compromise the method: the correctness
statement that matters is the trajectory-level one in Section 3.3, which is exact
for any `phi`. The role of `q_phi -> P_F(.|x)` is purely **variance reduction**,
not unbiasedness. I would recommend presenting 3.3 as the derivation and 3.4 only
as intuition.

---

## 4. `q_phi`: exact inputs, outputs, and why it is normalised

`q_phi` factorises the reverse walk from a terminal back to the root into
per-node choices of "which incoming edge did we arrive on".

### 4.1 The reverse decision problem

Reverse state: `(u, x)` — the current node `u` (non-root) and the fixed terminal
`x` we are conditioning on. Reverse action: which parent edge to undo, i.e. a
pair `(d, l)` meaning "we arrived at `u` by moving `l` in direction `d`", so the
parent is `u - l * e_d`. Valid reverse actions at `u` (`reverse_action_mask`):

```
(RIGHT, l) valid  iff  l <= u.x
(UP,    l) valid  iff  l <= u.y            for l = 1..K
```

so the number of valid parents is `min(K, u.x) + min(K, u.y)`, which is `>= 1`
for any non-root `u`.

### 4.2 Network inputs and outputs (`LearnedReversePolicy`)

**Input — one row per traversed edge, 6 continuous features** (`reverse_context`,
all divided by `B`):

| index | feature |
|---|---|
| 0 | `u.x / B` — current (child) node x |
| 1 | `u.y / B` — current node y |
| 2 | `x.x / B` — terminal x |
| 3 | `x.y / B` — terminal y |
| 4 | `(x.x - u.x) / B` — x-distance still to cover |
| 5 | `(x.y - u.y) / B` — y-distance still to cover |

Features 4-5 are linear functions of 0-3, included as a convenience for the MLP.
`(u, x)` is the complete reverse state, so this input is *sufficient* — the
reverse decision is Markov in it.

**Architecture.** `num_layers` blocks of `Linear + Tanh` (width
`hidden_size`, default 2 x 128), then a linear head producing `2K` logits.

**Output — `2K` logits, one per joint action `(d, l)`**, indexed by

```
index(d, l) = d * K + (l - 1)          # RIGHT=0, UP=1
```

so indices `0..K-1` are `RIGHT` with lengths `1..K` and `K..2K-1` are `UP`.
Invalid entries are set to `-1e9` before the softmax, so `q_phi` is a masked
categorical over exactly the valid parents.

**Initialisation.** The head weight *and* bias are zeroed
(`nn.init.zeros_`), so at step 0 all logits are 0 and `q_phi` is *exactly*
uniform over valid parents — i.e. it reproduces the fixed reference policy
`uniform_backward_log_probability` in `dag_env.py`. The method therefore starts
from a well-defined, already-valid reverse reference and only improves from
there. (This is the same estimator as `backward_corrected_ips.py`; the
contribution here is learning `q` instead of freezing it.)

**Path probability.** For a trajectory with edges `1..T`, letting `u_t` be the
*child* endpoint of edge `t`:

```
log q_phi(tau | x) = sum_{t=1..T} log q_phi( (d_t, l_t) | u_t, x )
```

Implementation: `_reverse_batch` walks each recorded episode forward from the
root, emitting one `(context, mask, action_index, episode_index)` row per edge;
`_reverse_path_log_probabilities_tensor` computes all per-edge log-probs in one
batched forward pass and `scatter_add`s them into per-episode path log-probs. It
also asserts that the replayed trajectory actually ends at the recorded terminal.
`reverse_path_log_probabilities` wraps this in `torch.inference_mode`, so the
weights entering the advantage computation are pure constants — no gradient flows
from the forward objective into `phi`, and none from `q_phi` into `theta`.

### 4.3 Why `sum_{tau in T(x)} q_phi(tau | x) = 1` for any `phi`

1. Each reverse step decreases depth by `l >= 1`, so the reverse walk cannot
   cycle and terminates in at most `B` steps.
2. The mask keeps both coordinates `>= 0`, and every non-root node has at least
   one valid parent, so the walk cannot get stuck: the root is the unique
   absorbing state and is reached with probability 1.
3. Reverse walks from `x` to the root correspond one-to-one with elements of
   `T(x)`.

Hence the reverse walk defines a proper probability distribution on `T(x)`, and
its path probabilities sum to 1. Nothing in this argument depends on `phi`, on
how well `q_phi` is trained, or on `m(x)`. **This is the structural reason the
method cannot be biased by a badly fitted `q`.**

**Support match.** `{tau : q_phi(tau|x) > 0}` is exactly `T(x)`, which is exactly
`{tau : P_F(tau) > 0, x(tau) = x}`: the reverse mask enumerates precisely the
in-DAG parents, and every such edge is also forward-legal (if the edge has length
`l` into a node of depth `<= B`, then the parent has `B - depth(parent) >= l`, so
the forward step mask permits it). The softmax makes `P_F(tau) > 0` for all
`tau`. So the importance weight never divides by zero and never misses mass.

### 4.4 Worked example: `B = 3`, `K = 3`, terminal `x = (1,2)`

The 5 paths to `(1,2)` and their reverse probabilities *at initialisation*
(uniform over parents; the parent count of a node `u` is
`min(3,u.x) + min(3,u.y)`):

| path | child nodes visited | per-edge reverse probs | `q(tau \| x)` |
|---|---|---|---|
| `R1-U2` | (1,0), (1,2) | 1 x 1/3 | 1/3 |
| `U2-R1` | (0,2), (1,2) | 1/2 x 1/3 | 1/6 |
| `R1-U1-U1` | (1,0), (1,1), (1,2) | 1 x 1/2 x 1/3 | 1/6 |
| `U1-R1-U1` | (0,1), (1,1), (1,2) | 1 x 1/2 x 1/3 | 1/6 |
| `U1-U1-R1` | (0,1), (0,2), (1,2) | 1 x 1/2 x 1/3 | 1/6 |

Sum = `1/3 + 4 * (1/6) = 1`. Two things this illustrates:

- Normalisation holds exactly, even though `m((1,2)) = 5` never appears anywhere.
- "Normalised" does **not** mean "uniform over paths" (`1/3` vs `1/6`). Uniform
  *per reverse step* is not uniform *per path*. Both are valid; they simply define
  different conditional targets. The terminal marginal is `R((1,2))/Z = 0.8/2.05
  = 39.02%` either way.

Concrete input rows for the path `R1-U2` (`B = 3`, `K = 3`, action order
`[R1,R2,R3,U1,U2,U3]`):

- edge 1: child `(1,0)`, context `(1/3, 0, 1/3, 2/3, 0, 2/3)`, mask
  `[1,0,0,0,0,0]`, action index `0` (`RIGHT`, `l=1`).
- edge 2: child `(1,2)`, context `(1/3, 2/3, 1/3, 2/3, 0, 0)`, mask
  `[1,0,0,1,1,0]`, action index `4` (`UP`, `l=2`).

At initialisation `log q = log 1 + log(1/3)`, matching the table.

---

## 5. How `q_phi` learns

### 5.1 Objective

`_update_reverse_policy` runs plain maximum likelihood on the trajectories the
forward policy just produced:

```
L_rev(phi) = - (1/N) * sum_i log q_phi(tau_i | x_i),      tau_i ~ P_F
```

`reverse_train_epochs` (default 4) Adam steps per update, gradient-norm clipped
at `reverse_grad_clip_norm` (default 1.0), learning rate `reverse_lr` (default
1e-3). No rewards, no advantages, no importance weights enter this loss: it is a
pure "given the terminal, predict which path was taken" supervised problem, where
the labels are the actual edges of the sampled trajectories.

### 5.2 What it converges to

Decomposing the population objective over terminals:

```
E_{P_F}[ -log q_phi(tau | x(tau)) ]
      = sum_x P_F(x) * [ H(P_F(.|x)) + KL( P_F(.|x) || q_phi(.|x) ) ]
```

The entropy term is independent of `phi`, so **MLE on forward samples minimises a
`P_F(x)`-weighted forward KL from the true conditional path distribution to
`q_phi`. With enough capacity the optimum is `q_phi(.|x) = P_F(.|x)` for every
terminal with `P_F(x) > 0`** — exactly the condition in Section 3.4 under which
`P_hat` becomes the true propensity `P_F(x)` and the weights become
`R(x)/P_F(x)` with zero within-terminal variance.

So the training loop is: `q_phi` chases the forward policy's own conditional path
distribution, and the better it does so, the lower the variance of the forward
policy's importance weights.

### 5.3 Diagnostics that measure this directly

- `reverse_loss` — the per-path NLL above.
- `reverse_edge_accuracy` — fraction of edges where the argmax reverse action is
  the true one.
- `reverse_edge_entropy` — mean entropy of the masked reverse categorical; starts
  at the uniform-over-parents value and should fall as `q` sharpens.
- **`implied_terminal_within_outcome_std`** — the standard deviation of
  `log[P_F(tau)/q_phi(tau|x)]` computed *among trajectories in the batch that hit
  the same terminal*. This is the cleanest test of Section 3.4: it is zero exactly
  when `q_phi` reproduces the forward conditional on the sampled paths, and it is
  the quantity whose decrease drives the IPS variance down. Plotted in
  `_plot_reverse_training` against the overall std.

---

## 6. From weights to advantages

`learned_reverse_ips_advantages` receives, per rollout: `R_i`, terminal id,
trajectory id, `log P_F(tau_i)`, `log q_phi(tau_i | x_i)`. It validates that
rewards are finite and strictly positive and that both log-probabilities are
`<= 0` (up to `1e-7` numerical slack), then forms `log w_i` in log space
throughout — no probability is ever exponentiated at full scale, which matters
because `P_F(tau)` for `B = 128` is around `e^{-100}`.

Two normalisation modes:

**(a) `batch` (default).** Shift by the batch max, then z-score:

```
s_i = exp(log w_i - max_j log w_j)  in (0, 1]
A_i = (s_i - mean(s)) / (std(s) + eps)
```

Since z-scoring is invariant to a common positive factor, this equals the z-score
of the raw `w_i`; the shift is purely for numerical safety. Same for ESS,
`ESS = (sum s)^2 / sum s^2`, reported as `ips_ess_fraction = ESS / N`. This is
the standard GRPO-style normalisation, but note it **discards the absolute scale
of the weights**: a batch whose weights are uniformly 100x too large produces the
same update magnitude as a nearly-correct batch.

**(b) `running`** (`RunningLogWeightNormalizer`). Keeps EMA first and second
moments of the raw weights, both stored in log space (updated with `logaddexp`,
decay 0.99), and uses the values **from previous batches only**:

```
scale_t   = sqrt( M2_{t-1} )                        # EMA RMS of w
A_i       = clip( w_i/scale_t - M1_{t-1}/scale_t , +/- advantage_clip )
```

with an inner clip of `log(w_i / scale_t)` to `+/- log_ratio_clip` (default 20)
to prevent overflow, and an `advantage_clip` of 10 by default. The first batch
bootstraps from its own moments. This deliberately **retains** the absolute
scale, so a batch containing a 100x error yields a larger correction than one
containing a 2x error, up to the clip. Because the baseline `M1_{t-1}` comes from
past batches and is constant within the current batch, it is a legitimate
policy-gradient baseline: subtracting it does not change the expected gradient.

---

## 7. The forward policy update

`FullTrajectoryPPOTrainer._joint_policy_loss` runs PPO at **path level**, not
token level:

```
rho_i = exp( sum_t log P_F^new(a_t|s_t) - sum_t log P_F^old(a_t|s_t) )
loss  = - mean_i min( rho_i * A_i , clip(rho_i, 1-eps, 1+eps) * A_i )
        - entropy_coef * mean_i (path entropy)
```

The advantage is one scalar per trajectory (copied onto each of its steps), which
is the correct granularity here since `w` is a trajectory-level quantity.

**What this optimises.** With `train_epochs = 1` (the default) the ratio is
identically 1 when the gradient is taken, so the update reduces to
importance-weighted REINFORCE, `-mean_i A_i * grad log P_F(tau_i)`. Ignoring
normalisation constants,

```
E_{P_F}[ w(tau) * grad_theta log P_F(tau) ]
      = Z * sum_tau p*_phi(tau) grad_theta log P_F(tau)
      = - Z * grad_theta KL( p*_phi || P_F^theta )
```

so **the forward update is a stochastic gradient step on the mass-covering KL
from the target trajectory distribution to the policy** — equivalently, a
weighted maximum-likelihood fit of `P_F` to `p*_phi`. Baseline subtraction leaves
this expectation unchanged; the std/RMS division rescales the step size; PPO
clipping acts as a trust region for `train_epochs > 1`.

---

## 8. One training iteration, in order

Per update step (`CountIPSTrainer.train` -> `LearnedReverseIPSTrainer`):

1. `_on_update_start`: set the forward learning rate (optional two-phase
   schedule: `lr` until `lr_decay_after`, then `lr_after_decay`).
2. **Roll out** `num_groups` independent groups of `group_size` episodes from the
   current forward policy (vectorised on device), recording per-action log-probs,
   the terminal, the action sequence, and the reward.
3. **Score** each group with `_group_advantages`: sum the recorded per-step
   log-probs into `log P_F(tau)`; evaluate `log q_phi(tau|x)` with the reverse
   network **as it stands now**; form `log w`; normalise into advantages; store
   the scalar advantage on every step of the episode.
4. **Update the forward policy** (`super().update`): `train_epochs` PPO steps on
   the pooled episodes from all groups, gradient clipped at `grad_clip_norm`.
5. **Then update the reverse policy** (`_update_reverse_policy`):
   `reverse_train_epochs` MLE steps of `q_phi` on the same episodes.
6. Log diagnostics; periodically evaluate by sampling (TV distance to
   `R/Z`, per-terminal probabilities, boundary terminals) and checkpoint.

### Why the ordering in steps 3-5 matters

The `q_phi` used to weight batch `t` was fitted only on batches `1..t-1`, so it
is **statistically independent of batch `t`**. Conditional on the history,
`E[w] = Z` exactly as derived in Section 3.3.

If instead `q_phi` were fitted on batch `t` *before* weighting it, `q_phi` would
be inflated precisely on the paths that happen to appear in that batch, which
inflates exactly those weights — a self-reinforcing bias that rewards a
trajectory for having been sampled. The comment at `update()` records this
intent: "Advantages were computed with the reverse policy frozen before this
batch. Update the forward policy first, then fit q for the next batch."

The price is a one-update lag: `q_phi^{(t)}` is fitted to samples from
`theta^{(t-1)}` but is used to weight samples from `theta^{(t)}`. Since the
forward policy moves slowly (one clipped PPO step), this lag is small, but it is
worth stating explicitly.

---

## 9. Fixed points: what the algorithm can and cannot pin down

Suppose the coupled system reaches a joint stationary point, i.e.
`P_F = p*_phi` (forward converged to its current target) and
`q_phi = P_F(.|x)` (reverse converged to the forward conditional). Substituting
the second into the first:

```
P_F(tau) = R(x) q_phi(tau|x) / Z = R(x) P_F(tau|x) / Z
```

and since `P_F(tau) = P_F(x) * P_F(tau|x)`, this gives

```
P_F(x) = R(x) / Z          for every terminal x
```

which is exactly the goal. Two observations:

- **The terminal marginal is the only thing pinned down.** The conditional path
  distribution `P_F(.|x)` cancels and is left completely free: *any* conditional
  is a fixed point, provided `q_phi` matches it. There is a continuum of fixed
  points, all correct at the terminal level.
- **Consequently nothing in the objective penalises path-space collapse.** The
  forward policy may concentrate on a few paths per terminal with `q_phi`
  faithfully following, which actually *reduces* weight variance while leaving the
  terminal distribution correct. This is the same freedom that a GFlowNet has in
  choosing its backward policy `P_B`. Whether this matters depends on the
  application: it is harmless for terminal-distribution fidelity, but it can hurt
  exploration (and hence coverage of rare, high-reward terminals) during training.
  `trajectory_sampling.png` and the conditional path-entropy diagnostics are there
  to monitor it.

---

## 10. Relation to the surrounding baselines

All of these live in the same folder with the same environment, networks, and
evaluation, which is what makes them comparable:

| method | weight / loss | reverse model |
|---|---|---|
| `count_ips.py` | `R(x) / p_hat(x)` | none (needs repeated terminals) |
| `exact_probability_ips.py` | `R(x) / P_F(tau)` | none (targets `m(x) R(x)`) |
| `trajectory_ips.py` | `R(x) / (m(x) p_hat(tau))` | none, but needs exact `m(x)` (DP oracle) |
| `backward_corrected_ips.py` | `R(x) P_B(tau\|x) / P_F(tau)` | fixed uniform-over-parents |
| **this file** | `R(x) q_phi(tau\|x) / P_F(tau)` | **learned, fitted by MLE to `P_F(.\|x)`** |
| `gflownet.py` | `(log Z + log P_F - log R - log P_B)^2` | fixed uniform; adds scalar `log Z` |

So the learned-reverse method is the `backward_corrected_ips` estimator with the
fixed uniform reverse replaced by a learned one. Since the terminal marginal of
the target is identical in both cases (Section 3.3), the learning of `q` buys
**only variance reduction** — but that reduction is the difference between
weights that vary by orders of magnitude within a single terminal and weights
that are nearly constant.

Compared with a GFlowNet: `q_phi` plays the role of `P_B`, but the training
signal is an importance-weighted policy gradient (PPO) rather than a squared
trajectory-balance residual, and no `log Z` parameter is estimated — the batch
normalisation absorbs `Z`.

---

## 11. Points I would like verified / known caveats

Correctness:

1. The exactness argument is the **trajectory-level** one (3.3): for any
   normalised `q_phi`, the target `p*_phi(tau) ∝ R(x) q_phi(tau|x)` has terminal
   marginal `R(x)/Z`, and `w = R q / P_F` is its importance weight, so
   `E_{P_F}[w] = Z`. The "implied terminal propensity" `P_F(tau)/q_phi(tau|x)` is
   an upward-biased plug-in for `P_F(x)` under on-policy sampling (3.4) — it is a
   useful diagnostic and the object whose variance we minimise, but it is not
   what the correctness argument rests on.
2. Normalisation of `q_phi` per terminal holds by construction for any `phi`
   (4.3), which is why a poorly fitted reverse cannot bias the terminal marginal.
   Supports match exactly, so no mass is missed.
3. The freeze-then-fit ordering (Section 8) is what keeps `q_phi` independent of
   the batch it weights.

Practical caveats:

4. `q_phi`'s fit is weighted by `P_F(x)`, so it is *worst* on rare terminals —
   which are exactly the terminals with the largest weights and the greatest
   influence on the update. Early in training, when the forward policy is far from
   `R/Z`, this is the main source of weight variance.
5. Batch z-scoring discards the absolute weight scale (a uniformly-mis-scaled
   batch still yields a unit-scale update); the `running` mode exists to retain
   it. This is a genuine change in the update semantics, not just numerics.
6. Clipping appears in three places (`log_ratio_clip`, `advantage_clip`, PPO
   `clip_ratio`). Each breaks the exact gradient identity of Section 7 when
   saturated, so `running_advantage_clip_fraction` and `clip_fraction` should be
   monitored.
7. The target `p*_phi` moves as `phi` moves — a non-stationary objective for the
   forward policy — plus the one-update lag noted in Section 8. Only the terminal
   marginal of the target is invariant.
8. The path-space degree of freedom is unconstrained (Section 9); path-entropy
   collapse is not penalised.
9. `q_phi` sees only 6 smooth, `B`-scaled features and shares parameters across
   all `(u, x)` pairs, so it is an approximation with possible aliasing. This
   affects variance only, never validity.
10. With `num_groups > 1` the running normaliser's EMA is updated once *per
    group*, i.e. `num_groups` times per update step.

## 12. Default hyperparameters (CLI)

| flag | default | meaning |
|---|---|---|
| `--budget` | 128 | `B`; frontier has `B+1` terminals |
| `--max-step` | 3 | `K`; action lengths `1..K` |
| `--group-size` / `--num-groups` | 16 / 1 | rollouts per independently normalised group / groups per update |
| `--num-updates` | 2000 | training iterations |
| `--lr`, `--lr-decay-after`, `--lr-after-decay` | 3e-4, none | forward LR and optional two-phase schedule |
| `--clip-ratio`, `--entropy-coef` | 0.2, 0.0 | PPO trust region, entropy bonus |
| `--reverse-lr` | 1e-3 | `q_phi` learning rate |
| `--reverse-hidden-size` / `--reverse-num-layers` | 128 / 2 | `q_phi` MLP |
| `--reverse-train-epochs` | 4 | MLE steps per update |
| `--reverse-grad-clip-norm` | 1.0 | `q_phi` gradient clip |
| `--advantage-normalization` | `batch` | `batch` z-score or `running` EMA scale |
| `--running-scale-decay` / `--running-advantage-clip` / `--running-log-ratio-clip` | 0.99 / 10 / 20 | running-mode settings |

Primary success metric: `tv_reward_target`, the total-variation distance between
the sampled terminal distribution and `R(x)/Z`; it is zero only when the terminal
marginal is exactly reward-proportional.
