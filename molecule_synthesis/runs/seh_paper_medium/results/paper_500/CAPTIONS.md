# Suggested captions

## Figure 1

**Learned reverse correction prevents early final-policy collapse on the
full-space sEH synthesis task.** All methods received 50,000 training oracle
calls (500 updates × 100 forward trajectories) and were evaluated using 5,000
independent samples from the saved checkpoint. (a) Fraction of valid samples
with a distinct SMILES and (b) empirical probability mass assigned to the most
frequent molecule; logarithmic axes expose the separation between collapsed
and non-collapsed policies. (c) Mean sampled sEH proxy. The comparatively high
Count IPS-GRPO mean is carried by a nearly deterministic policy and therefore
does not represent diverse discovery. (d) Greedy leader-mode count among
unique molecules with proxy > 7; parenthetical labels give the unclustered
number of threshold-crossing candidates. Leader modes have maximum pairwise
Morgan-fingerprint Tanimoto similarity ≤ 0.5. Results are from seed 0 and do
not quantify run-to-run uncertainty.

## Table 1

**Final-policy statistics after 500 updates on full-space sEH.** Each method
was evaluated with 5,000 independent checkpoint samples. Unique fractions are
computed among valid SMILES. Top-1 mass is the empirical frequency of the most
common molecule. A run is labelled collapsed when almost all final-policy mass
is concentrated on one or a few molecules. This is an interim seed-0 result.

## Figure S1

**Sampling convergence for the two non-collapsed policies at 500 updates.**
Distinct SMILES accumulation and cumulative mean sEH proxy are shown as a
function of final-policy samples drawn. GRPO and Count IPS-GRPO are omitted
because their collapse is reported directly in Figure 1 and Table 1.
