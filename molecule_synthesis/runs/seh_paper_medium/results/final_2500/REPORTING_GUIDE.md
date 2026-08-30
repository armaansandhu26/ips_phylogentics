# How to report the final sEH experiment

## What the original RGFN paper emphasizes

The original [RGFN paper](https://papers.nips.cc/paper_files/paper/2024/file/53704142f230054140418ecd8857f391-Paper-Conference.pdf)
does not make its case from average reward alone. Its main evaluation pairs:

1. final reward distributions (Figure 3);
2. discovery of diverse high-reward modes as a function of normalized
   iterations/oracle calls (Figure 4);
3. top-mode chemical properties and synthesizability checks (Table 1).

For sEH, it defines a mode as a molecule with proxy > 7 whose Tanimoto
similarity to every previously selected leader is at most 0.5. The code in this
repository uses the same threshold and greedy leader rule. The paper's central
framing is a quality-diversity tradeoff under a synthesis-constrained action
space: reward and mode discovery are reported together.

That framing is especially important here. A collapsed policy can have a
reasonable mean proxy while assigning nearly all mass to a few molecules.
Final-policy uniqueness, high-reward mode count, and scaffold count therefore
belong beside mean proxy in the main result.

## Recommended results paragraph

> We evaluated all methods on the full reaction-based sEH environment using
> three seeds, 2,500 training updates (250,000 forward trajectories), and
> 50,000 independent samples from each final policy. MIPS-GRPO maintained
> 93.59 ± 2.43% unique valid samples, compared with 74.54 ± 3.99% for RGFN,
> while GRPO and Count IPS-GRPO collapsed to 0.011 ± 0.010% and
> 0.173 ± 0.286% uniqueness, respectively. MIPS-GRPO and RGFN achieved similar
> mean sampled sEH proxy values (4.281 ± 0.177 and 4.227 ± 0.421), but
> MIPS-GRPO found more diverse high-reward modes on average (22.3 ± 17.0 versus
> 14.0 ± 9.2). Neither GRPO nor Count IPS-GRPO found a mode above the sEH
> threshold in any seed. These results show that the learned-reverse correction
> prevents the severe final-policy collapse observed with the two GRPO
> baselines while retaining reward quality comparable to RGFN. Mode counts were
> variable across seeds, so the apparent MIPS advantage over RGFN in mode
> discovery should be interpreted as suggestive rather than statistically
> conclusive.

All values above are mean ± sample standard deviation. With only three seeds,
avoid “statistically significant,” “state of the art,” or other inferential
language unless additional runs or a prespecified statistical analysis are
added.

## Figure caption

**Full-space sEH final-policy comparison.** Each method was trained for 2,500
updates (250,000 forward trajectories) and evaluated with 50,000 independent
samples for each of three seeds. Bars show the arithmetic mean across seeds,
whiskers show ±1 sample standard deviation, and circles show individual seeds.
Modes have sEH proxy > 7 and maximum Tanimoto similarity ≤ 0.5 under greedy
leader selection. MIPS-GRPO and RGFN retain broad support and discover
high-reward modes, whereas GRPO and Count IPS-GRPO show severe final-policy
collapse. The lower uniqueness whisker is clipped to the display floor where
necessary because that panel uses a log scale.

## Claims to make—and not make

- Main claim: MIPS-GRPO prevents the collapse seen in GRPO and count IPS-GRPO
  and retains much broader final-policy support than RGFN in all three seeds.
- Supported secondary claim: MIPS-GRPO has a higher mean mode count than RGFN
  and wins on mode count in two of three paired seeds.
- Quality claim: MIPS-GRPO and RGFN have similar mean sampled proxy; the
  three-seed evidence does not establish that either is better on this metric.
- Do not use a best-seed mean proxy as the headline result. Count IPS-GRPO's
  selected seed has a mean proxy of 5.27 despite producing only two unique
  molecules and zero modes, which illustrates why reward alone is misleading.
- Do not claim exact reward-proportional sampling. The full chemical space has
  no enumerated target distribution here, and the stored importance-weight ESS
  values are very small.
- The action space guarantees an explicit synthesis route under the shared
  reaction grammar, but this experiment does not independently validate
  laboratory yield, retrosynthesis success, or target binding.

## Artifact map

- `table1_all_seeds.md`: compact paper table.
- `table1_all_seeds.csv`: machine-readable aggregate values.
- `per_seed_results.csv`: all 12 final observations.
- `best_runs.csv`: retained as a diagnostic only; it is not used in the figure.
- `figure_best_runs.pdf`: vector figure for the paper.
- `figure_best_runs.png`: preview/raster figure.

The historical `figure_best_runs` filename is kept for stable downstream links;
the current figure displays three-seed means rather than selected runs.

The [RGFN repository](https://github.com/koziarskilab/RGFN) is the primary
implementation reference for the reaction environment and released sEH setup.
