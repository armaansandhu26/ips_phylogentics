create this tree for evaluating experiments:
x-axis: log probability assigned by the sampler
y-axis: true unnormalized log posterior score of that tree

next steps to verify / implement for final pearson correlation plot:

1. [verified] verify what x-axis means:
   - use log q(tree), not just log q(path)
   - confirm we estimate tree probability by sampling backward trajectories and using
     log q(tree) ~= logmeanexp(log_pf - log_pb)
   - conclusion: for the final Pearson plot, x-axis should be estimated log q(tree)

2. [verified] verify existing phylogfn code path works for grpo-trained checkpoints:
   - load generator from a grpo_experiments run
   - confirm env.sample_backward_from_tree(tree) works on sampled trees
   - confirm rollout replay on fixed action sequences returns log_paths_pf and log_paths_pb
   - smoke test script added: grpo_experiments/scripts/smoke_test_tree_logq.py
   - verified on IPS-GRPO ablation checkpoint:
     grpo_experiments/runs/ips_replay_ablation/topo/20260603_115451_ablation_hyb_ips_pfloor_005_hybrid_ips_grpo
   - medium smoke test result:
     sample_trees=20, backward_trajectories=32, repeat_estimates=2
   - result: smoke_test_passed=True, tree_level_pearson_r_smoke=0.7536
   - note: this verifies compatibility, not final benchmark quality

3. [verified] implement a reusable helper:
   - estimate_tree_log_q(generator, env, rollout_worker, tree, n_backward_samples)
   - inputs: final tree, trained checkpoint, number of backward samples
   - outputs: estimated log q(tree), true log_score, optional stderr / diagnostics
   - current status: shared helper implemented in grpo_experiments/eval_utils.py
   - smoke test script now reuses the shared helper:
     grpo_experiments/scripts/smoke_test_tree_logq.py

4. [verified] define the evaluation tree set:
   - option A (paper-faithful reference): use trees generated from VBPI-GNN with
     0% / 30% / 50% random action insertion during sequential construction
        - note: this is what the original PhyloGFN paper used for the Pearson table/plot
        - rationale: it gives a shared external tree set covering high / medium / low posterior
            regions, and it specifically stress-tests whether the sampler models tree space beyond
            the narrow high-probability region explored by VBPI-GNN
        - for this repo we will use option B below unless we later decide we need paper-faithful
            reproduction
   - option B (chosen for this repo): build one fixed shared benchmark tree set from a pooled
     candidate set sampled from strong reference checkpoints in this repo
        - initial candidate sources:
            - PhyloGFN baseline used in the same ablation comparisons:
              grpo_experiments/runs/ips_replay_ablation/topo/20260603_112929_ablation_phylgfn_r64_phylgfn
            - hybrid IPS-GRPO checkpoint: pfloor=0.005
              grpo_experiments/runs/ips_replay_ablation/topo/20260603_115451_ablation_hyb_ips_pfloor_005_hybrid_ips_grpo
            - hybrid IPS-GRPO checkpoint: pfloor=0.002
              grpo_experiments/runs/ips_replay_ablation/topo/20260603_115912_ablation_hyb_ips_pfloor_002_hybrid_ips_grpo
        - initial pool construction plan:
            - sample ~1000 trees from each source checkpoint
            - merge all sampled trees into one candidate pool
            - store topology id, signature, source run, and true log_score for each tree
        - deduplication plan:
            - for DS1 / 5-species toy setting: use signature dedup for the main benchmark
            - keep topology id as metadata for secondary analysis
            - for larger datasets, topology-level dedup is likely the better primary benchmark
        - stratification plan:
            - sort deduplicated trees by true log_score
            - split into three score bands: high / medium / low posterior region
            - default split: score quantiles (top third / middle third / bottom third)
        - initial final benchmark target:
            - sample 100 topology-unique trees from each band
            - total benchmark size = 300 trees
        - benchmarking rule:
            - every compared method must use the exact same frozen tree set
            - do not let each method evaluate on its own sampled trees
        - artifact plan:
            - save the candidate pool and final frozen benchmark to disk so future evals reuse the same set
        - implementation status:
            - benchmark builder added:
              grpo_experiments/scripts/build_tree_eval_benchmark.py
            - dry run verified on the chosen three topology checkpoints
            - full benchmark built for DS1 / r64 with signature dedup:
              grpo_experiments/eval_benchmarks/signature_pooled_ds1_r64/benchmark.json
            - companion artifacts:
              grpo_experiments/eval_benchmarks/signature_pooled_ds1_r64/candidate_pool.json
              grpo_experiments/eval_benchmarks/signature_pooled_ds1_r64/summary.json

5. decide evaluation regimes:
   - one overall plot first
   - then optional subsets analogous to 0% / 30% / 50% random if needed
   - keep the same tree subsets for all compared methods

6. [verified] implement eval script in grpo_experiments/scripts:
   - load run checkpoint
   - load or build evaluation tree set
   - compute (log q(tree), log_score(tree)) for each tree
   - save raw pairs to json/csv/npz
   - implementation status:
     - Pearson eval script added:
       grpo_experiments/scripts/eval_tree_logq_pearson.py
     - script loads benchmark.json, reconstructs each frozen tree from saved actions,
       estimates log q(tree), and writes raw pairs + summary JSON
     - tiny validation run completed successfully on test benchmark

7. [verified] implement plotting:
   - scatter plot: x = log q(tree), y = true unnormalized log posterior
   - compute pearson r on the same pairs
   - annotate/save pearson value
   - optionally make one panel per evaluation regime
   - current status:
     - overall scatter and per-band scatter panels are already produced by
       grpo_experiments/scripts/eval_tree_logq_pearson.py
     - full Pearson eval completed on the frozen DS1 benchmark for:
       - phylgfn_r64
       - hyb_ips_p005
       - hyb_ips_p002
     - outputs saved under:
       grpo_experiments/eval_benchmarks/signature_pooled_ds1_r64/pearson_eval
     - summary JSON:
       grpo_experiments/eval_benchmarks/signature_pooled_ds1_r64/pearson_eval/summary.json
     - headline DS1 / r64 results (signature-dedup benchmark, 300 trees, 200 backward trajectories):
       - phylgfn_r64 overall Pearson = 0.9103
       - hyb_ips_p005 overall Pearson = 0.0144
       - hyb_ips_p002 overall Pearson = 0.0175
     - by-band Pearson:
       - phylgfn_r64: low=0.8891, medium=0.7520, high=0.9704
       - hyb_ips_p005: low=-0.2709, medium=0.0265, high=0.5580
       - hyb_ips_p002: low=-0.2760, medium=0.0666, high=0.5588

8. verify numerical behavior:
   - choose enough backward trajectories per tree for stable estimates
   - check variance / stderr of estimated log q(tree)
   - confirm no obvious bias from too few backward samples

9. [partly done] compare methods fairly:
   - run the same eval tree set for phylgfn and grpo variants
   - use same number of backward samples per tree
   - report pearson table and scatter plots side by side
   - current status:
     - same frozen benchmark used for PhyloGFN, hyb_ips p005, and hyb_ips p002
     - same backward-trajectory count used for all compared methods
     - summary + plots produced for the first DS1 / r64 comparison
     - topology-collapsed diagnostic added:
       grpo_experiments/scripts/topology_collapse_pearson.py
     - topology-collapsed outputs saved under:
       grpo_experiments/eval_benchmarks/signature_pooled_ds1_r64/pearson_eval/topology_collapsed
     - topology-collapsed overall Pearson (mean collapsed log q, mean collapsed score):
       - phylgfn_r64 = 0.9811
       - hyb_ips_p005 = -0.5198
       - hyb_ips_p002 = -0.4626
     - topology-collapsed high / medium / low:
       - phylgfn_r64: low=0.9794, medium=n/a, high=0.9863
       - hyb_ips_p005: low=-0.5859, medium=n/a, high=0.4422
       - hyb_ips_p002: low=-0.5027, medium=n/a, high=0.4262
     - interpretation:
       - this does not look like only a signature-level artifact on DS1
       - even after collapsing to topology level, PhyloGFN stays strongly aligned
         while the two hyb_ips checkpoints remain misaligned overall
       - current simple takeaway: hyb_ips can still sample good trees, but its assigned
         probability mass is not aligning with posterior quality the way PhyloGFN does

10. optional extensions:
   - compare topology-level vs signature-level evaluation sets
   - store the evaluation tree set once so future runs reuse the exact same benchmark



------
p - 10^-6
p - 0.002
r = 128
steps = 1L
batch size = 1024 (batch size -> confirm with ablation as well once)