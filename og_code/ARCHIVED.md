# og_code experiments archived

Training run outputs were moved off disk on **2026-08-13** to free space.

| What | Location |
|------|----------|
| **Archive** | `../_archives/og_code__experiments.tar.gz` (~778M) |
| **Manifest** | `../_archives/og_code__experiments.MANIFEST.txt` |
| **Comparison plots** | `grpo_experiments/comparisons/{5,10,27}taxa/gflownet/` |

This folder still contains the **modified og_code source** (`src/`, `train.py`, configs) for the shifted-reward GFlowNet baseline. New paper-faithful runs use `../phylogfn_paper/`.

## Restore a run

```bash
cd /home/armaan/phylogfn/og_code
tar -xzf ../_archives/og_code__experiments.tar.gz experiments/full_model/<RUN_DIR>
```

## Resume archived run (example: replay shift0 @ epoch 9999)

```bash
cd /home/armaan/phylogfn/og_code
tar -xzf ../_archives/og_code__experiments.tar.gz \
  experiments/full_model/20260811_161709_phylgfn_logreward_27taxa_g819_r205_replay_shift0
CUDA_VISIBLE_DEVICES=3 /home/armaan/phylogfn/.venv/bin/python -u train.py resume \
  experiments/full_model/20260811_161709_phylgfn_logreward_27taxa_g819_r205_replay_shift0 \
  dataset/benchmark_datasets/DS1.pickle \
  experiments/full_model/phylgfn_27taxa_replay_shift0_resumed \
  --nb_device 1
```

Note: some `grpo_experiments/scripts/plot_*` scripts still reference original `og_code/experiments/.../*.npz` paths for replotting; extract the relevant run from the archive first.
