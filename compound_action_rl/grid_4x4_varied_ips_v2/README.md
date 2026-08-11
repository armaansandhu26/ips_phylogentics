# IPS-GRPO v2 — 4×4 grid with spatial red/green variation

Extension of `grid_3x3_varied_ips_v2` to the **4×4 compound-action grid** (1280 trajectories).

## Scale vs 3×3

| | 3×3 (`grid_3x3_varied_ips_v2`) | **4×4 (this)** |
|---|-------------------------------|----------------|
| Grid | 3×3 | **4×4** |
| Minimal paths | 6 | **20** |
| Steps / episode | 4 | **6** |
| Trajectories | 96 | **1,280** |
| Obs dim | 33 | **56** |
| Default `group_size` | 128 | **256** |
| Default `max_inverse_weight` | 1280 | **2560** |

Same IPS-GRPO v2 machinery: exact propensities, split per-model losses, trajectory color credit, detached rep + position aux head, log π vs log R eval.

## Environment

- Start `(0,0)` → goal `(3,3)`, **6 steps** (up/right + red/green each step).
- Terminal reward: Gaussian red/green fields on painted cells, normalized.
- **1,280 outcomes** = 20 paths × 2⁶ color sequences (still enumerable for toy eval).

## Color profiles (`default`)

| Profile | red_center | green_center | T | reward range (approx) |
|---------|------------|--------------|---|------------------------|
| `default` | (2, 0) | (3, 3) | 2.0 | [0.005, 0.655] |
| `swapped` | (3, 3) | (2, 0) | 2.0 | [0.005, 0.655] |
| `split_corners` | (0, 3) | (3, 0) | 2.0 | [0.005, 0.424] |
| `center_green` | (0, 0) | (2, 2) | 1.5 | [0.018, 0.569] |

## Run

```bash
cd grid_4x4_varied_ips_v2

python run_experiment.py \
  --color-profile default \
  --group-size 256 \
  --num-updates 5000 \
  --eval-every 500 \
  --eval-episodes 10000 \
  --plot-episodes 25000 \
  --log-every 50
```

### Recommended starting hyperparams

- **`group_size=256`** (or 512) — batch must cover enough of 1280 outcomes; exact IPS helps but larger groups stabilize SNIPS.
- **`eval-every 500`** — watch `hit/1280` early; collapse signature is `hit << 1280` by eval@500.
- **`plot-episodes 25000+`** — rare trajectories need more samples for clean density scatter (log scatter is primary at scale).

### Metrics to watch

1. **Primary:** `hit / 1280` during eval (want high coverage, not just good R² on a subset)
2. **Secondary:** `log_slope → 1.0` (proportional sampling)
3. **Tertiary:** density R² (noisy until enough plot episodes)

## Artifacts

Same layout as 3×3 v2: `config.json`, `train.log`, `history.json`, `checkpoint.pt`, `summary.json`, `training_curves.png`, `sampling_scatter.png`, `log_scatter.png`.

## Architecture

```text
obs (56-d) → Model 1 → move + state_rep (128-d) + position aux
            → Model 2(detached rep, move) → color
            → env.step (×6) → terminal reward
```

Trajectory probability for IPS:
```
log p_θ(τ) = Σ_{t=1}^{6} [log π_path(move_t) + log π_color(color_t)]
weight = exp(-log p_θ(τ))  →  SNIPS normalize  →  × return  →  advantage
```
