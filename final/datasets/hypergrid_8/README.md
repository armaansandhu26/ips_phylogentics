# Hyper-Grid toy dataset (8^2)

Precomputed terminal rewards for the Trajectory Balance Hyper-Grid task.

## Files

| File | Description |
|------|-------------|
| `rewards.npy` | `8x8` float32 reward grid |
| `target_distribution.npz` | `rewards` and `probs` where `probs ∝ rewards` |
| `meta.json` | Environment parameters and summary stats |

## Reward

```
R(x) = R0 + R1 * prod_d I(|x_d/(H-1) - 0.5| in (0.25, 0.5])
     + R2 * prod_d I(|x_d/(H-1) - 0.5| in (0.3, 0.4))
```

With R0=0.1, R1=0.5, R2=2.0.

## Regenerate

```bash
python -m final.toy.build_dataset --H 8 --D 2
python -m final.toy.plot_reward --dataset /home/armaan/phylogfn/final/datasets/hypergrid_8
```
