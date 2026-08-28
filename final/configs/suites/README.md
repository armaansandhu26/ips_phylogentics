# Comparison suite configs

Each JSON file defines one **paper row**: same taxa count, dataset, training budget, and sampling size for all four methods.

## Fields

| Field | Meaning |
|-------|---------|
| `id` | Suite identifier used in `final/runs/<id>/` |
| `taxa` | Number of taxa (5, 10, or 27) |
| `dataset` | Path relative to repo root |
| `log_score_shift` | Reward shift used at sampling/plot time |
| `training` | Shared epochs, batch sizes, replay settings |
| `sampling` | Post-train 1M eval defaults |
| `methods.*.cfg` | Model yaml path (may differ for PhyloGFN) |

## Adding a new suite

1. Copy an existing JSON and edit `id`, taxa, dataset, and training block.
2. Point each method's `cfg` at the correct yaml.
3. Run: `python -m final run_suite --suite <new_id>`

## PhyloGFN backend

Default `"backend": "og_code"` uses shifted reward (comparable to GRPO family).

For upstream paper reward, set:

```json
"phylgfn": {
  "backend": "paper",
  "cfg": "phylogfn_paper/src/configs/.../cfg_ds1_paper_27taxa_g1024_noreplay.yaml"
}
```

The checked-in paper-faithful DS1 suites are
`27taxa_phylogfn_paper_40pct_seed{0,1,2}`. They use the official continuous-branch
configuration unchanged: 500 epochs, 200 updates per epoch, 64 on-policy plus
64 replay trajectories per update (12.8M total trajectories). Run only its
`phylgfn` method; it is intentionally separate from the shifted-reward,
no-replay comparison suites. After training, the pipeline uses the final
checkpoint for both the paper-style terminal-density estimate and a 100,000
forward-trajectory pathwise diagnostic.
