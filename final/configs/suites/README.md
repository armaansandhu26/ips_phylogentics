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
