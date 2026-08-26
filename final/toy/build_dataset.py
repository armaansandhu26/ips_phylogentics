"""Build and write a precomputed Hyper-Grid toy dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from final.paths import FINAL_ROOT
from final.toy.hypergrid import HyperGridSpec, build_reward_grid, summarize_reward_grid, target_distribution

DEFAULT_OUT_DIR = FINAL_ROOT / "datasets" / "hypergrid_4096"


def build_hypergrid_dataset(
    *,
    out_dir: Path,
    spec: HyperGridSpec,
    write_target: bool = True,
) -> Path:
    spec.validate()
    out_dir.mkdir(parents=True, exist_ok=True)

    rewards = build_reward_grid(spec)
    np.save(out_dir / "rewards.npy", rewards)

    summary = summarize_reward_grid(rewards, spec)
    meta = {
        **spec.to_dict(),
        **summary,
        "dataset_kind": "hypergrid",
        "coordinate_domain": {"min": 0, "max": spec.H - 1},
        "action_space": {
            "increment_coordinate": list(range(spec.D)),
            "terminate": "stop and receive R(x)",
        },
        "initial_state": [0] * spec.D,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if write_target:
        probs = target_distribution(rewards)
        np.savez_compressed(
            out_dir / "target_distribution.npz",
            rewards=rewards,
            probs=probs.astype(np.float32),
        )

    readme = f"""# Hyper-Grid toy dataset ({spec.H}^{spec.D})

Precomputed terminal rewards for the Trajectory Balance Hyper-Grid task.

## Files

| File | Description |
|------|-------------|
| `rewards.npy` | `{spec.H}x{spec.H}` float32 reward grid |
| `target_distribution.npz` | `rewards` and `probs` where `probs ∝ rewards` |
| `meta.json` | Environment parameters and summary stats |

## Reward

```
R(x) = R0 + R1 * prod_d I(|x_d/(H-1) - 0.5| in ({spec.outer_lo}, {spec.outer_hi}])
     + R2 * prod_d I(|x_d/(H-1) - 0.5| in ({spec.inner_lo}, {spec.inner_hi}))
```

With R0={spec.R0}, R1={spec.R1}, R2={spec.R2}.

## Regenerate

```bash
python -m final.toy.build_dataset --H {spec.H} --D {spec.D}
python -m final.toy.plot_reward --dataset {out_dir}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a Hyper-Grid toy dataset.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--H", type=int, default=4096, help="grid side length")
    parser.add_argument("--D", type=int, default=2, help="number of dimensions")
    parser.add_argument("--R0", type=float, default=0.1)
    parser.add_argument("--R1", type=float, default=0.5)
    parser.add_argument("--R2", type=float, default=2.0)
    parser.add_argument("--no-target", action="store_true")
    args = parser.parse_args(argv)

    spec = HyperGridSpec(H=args.H, D=args.D, R0=args.R0, R1=args.R1, R2=args.R2)
    out_dir = build_hypergrid_dataset(
        out_dir=args.out_dir.resolve(),
        spec=spec,
        write_target=not args.no_target,
    )
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    print(json.dumps({"out_dir": str(out_dir), **meta}, indent=2))


if __name__ == "__main__":
    main()
