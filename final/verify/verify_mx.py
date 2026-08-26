"""Pre-launch verification: m(x) trajectory counts at 5 taxa."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final.logging.precompute import precompute_for_suite, verify_mx_consistency  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify m(x) at 5 taxa: 180 trajectories, 105 topologies, m(x) in [8,48]."
    )
    parser.add_argument("--suite", default="5taxa_noreplay")
    parser.add_argument(
        "--cfg",
        default="src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml",
    )
    parser.add_argument(
        "--dataset",
        default="dataset/benchmark_datasets/DS1_reduced.pickle",
    )
    args = parser.parse_args()

    out_dir = precompute_for_suite(args.suite, cfg_rel=args.cfg, dataset_rel=args.dataset)
    result = verify_mx_consistency(out_dir)
    result["precomputed_dir"] = str(out_dir)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit("m(x) verification FAILED")
    print("OK: 180 trajectories, 105 topologies, brute-force m(x) verified")


if __name__ == "__main__":
    main()
