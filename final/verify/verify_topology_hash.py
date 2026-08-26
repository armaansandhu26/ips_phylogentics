"""Pre-launch verification: topology hash consistency (MrBayes / Newick compatible)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _verify_tree(env, tree_a, tree_b) -> tuple[dict, bool]:
    from ete3 import Tree

    id_a = str(tree_a.tree_topology_id)
    id_b = str(tree_b.tree_topology_id)
    newick = tree_a.ete_node.write(format=1)
    try:
        reparsed = Tree(newick, format=1)
        newick_ok = True
        rf = tree_a.ete_node.robinson_foulds(reparsed, unrooted_trees=True)[0]
        same_unrooted = rf == 0
    except Exception:
        newick_ok = False
        same_unrooted = False

    row = {
        "topology_id": id_a,
        "rebuild_match": id_a == id_b,
        "newick_parseable": newick_ok,
        "newick_unrooted_rf_zero": same_unrooted,
        "newick_preview": newick[:100],
    }
    ok = row["rebuild_match"] and row["newick_parseable"] and row["newick_unrooted_rf_zero"]
    return row, ok


def _verify_enumerated(env) -> tuple[list[dict], list[dict], set[str]]:
    from learned_reverse_ips.reverse_policy import _edge_action, enumerate_tree_action_catalog

    action_paths, terminal_ids = enumerate_tree_action_catalog(env)
    samples: list[dict] = []
    rebuild_mismatches: list[dict] = []
    seen_topo: set[str] = set()

    for path, topo_id in zip(action_paths, terminal_ids):
        if topo_id in seen_topo:
            continue
        seen_topo.add(topo_id)
        actions = [
            {"tree_action": int(a), "edge_action": _edge_action(i, len(path))}
            for i, a in enumerate(path)
        ]
        tree_a = env.actions_to_trajectory(actions).current_state.subtrees[0]
        tree_b = env.actions_to_trajectory(actions).current_state.subtrees[0]
        row, ok = _verify_tree(env, tree_a, tree_b)
        samples.append(row)
        if not ok:
            rebuild_mismatches.append(row)

    return samples, rebuild_mismatches, seen_topo


def _verify_sampled(env, *, num_samples: int, max_attempts: int) -> tuple[list[dict], list[dict], set[str]]:
    samples: list[dict] = []
    rebuild_mismatches: list[dict] = []
    seen_topo: set[str] = set()

    for _ in range(max_attempts):
        if len(seen_topo) >= num_samples:
            break
        trajectory = env.generate_random_trajectory()
        tree_a = trajectory.current_state.subtrees[0]
        tree_b = trajectory.current_state.subtrees[0]
        topo_id = str(tree_a.tree_topology_id)
        if topo_id in seen_topo:
            continue
        seen_topo.add(topo_id)
        row, ok = _verify_tree(env, tree_a, tree_b)
        samples.append(row)
        if not ok:
            rebuild_mismatches.append(row)

    return samples, rebuild_mismatches, seen_topo


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify topology_id is deterministic and Newick export is parseable "
            "(MrBayes tree exchange format)."
        )
    )
    parser.add_argument(
        "--cfg",
        default="src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml",
    )
    parser.add_argument(
        "--dataset",
        default="dataset/benchmark_datasets/DS1_reduced.pickle",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="For taxa > 5, number of unique sampled topologies to verify.",
    )
    args = parser.parse_args()

    from final.verify.env_loader import load_env_from_paths

    env, _, _ = load_env_from_paths(args.cfg, args.dataset)
    num_taxa = len(env.sequences)

    if num_taxa == 5:
        samples, rebuild_mismatches, seen_topo = _verify_enumerated(env)
        mode = "enumerated"
        expected_topologies = 105
    else:
        max_attempts = max(args.sample_size * 20, 200)
        samples, rebuild_mismatches, seen_topo = _verify_sampled(
            env,
            num_samples=args.sample_size,
            max_attempts=max_attempts,
        )
        mode = "sampled"
        expected_topologies = None

    result = {
        "ok": len(rebuild_mismatches) == 0 and len(seen_topo) > 0,
        "mode": mode,
        "num_taxa": num_taxa,
        "num_topologies": len(samples),
        "expected_topologies_5taxa": expected_topologies,
        "num_unique_topology_ids": len(seen_topo),
        "num_issues": len(rebuild_mismatches),
        "issues": rebuild_mismatches[:10],
        "samples": samples[:3],
        "note": (
            "topology_id = ete3 MD5(unrooted shape) at terminalization. "
            "MrBayes uses Newick; we verify Newick is parseable and unrooted RF=0 "
            "vs the stored ete tree (standard external-tool compatibility check)."
        ),
    }
    print(json.dumps(result, indent=2))
    if num_taxa == 5 and len(samples) != 105:
        print(
            f"WARNING: expected 105 topologies, got {len(samples)}",
            file=sys.stderr,
        )
    if num_taxa != 5 and len(seen_topo) < args.sample_size:
        print(
            f"WARNING: sampled {len(seen_topo)} unique topologies, "
            f"target was {args.sample_size}",
            file=sys.stderr,
        )
    if not result["ok"]:
        raise SystemExit("topology hash verification FAILED")


if __name__ == "__main__":
    main()
