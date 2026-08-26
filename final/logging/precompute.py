"""Precompute exact m(x) and terminal catalogs for small environments."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from final.paths import FINAL_ROOT, REPO_ROOT


def enumerate_phylo_terminals(
    cfg_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    from final.verify.env_loader import load_env_from_paths
    from learned_reverse_ips.reverse_policy import _edge_action

    env, cfg, sequences = load_env_from_paths(cfg_path, dataset_path)
    num_taxa = len(sequences)
    action_ranges = [
        range(n * (n - 1) // 2) for n in range(num_taxa, 1, -1)
    ]

    trajectories: list[dict[str, Any]] = []
    signature_counts: dict[str, int] = {}
    topology_counts: dict[str, int] = {}
    mx_by_signature: dict[str, list[int]] = {}

    for action_path in itertools.product(*action_ranges):
        actions = [
            {
                "tree_action": int(tree_action),
                "edge_action": _edge_action(step, len(action_path)),
            }
            for step, tree_action in enumerate(action_path)
        ]
        traj = env.actions_to_trajectory(actions)
        tree = traj.current_state.subtrees[0]
        topology_id = str(tree.tree_topology_id)
        signature = str(tree.signature)
        sig_hash = hashlib.sha256(signature.encode()).hexdigest()[:16]
        trajectories.append(
            {
                "action_path": list(map(int, action_path)),
                "topology_id": topology_id,
                "signature": signature,
                "signature_hash": sig_hash,
                "log_score": float(tree.log_score),
            }
        )
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        topology_counts[topology_id] = topology_counts.get(topology_id, 0) + 1

    return {
        "num_taxa": num_taxa,
        "num_trajectories": len(trajectories),
        "num_topologies": len(topology_counts),
        "num_signatures": len(signature_counts),
        "topology_counts": topology_counts,
        "signature_counts": signature_counts,
        "trajectories": trajectories,
        "cfg_path": str(cfg_path),
        "dataset_path": str(dataset_path),
    }


def precompute_for_suite(suite_id: str, *, cfg_rel: str, dataset_rel: str) -> Path:
    cfg_path = (REPO_ROOT / cfg_rel).resolve()
    dataset_path = (REPO_ROOT / dataset_rel).resolve()
    out_dir = FINAL_ROOT / "precomputed" / suite_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = enumerate_phylo_terminals(cfg_path, dataset_path)
    out_path = out_dir / "terminals.json"
    # Store summary separately from full trajectory list for size
    summary = {k: v for k, v in payload.items() if k != "trajectories"}
    summary["trajectory_sample"] = payload["trajectories"][:5]
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    traj_path = out_dir / "trajectories.jsonl"
    with traj_path.open("w", encoding="utf-8") as handle:
        for row in payload["trajectories"]:
            handle.write(json.dumps(row) + "\n")
    mx_path = out_dir / "mx_exact.json"
    mx_path.write_text(
        json.dumps(
            {
                "topology_counts": payload["topology_counts"],
                "signature_counts": payload["signature_counts"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_dir


def verify_mx_consistency(precomputed_dir: Path) -> dict[str, Any]:
    mx_path = precomputed_dir / "mx_exact.json"
    data = json.loads(mx_path.read_text(encoding="utf-8"))
    topo = data["topology_counts"]
    values = [int(v) for v in topo.values()]
    total = sum(values)
    return {
        "ok": total == 180 and len(topo) == 105 and min(values) >= 1,
        "num_topologies": len(topo),
        "num_trajectories": total,
        "mx_min": min(values) if values else None,
        "mx_max": max(values) if values else None,
        "expected": {"trajectories": 180, "topologies": 105},
    }
