"""Enumerate the MiniChem terminal space and its exact reward target."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from .config import REPO_ROOT
from .upstream import resolve_rgfn_root, validate_rgfn_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rgfn-root", default=None)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--max-reactions", type=int, default=1)
    parser.add_argument("--fragments-per-group", type=int, default=8)
    parser.add_argument("--max-partial-trajectories", type=int, default=100000)
    return parser


def enumerate_target(
    rgfn_root: Path,
    *,
    beta: float,
    max_reactions: int,
    fragments_per_group: int,
    max_partial_trajectories: int = 100000,
) -> dict:
    os.environ.setdefault("DGLBACKEND", "pytorch")
    os.environ.setdefault("RGFN_MINIMAL_PROXIES", "1")
    for path in (str(REPO_ROOT), str(rgfn_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from rdkit.Chem.QED import qed
    from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal
    from rgfn.gfns.reaction_gfn.reaction_env import ReactionEnv

    from molecule_synthesis.minichem import MiniReactionDataFactory

    factory = MiniReactionDataFactory(
        reaction_path=rgfn_root / "data" / "chemistry.xlsx",
        reaction_families=("Amide synthesis",),
        fragment_groups=("Acids", "Amines"),
        fragments_per_group=fragments_per_group,
        docking=False,
    )
    env = ReactionEnv(data_factory=factory, max_num_reactions=max_reactions)
    stack = [(env.sample_source_states(1)[0], 0)]
    terminal_route_counts: Counter[str] = Counter()
    terminal_molecules = {}
    n_partial = 0
    n_early_terminal = 0
    max_action_depth = 0

    while stack:
        state, depth = stack.pop()
        n_partial += 1
        if n_partial > max_partial_trajectories:
            raise RuntimeError(
                "MiniChem enumeration exceeded max_partial_trajectories; "
                "reduce the vocabulary or raise the explicit safety limit"
            )
        max_action_depth = max(max_action_depth, depth)
        if env.get_terminal_mask([state])[0]:
            if isinstance(state, ReactionStateTerminal):
                smiles = state.molecule.smiles
                terminal_route_counts[smiles] += 1
                terminal_molecules[smiles] = state.molecule
            else:
                n_early_terminal += 1
            continue

        action_space = env.get_forward_action_spaces([state])[0]
        for action_idx in action_space.get_possible_actions_indices():
            action = action_space.get_action_at_idx(action_idx)
            next_state = env.apply_forward_actions([state], [action])[0]
            stack.append((next_state, depth + 1))

    outcomes = []
    for smiles in sorted(terminal_route_counts):
        score = float(qed(terminal_molecules[smiles].rdkit_mol))
        reward = math.exp(beta * score)
        outcomes.append(
            {
                "smiles": smiles,
                "qed": score,
                "reward": reward,
                "trajectory_count": terminal_route_counts[smiles],
            }
        )
    normalizer = sum(row["reward"] for row in outcomes)
    for row in outcomes:
        row["target_probability"] = row["reward"] / normalizer

    route_counts = [row["trajectory_count"] for row in outcomes]
    return {
        "schema_version": 1,
        "name": "minichem_exact",
        "reward": {"proxy": "QED", "transform": "exp(beta * QED)", "beta": beta},
        "space": {
            "reaction_families": ["Amide synthesis"],
            "fragment_groups": ["Acids", "Amines"],
            "fragments_per_group": fragments_per_group,
            "n_fragments": len(factory.fragments),
            "n_reactions": len(factory.reactions),
            "n_anchored_reactions": len(factory.anchored_reactions),
            "max_reactions": max_reactions,
            "n_terminal_outcomes": len(outcomes),
            "n_terminal_trajectories": sum(route_counts),
            "n_multi_route_outcomes": sum(count > 1 for count in route_counts),
            "max_routes_per_outcome": max(route_counts, default=0),
            "n_early_terminal_trajectories": n_early_terminal,
            "n_partial_trajectories": n_partial,
            "max_action_depth": max_action_depth,
        },
        "outcomes": outcomes,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rgfn_root = resolve_rgfn_root(args.rgfn_root)
    validate_rgfn_root(rgfn_root)
    result = enumerate_target(
        rgfn_root,
        beta=args.beta,
        max_reactions=args.max_reactions,
        fragments_per_group=args.fragments_per_group,
        max_partial_trajectories=args.max_partial_trajectories,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["space"], indent=2, sort_keys=True))
    print(f"TARGET_DISTRIBUTION={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
